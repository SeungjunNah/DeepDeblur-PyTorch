import os
from importlib import import_module

import torch
from torch import nn
import torch.distributed as dist

import matplotlib.pyplot as plt
plt.switch_backend('agg')	# https://github.com/matplotlib/matplotlib/issues/3466

from .metric import PSNR, SSIM

from utils import interact

class Loss(torch.nn.modules.loss._Loss):
    def __init__(self, args, epoch=None, model=None, optimizer=None):
        """
            input:
                args.loss       use '+' to sum over different loss functions
                                use '*' to specify the loss weight

                example:
                    1*MSE+0.5*VGG54
                                loss = sum of MSE and VGG54(weight=0.5)

                args.measure    similar to args.loss, but without weight

                example:
                    MSE+PSNR
                                measure MSE and PSNR, independently
        """
        super(Loss, self).__init__()

        self.args = args

        self.rgb_range = args.rgb_range
        self.device_type = args.device_type
        self.synchronized = False

        self.epoch = args.start_epoch if epoch is None else epoch
        self.save_dir = args.save_dir
        self.save_name = os.path.join(self.save_dir, 'loss.pt')

        # self.training = True
        self.validating = False
        self.testing = False
        self.mode = 'train'
        self.modes = ('train', 'val', 'test')

        # Loss
        self.loss = nn.ModuleDict()
        self.loss_types = []
        self.weight = {}

        self.loss_stat = {mode:{} for mode in self.modes}
        # loss_stat[mode][loss_type][epoch] = loss_value
        # loss_stat[mode]['Total'][epoch] = loss_total

        for weighted_loss in args.loss.split('+'):
            w, l = weighted_loss.split('*')
            l = l.upper()
            if l in ('ABS', 'L1'):
                loss_type = 'L1'
                func = nn.L1Loss()
            elif l in ('MSE', 'L2'):
                loss_type = 'L2'
                func = nn.MSELoss()
            elif l in ('ADV', 'GAN'):
                loss_type = 'ADV'
                m = import_module('loss.adversarial')
                func = getattr(m, 'Adversarial')(args, model, optimizer)
            else:
                loss_type = l
                m = import_module*'loss.{}'.format(l.lower())
                func = getattr(m, l)(args)

            self.loss_types += [loss_type]
            self.loss[loss_type] = func
            self.weight[loss_type] = float(w)

        print('Loss function: {}'.format(args.loss))

        # Metrics
        self.do_measure = args.metric.lower() != 'none'

        self.metric = nn.ModuleDict()
        self.metric_types = []
        self.metric_stat = {mode:{} for mode in self.modes}
        # metric_stat[mode][metric_type][epoch] = metric_value

        if self.do_measure:
            for metric_type in args.metric.split(','):
                metric_type = metric_type.upper()
                if metric_type == 'PSNR':
                    metric_func = PSNR()
                elif metric_type == 'SSIM':
                    metric_func = SSIM(args.device_type)    # single precision
                else:
                    raise NotImplementedError

                self.metric_types += [metric_type]
                self.metric[metric_type] = metric_func

        print('Metrics: {}'.format(args.metric))

        if args.start_epoch != 1:
            self.load(args.start_epoch - 1)

        for mode in self.modes:
            for loss_type in self.loss:
                if loss_type not in self.loss_stat[mode]:
                    self.loss_stat[mode][loss_type] = {}   # initialize loss

            if 'Total' not in self.loss_stat[mode]:
                self.loss_stat[mode]['Total'] = {}

            if self.do_measure:
                for metric_type in self.metric:
                    if metric_type not in self.metric_stat[mode]:
                        self.metric_stat[mode][metric_type] = {}

        self.count = 0
        self.count_m = 0

        self.to(args.device, dtype=args.dtype)

    def train(self, mode=True):
        super(Loss, self).train(mode)
        if mode:
            self.validating = False
            self.testing = False
            self.mode = 'train'
        else:   # default test mode
            self.validating = False
            self.testing = True
            self.mode = 'test'

    def validate(self):
        super(Loss, self).eval()
        # self.training = False
        self.validating = True
        self.testing = False
        self.mode = 'val'

    def test(self):
        super(Loss, self).eval()
        # self.training = False
        self.validating = False
        self.testing = True
        self.mode = 'test'

    def forward(self, input, target):
        self.synchronized = False

        loss = 0

        def _ms_forward(input, target, func):
            if isinstance(input, (list, tuple)): # loss for list output
                _loss = []
                for (input_i, target_i) in zip(input, target):
                    _loss += [func(input_i, target_i)]
                return sum(_loss)
            elif isinstance(input, dict):   # loss for dict output
                _loss = []
                for key in input:
                    _loss += [func(input[key], target[key])]
                return sum(_loss)
            else:   # loss for tensor output
                return func(input, target)

        # initialize
        if self.count == 0:
            for loss_type in self.loss_types:
                self.loss_stat[self.mode][loss_type][self.epoch] = 0
            self.loss_stat[self.mode]['Total'][self.epoch] = 0

        if isinstance(input, list):
            count = input[0].shape[0]
        else:   # Tensor
            count = input.shape[0]  # batch size

        isnan = False
        for loss_type in self.loss_types:

            if loss_type == 'ADV':
                _loss = self.loss[loss_type](input[0], target[0], self.training) * self.weight[loss_type]
            else:
                _loss = _ms_forward(input, target, self.loss[loss_type]) * self.weight[loss_type]

            if torch.isnan(_loss):
                isnan = True    # skip recording (will also be skipped at backprop)
            else:
                self.loss_stat[self.mode][loss_type][self.epoch] += _loss.item() * count
                self.loss_stat[self.mode]['Total'][self.epoch] += _loss.item() * count

            loss += _loss

        if not isnan:
            self.count += count

        if not self.training and self.do_measure:
            self.measure(input, target)

        return loss

    def measure(self, input, target):
        if isinstance(input, (list, tuple)):
            self.measure(input[0], target[0])
            return
        elif isinstance(input, dict):
            first_key = list(input.keys())[0]
            self.measure(input[first_key], target[first_key])
            return
        else:
            pass

        if self.count_m == 0:
            for metric_type in self.metric_stat[self.mode]:
                self.metric_stat[self.mode][metric_type][self.epoch] = 0

        if isinstance(input, list):
            count = input[0].shape[0]
        else:   # Tensor
            count = input.shape[0]  # batch size

        for metric_type in self.metric_stat[self.mode]:

            input = input.clamp(0, self.rgb_range)  # not in_place
            if self.rgb_range == 255:
                input.round_()

            _metric = self.metric[metric_type](input, target)
            self.metric_stat[self.mode][metric_type][self.epoch] += _metric.item() * count

        self.count_m += count

        return

    def normalize(self):
        if self.args.distributed:
            dist.barrier()
            if not self.synchronized:
                self.all_reduce()

        if self.count > 0:
            for loss_type in self.loss_stat[self.mode]: # including 'Total'
                self.loss_stat[self.mode][loss_type][self.epoch] /= self.count
            self.count = 0

        if self.count_m > 0:
            for metric_type in self.metric_stat[self.mode]:
                self.metric_stat[self.mode][metric_type][self.epoch] /= self.count_m
            self.count_m = 0

        return

    def all_reduce(self, epoch=None):
        # synchronize loss for distributed GPU processes

        if epoch is None:
            epoch = self.epoch

        def _reduce_value(value, ReduceOp=dist.ReduceOp.SUM):
            value_tensor = torch.Tensor([value]).to(self.args.device, self.args.dtype, non_blocking=True)
            dist.all_reduce(value_tensor, ReduceOp, async_op=False)
            value = value_tensor.item()
            del value_tensor

            return value

        dist.barrier()
        if self.count > 0:  # I assume this should be true
            self.count = _reduce_value(self.count, dist.ReduceOp.SUM)

            for loss_type in self.loss_stat[self.mode]:
                self.loss_stat[self.mode][loss_type][epoch] = _reduce_value(
                    self.loss_stat[self.mode][loss_type][epoch],
                    dist.ReduceOp.SUM
                )

        if self.count_m > 0:
            self.count_m = _reduce_value(self.count_m, dist.ReduceOp.SUM)

            for metric_type in self.metric_stat[self.mode]:
                self.metric_stat[self.mode][metric_type][epoch] = _reduce_value(
                    self.metric_stat[self.mode][metric_type][epoch],
                    dist.ReduceOp.SUM
                )

        self.synchronized = True

        return

    def print_metrics(self):

        print(self.get_metric_desc())
        return

    def get_last_loss(self):
        return self.loss_stat[self.mode]['Total'][self.epoch]

    def get_loss_desc(self):

        if self.mode == 'train':
            desc_prefix = 'Train'
        elif self.mode == 'val':
            desc_prefix = 'Validation'
        else:
            desc_prefix = 'Test'

        loss = self.loss_stat[self.mode]['Total'][self.epoch]
        if self.count > 0:
            loss /= self.count
        desc = '{} Loss: {:.1f}'.format(desc_prefix, loss)

        if self.mode in ('val', 'test'):
            metric_desc = self.get_metric_desc()
            desc = '{}{}'.format(desc, metric_desc)

        return desc

    def get_metric_desc(self):
        desc = ''
        for metric_type in self.metric_stat[self.mode]:
            measured = self.metric_stat[self.mode][metric_type][self.epoch]
            if self.count_m > 0:
                measured /= self.count_m

            if metric_type == 'PSNR':
                desc += ' {}: {:2.2f}'.format(metric_type, measured)
            elif metric_type == 'SSIM':
                desc += ' {}: {:1.4f}'.format(metric_type, measured)
            else:
                desc += ' {}: {:2.4f}'.format(metric_type, measured)

        return desc

    def step(self, plot_name=None):
        self.normalize()
        self.plot(plot_name)
        if not self.training and self.do_measure:
            # self.print_metrics()
            self.plot_metric()
        # self.epoch += 1

        return

    def save(self):

        state = {
            'loss_stat': self.loss_stat,
            'metric_stat': self.metric_stat,
        }
        torch.save(state, self.save_name)

        return

    def load(self, epoch=None):

        print('Loading loss record from {}'.format(self.save_name))
        if os.path.exists(self.save_name):
            state = torch.load(self.save_name, map_location=self.args.device)

            self.loss_stat = state['loss_stat']
            if 'metric_stat' in state:
                self.metric_stat = state['metric_stat']
            else:
                pass
        else:
            print('no loss record found for {}!'.format(self.save_name))

        if epoch is not None:
            self.epoch = epoch

        return

    def plot(self, plot_name=None, metric=False):

        self.plot_loss(plot_name)

        if metric:
            self.plot_metric(plot_name)
        # else:
        #     self.plot_loss(plot_name)

        return


    def plot_loss(self, plot_name=None):
        if plot_name is None:
            plot_name = os.path.join(self.save_dir, "{}_loss.pdf".format(self.mode))

        title = "{} loss".format(self.mode)

        fig = plt.figure()
        plt.title(title)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid(True, linestyle=':')

        for loss_type, loss_record in self.loss_stat[self.mode].items(): # including Total
            axis = sorted([epoch for epoch in loss_record.keys() if epoch <= self.epoch])
            value = [self.loss_stat[self.mode][loss_type][epoch] for epoch in axis]
            label = loss_type

            plt.plot(axis, value, label=label)

        plt.xlim(0, self.epoch)
        plt.legend()
        plt.savefig(plot_name)
        plt.close(fig)

        return

    def plot_metric(self, plot_name=None):
        # assume there are only max 2 metrics
        if plot_name is None:
            plot_name = os.path.join(self.save_dir, "{}_metric.pdf".format(self.mode))

        title = "{} metrics".format(self.mode)

        fig, ax1 = plt.subplots()
        plt.title(title)
        plt.grid(True, linestyle=':')
        ax1.set_xlabel('epochs')

        plots = None
        for metric_type, metric_record in self.metric_stat[self.mode].items():
            axis = sorted([epoch for epoch in metric_record.keys() if epoch <= self.epoch])
            value = [metric_record[epoch] for epoch in axis]
            label = metric_type

            if metric_type == 'PSNR':
                ax = ax1
                color='C0'
            elif metric_type == 'SSIM':
                ax2 = ax1.twinx()
                ax = ax2
                color='C1'

            ax.set_ylabel(metric_type)
            if plots is None:
                plots = ax.plot(axis, value, label=label, color=color)
            else:
                plots += ax.plot(axis, value, label=label, color=color)

        labels = [plot.get_label() for plot in plots]
        plt.legend(plots, labels)
        plt.xlim(0, self.epoch)
        plt.savefig(plot_name)
        plt.close(fig)

        return

    def sort(self):
        # sort the loss/metric record
        for mode in self.modes:
            for loss_type, loss_epochs in self.loss_stat[mode].items():
                self.loss_stat[mode][loss_type] = {epoch: loss_epochs[epoch] for epoch in sorted(loss_epochs)}

            for metric_type, metric_epochs in self.metric_stat[mode].items():
                self.metric_stat[mode][metric_type] = {epoch: metric_epochs[epoch] for epoch in sorted(metric_epochs)}

        return self
