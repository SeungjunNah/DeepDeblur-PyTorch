import os
from tqdm import tqdm

import torch

import data.common
from utils import interact, MultiSaver

import torch.cuda.amp as amp

class Trainer():

    def __init__(self, args, model, criterion, optimizer, loaders):
        print('===> Initializing trainer')
        self.args = args
        self.mode = 'train' # 'val', 'test'
        self.epoch = args.start_epoch
        self.save_dir = args.save_dir

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loaders = loaders

        self.do_train = args.do_train
        self.do_validate = args.do_validate
        self.do_test = args.do_test

        self.device = args.device
        self.dtype = args.dtype
        self.dtype_eval = torch.float32 if args.precision == 'single' else torch.float16

        if self.args.demo and self.args.demo_output_dir:
            self.result_dir = self.args.demo_output_dir
        else:
            self.result_dir = os.path.join(self.save_dir, 'result')
        os.makedirs(self.result_dir, exist_ok=True)
        print('results are saved in {}'.format(self.result_dir))

        self.imsaver = MultiSaver(self.result_dir)

        self.is_slave = self.args.launched and self.args.rank != 0

        self.scaler = amp.GradScaler(
            init_scale=self.args.init_scale,
            enabled=self.args.amp
        )

    def save(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        if epoch % self.args.save_every == 0:
            if self.mode == 'train':
                self.model.save(epoch)
                self.optimizer.save(epoch)
            self.criterion.save()

        return

    def load(self, epoch=None, pretrained=None):
        if epoch is None:
            epoch = self.args.load_epoch
        self.epoch = epoch
        self.model.load(epoch, pretrained)
        self.optimizer.load(epoch)
        self.criterion.load(epoch)

        return

    def train(self, epoch):
        self.mode = 'train'
        self.epoch = epoch

        self.model.train()
        self.model.to(dtype=self.dtype)

        self.criterion.train()
        self.criterion.epoch = epoch

        if not self.is_slave:
            print('[Epoch {} / lr {:.2e}]'.format(
                epoch, self.optimizer.get_lr()
            ))

        if self.args.distributed:
            self.loaders[self.mode].sampler.set_epoch(epoch)
        if self.is_slave:
            tq = self.loaders[self.mode]
        else:
            tq = tqdm(self.loaders[self.mode], ncols=80, smoothing=0, bar_format='{desc}|{bar}{r_bar}')

        torch.set_grad_enabled(True)
        for idx, batch in enumerate(tq):
            self.optimizer.zero_grad()

            input, target = data.common.to(
                batch[0], batch[1], device=self.device, dtype=self.dtype)

            with amp.autocast(self.args.amp):
                output = self.model(input)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer.G)
            self.scaler.update()

            if isinstance(tq, tqdm):
                tq.set_description(self.criterion.get_loss_desc())

        self.criterion.normalize()
        if isinstance(tq, tqdm):
            tq.set_description(self.criterion.get_loss_desc())
            tq.display(pos=-1)  # overwrite with synchronized loss

        self.criterion.step()
        self.optimizer.schedule(self.criterion.get_last_loss())

        if self.args.rank == 0:
            self.save(epoch)

        return

    def evaluate(self, epoch, mode='val'):
        self.mode = mode
        self.epoch = epoch

        self.model.eval()
        self.model.to(dtype=self.dtype_eval)

        if mode == 'val':
            self.criterion.validate()
        elif mode == 'test':
            self.criterion.test()
        self.criterion.epoch = epoch

        self.imsaver.join_background()

        if self.is_slave:
            tq = self.loaders[self.mode]
        else:
            tq = tqdm(self.loaders[self.mode], ncols=80, smoothing=0, bar_format='{desc}|{bar}{r_bar}')

        compute_loss = True
        torch.set_grad_enabled(False)
        for idx, batch in enumerate(tq):
            input, target = data.common.to(
                batch[0], batch[1], device=self.device, dtype=self.dtype_eval)
            with amp.autocast(self.args.amp):
                output = self.model(input)

            if mode == 'demo':  # remove padded part
                pad_width = batch[2]
                output[0], _ = data.common.pad(output[0], pad_width=pad_width, negative=True)

            if isinstance(batch[1], torch.BoolTensor):
                compute_loss = False

            if compute_loss:
                self.criterion(output, target)
                if isinstance(tq, tqdm):
                    tq.set_description(self.criterion.get_loss_desc())

            if self.args.save_results != 'none':
                if isinstance(output, (list, tuple)):
                    result = output[0]  # select last output in a pyramid
                elif isinstance(output, torch.Tensor):
                    result = output

                names = batch[-1]

                if self.args.save_results == 'part' and compute_loss: # save all when GT not available
                    indices = batch[-2]
                    save_ids = [save_id for save_id, idx in enumerate(indices) if idx % 10 == 0]

                    result = result[save_ids]
                    names = [names[save_id] for save_id in save_ids]

                self.imsaver.save_image(result, names)

        if compute_loss:
            self.criterion.normalize()
            if isinstance(tq, tqdm):
                tq.set_description(self.criterion.get_loss_desc())
                tq.display(pos=-1)  # overwrite with synchronized loss

            self.criterion.step()
            if self.args.rank == 0:
                self.save()

        self.imsaver.end_background()

    def validate(self, epoch):
        self.evaluate(epoch, 'val')
        return

    def test(self, epoch):
        self.evaluate(epoch, 'test')
        return

    def fill_evaluation(self, epoch, mode=None, force=False):
        if epoch <= 0:
            return

        if mode is not None:
            self.mode = mode

        do_eval = force
        if not force:
            loss_missing = epoch not in self.criterion.loss_stat[self.mode]['Total']    # should it switch to all loss types?

            metric_missing = False
            for metric_type in self.criterion.metric:
                if epoch not in self.criterion.metric_stat[mode][metric_type]:
                    metric_missing = True

            do_eval = loss_missing or metric_missing

        if do_eval:
            try:
                self.load(epoch)
                self.evaluate(epoch, self.mode)
            except:
                # print('saved model/optimizer at epoch {} not found!'.format(epoch))
                pass

        return
