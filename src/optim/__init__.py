import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import os
from collections import Counter

from model import Model
from utils import interact, Map

class Optimizer(object):
    def __init__(self, args, model):
        self.args = args

        self.save_dir = os.path.join(self.args.save_dir, 'optim')
        os.makedirs(self.save_dir, exist_ok=True)

        if isinstance(model, Model):
            model = model.model

        # set base arguments
        kwargs_optimizer = {
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }

        if args.optimizer == 'SGD':
            optimizer_class = optim.SGD
            kwargs_optimizer['momentum'] = args.momentum
        elif args.optimizer == 'ADAM':
            optimizer_class = optim.Adam
            kwargs_optimizer['betas'] = args.betas
            kwargs_optimizer['eps'] = args.epsilon
        elif args.optimizer == 'RMSPROP':
            optimizer_class = optim.RMSprop
            kwargs_optimizer['eps'] = args.epsilon

        # scheduler
        if args.scheduler == 'step':
            scheduler_class = lrs.MultiStepLR
            kwargs_scheduler = {
                'milestones': args.milestones,
                'gamma': args.gamma,
            }
        elif args.scheduler == 'plateau':
            scheduler_class = lrs.ReduceLROnPlateau
            kwargs_scheduler = {
                'mode': 'min',
                'factor': args.gamma,
                'patience': 10,
                'verbose': True,
                'threshold': 0,
                'threshold_mode': 'abs',
                'cooldown': 10,
            }

        self.kwargs_optimizer = kwargs_optimizer
        self.scheduler_class = scheduler_class
        self.kwargs_scheduler = kwargs_scheduler

        def _get_optimizer(model):

            class _Optimizer(optimizer_class):
                def __init__(self, model, args, scheduler_class, kwargs_scheduler):
                    trainable = filter(lambda x: x.requires_grad, model.parameters())
                    super(_Optimizer, self).__init__(trainable, **kwargs_optimizer)

                    self.args = args

                    self._register_scheduler(scheduler_class, kwargs_scheduler)

                def _register_scheduler(self, scheduler_class, kwargs_scheduler):
                    self.scheduler = scheduler_class(self, **kwargs_scheduler)

                def schedule(self, metrics=None):
                    if isinstance(self, lrs.ReduceLROnPlateau):
                        self.scheduler.step(metrics)
                    else:
                        self.scheduler.step()

                def get_last_epoch(self):
                    return self.scheduler.last_epoch

                def get_lr(self):
                    return self.param_groups[0]['lr']

                def get_last_lr(self):
                    return self.scheduler.get_last_lr()[0]

                def state_dict(self):
                    state_dict = super(_Optimizer, self).state_dict()  # {'state': ..., 'param_groups': ...}
                    state_dict['scheduler'] = self.scheduler.state_dict()

                    return state_dict

                def load_state_dict(self, state_dict, epoch=None):
                    # optimizer
                    super(_Optimizer, self).load_state_dict(state_dict)  # load 'state' and 'param_groups' only
                    # scheduler
                    self.scheduler.load_state_dict(state_dict['scheduler']) # should work for plateau or simple resuming

                    reschedule = False
                    if isinstance(self.scheduler, lrs.MultiStepLR):
                        if self.args.milestones != list(self.scheduler.milestones) or self.args.gamma != self.scheduler.gamma:
                            reschedule = True

                    if reschedule:
                        if epoch is None:
                            if self.scheduler.last_epoch > 1:
                                epoch = self.scheduler.last_epoch
                            else:
                                epoch = self.args.start_epoch - 1

                        # if False:
                        #     # option 1. new scheduler
                        #     for i, group in enumerate(self.param_groups):
                        #         self.param_groups[i]['lr'] = group['initial_lr']    # reset optimizer learning rate to initial
                        #     # self.scheduler = None
                        #     self._register_scheduler(scheduler_class, kwargs_scheduler)

                        #     self.zero_grad()
                        #     self.step()
                        #     for _ in range(epoch):
                        #         self.scheduler.step()
                        #     self._step_count -= 1

                        # else:
                        # option 2. modify existing scheduler
                        self.scheduler.milestones = Counter(self.args.milestones)
                        self.scheduler.gamma = self.args.gamma
                        for i, group in enumerate(self.param_groups):
                            self.param_groups[i]['lr'] = group['initial_lr']    # reset optimizer learning rate to initial
                            multiplier = 1
                            for milestone in self.scheduler.milestones:
                                if epoch >= milestone:
                                    multiplier *= self.scheduler.gamma

                            self.param_groups[i]['lr'] *= multiplier

            return _Optimizer(model, args, scheduler_class, kwargs_scheduler)

        self.G = _get_optimizer(model.G)
        if model.D is not None:
            self.D = _get_optimizer(model.D)
        else:
            self.D = None

        self.load(args.load_epoch)

    def zero_grad(self):
        self.G.zero_grad()

    def step(self):
        self.G.step()

    def schedule(self, metrics=None):
        self.G.schedule(metrics)
        if self.D is not None:
            self.D.schedule(metrics)

    def get_last_epoch(self):
        return self.G.get_last_epoch()

    def get_lr(self):
        return self.G.get_lr()

    def get_last_lr(self):
        return self.G.get_last_lr()

    def state_dict(self):
        state_dict = Map()
        state_dict.G = self.G.state_dict()
        if self.D is not None:
            state_dict.D = self.D.state_dict()

        return state_dict.toDict()

    def load_state_dict(self, state_dict, epoch=None):
        state_dict = Map(**state_dict)
        self.G.load_state_dict(state_dict.G, epoch)
        if self.D is not None:
            self.D.load_state_dict(state_dict.D, epoch)

    def _save_path(self, epoch=None):
        epoch = epoch if epoch is not None else self.get_last_epoch()
        save_path = os.path.join(self.save_dir, 'optim-{:d}.pt'.format(epoch))

        return save_path

    def save(self, epoch=None):
        if epoch is None:
            epoch = self.G.scheduler.last_epoch
        torch.save(self.state_dict(), self._save_path(epoch))

    def load(self, epoch):
        if epoch > 0:
            print('Loading optimizer from {}'.format(self._save_path(epoch)))
            self.load_state_dict(torch.load(self._save_path(epoch), map_location=self.args.device), epoch=epoch)

        elif epoch == 0:
            pass
        else:
            raise NotImplementedError

        return

