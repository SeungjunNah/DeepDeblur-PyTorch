import os
import re
from importlib import import_module

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .discriminator import Discriminator

from utils import interact

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.device = args.device
        self.n_GPUs = args.n_GPUs
        self.save_dir = os.path.join(args.save_dir, 'models')
        os.makedirs(self.save_dir, exist_ok=True)

        module = import_module('model.' + args.model)

        self.model = nn.ModuleDict()
        self.model.G = module.build_model(args)
        if self.args.loss.lower().find('adv') >= 0:
            self.model.D = Discriminator(self.args)
        else:
            self.model.D = None

        self.to(args.device, dtype=args.dtype, non_blocking=True)
        self.load(args.load_epoch, path=args.pretrained)

    def parallelize(self):
        if self.args.device_type == 'cuda':
            if self.args.distributed:
                Parallel = DistributedDataParallel
                parallel_args = {
                    "device_ids": [self.args.rank],
                    "output_device": self.args.rank,
                }
            else:
                Parallel = DataParallel
                parallel_args = {
                    'device_ids': list(range(self.n_GPUs)),
                    'output_device': self.args.rank # always 0
                }

            for model_key in self.model:
                if self.model[model_key] is not None:
                    self.model[model_key] = Parallel(self.model[model_key], **parallel_args)

    def forward(self, input):
        return self.model.G(input)

    def _save_path(self, epoch):
        model_path = os.path.join(self.save_dir, 'model-{:d}.pt'.format(epoch))
        return model_path

    def state_dict(self):
        state_dict = {}
        for model_key in self.model:
            if self.model[model_key] is not None:
                parallelized = isinstance(self.model[model_key], (DataParallel, DistributedDataParallel))
                if parallelized:
                    state_dict[model_key] = self.model[model_key].module.state_dict()
                else:
                    state_dict[model_key] = self.model[model_key].state_dict()

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        for model_key in self.model:
            parallelized = isinstance(self.model[model_key], (DataParallel, DistributedDataParallel))
            if model_key in state_dict:
                if parallelized:
                    self.model[model_key].module.load_state_dict(state_dict[model_key], strict)
                else:
                    self.model[model_key].load_state_dict(state_dict[model_key], strict)

    def save(self, epoch):
        torch.save(self.state_dict(), self._save_path(epoch))

    def load(self, epoch=None, path=None):
        if path:
            model_name = path
        elif isinstance(epoch, int):
            if epoch < 0:
                epoch = self.get_last_epoch()
            if epoch == 0:   # epoch 0
                # make sure model parameters are synchronized at initial
                # for multi-node training (not in current implementation)
                # self.synchronize()

                return  # leave model as initialized

            model_name = self._save_path(epoch)
        else:
            raise Exception('no epoch number or model path specified!')

        print('Loading model from {}'.format(model_name))
        state_dict = torch.load(model_name, map_location=self.args.device)
        self.load_state_dict(state_dict)

        return

    def synchronize(self):
        if self.args.distributed:
            # synchronize model parameters across nodes
            vector = parameters_to_vector(self.parameters())

            dist.broadcast(vector, 0)   # broadcast parameters to other processes
            if self.args.rank != 0:
                vector_to_parameters(vector, self.parameters())

            del vector

        return

    def get_last_epoch(self):
        model_list = sorted(os.listdir(self.save_dir))
        if len(model_list) == 0:
            epoch = 0
        else:
            epoch = int(re.findall('\\d+', model_list[-1])[0]) # model example name model-100.pt

        return epoch

    def print(self):
        print(self.model)

        return
