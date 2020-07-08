import os
import re
from importlib import import_module

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .discriminator import Discriminator

from utils import interact, Map

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
        self.load(args.loadEpoch, path=args.pretrained)

    def parallelize(self):
        if self.args.distributed:
            self.model.G = DistributedDataParallel(self.model.G, device_ids=[self.args.rank], output_device=self.args.rank)
            if self.model.D is not None:
                self.model.D = DistributedDataParallel(self.model.D, device_ids=[self.args.rank], output_device=self.args.rank)

        else:
            self.model.G = DataParallel(self.model.G, range(self.n_GPUs))
            if self.model.D is not None:
                self.model.D = DataParallel(self.model.D, range(self.n_GPUs))

    def forward(self, input):
        return self.model.G(input)

    def _save_path(self, epoch):
        model_path = os.path.join(self.save_dir, 'model-{:d}.pt'.format(epoch))
        return model_path

    def state_dict(self):
        state_dict = Map()

        if isinstance(self.model.G, DataParallel) or isinstance(self.model.G, DistributedDataParallel):
            state_dict.G = self.model.G.module.state_dict()
            if self.model.D is not None:
                state_dict.D = self.model.D.module.state_dict()
        else:
            state_dict.G = self.model.G.state_dict()
            if self.model.D is not None:
                state_dict.D = self.model.D.state_dict()

        return state_dict.toDict()

    def load_state_dict(self, state_dict):
        state_dict = Map(**state_dict)
        if isinstance(self.model.G, DataParallel) or isinstance(self.model.G, DistributedDataParallel):
            self.model.G.module.load_state_dict(state_dict.G)
            if self.model.D is not None:
                self.model.D.module.load_state_dict(state_dict.D)
        else:
            self.model.G.load_state_dict(state_dict.G)
            if self.model.D is not None:
                self.model.D.load_state_dict(state_dict.D)

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
