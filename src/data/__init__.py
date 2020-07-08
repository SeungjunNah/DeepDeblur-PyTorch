"""Generic dataset loader"""

from importlib import import_module

from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from .sampler import DistributedEvalSampler

class Data():
    def __init__(self, args):

        self.modes = ['train', 'val', 'test', 'demo']

        self.action = {
            'train': args.do_train,
            'val':  args.do_validate,
            'test': args.do_test,
            'demo': args.demo
        }

        self.dataset_name = {
            'train': args.data_train,
            'val': args.data_val,
            'test': args.data_test,
            'demo': 'Demo'
        }

        self.args = args

        def _get_data_loader(mode='train'):
            dataset_name = self.dataset_name[mode]
            dataset = import_module('data.' + dataset_name.lower())
            dataset = getattr(dataset, dataset_name)(args, mode)

            if mode == 'train':
                if args.distributed:
                    batch_size = int(args.batch_size / args.n_GPUs)   # batch size per GPU (single-node training)
                    sampler = DistributedSampler(dataset, shuffle=True, num_replicas=args.world_size, rank=args.rank)
                    num_workers = int((args.num_workers + args.n_GPUs - 1) / args.n_GPUs)    # num_workers per GPU (single-node training)
                else:
                    batch_size = args.batch_size
                    sampler = RandomSampler(dataset, replacement=False)
                    num_workers = args.num_workers
                drop_last = True

            elif mode in ('val', 'test', 'demo'):
                if args.distributed:
                    batch_size = 1  # 1 image per GPU
                    sampler = DistributedEvalSampler(dataset, shuffle=False, num_replicas=args.world_size, rank=args.rank)
                    num_workers = int((args.num_workers + args.n_GPUs - 1) / args.n_GPUs)    # num_workers per GPU (single-node training)
                else:
                    batch_size = args.n_GPUs    # 1 image per GPU
                    sampler = SequentialSampler(dataset)
                    num_workers = args.num_workers
                drop_last = False

            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=drop_last,
            )

            return loader

        self.loaders = {}
        for mode in self.modes:
            if self.action[mode]:
                self.loaders[mode] = _get_data_loader(mode)
                print('===> Loading {} dataset: {}'.format(mode, self.dataset_name[mode]))
            else:
                self.loaders[mode] = None

    def get_loader(self):
        return self.loaders
