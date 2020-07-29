import os
import random
import imageio
import numpy as np
import torch.utils.data as data

from data import common

from utils import interact

class Dataset(data.Dataset):
    """Basic dataloader class
    """
    def __init__(self, args, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.mode = mode

        self.modes = ()
        self.set_modes()
        self._check_mode()

        self.set_keys()

        if self.mode == 'train':
            dataset = args.data_train
        elif self.mode == 'val':
            dataset = args.data_val
        elif self.mode == 'test':
            dataset = args.data_test
        elif self.mode == 'demo':
            pass
        else:
            raise NotImplementedError('not implemented for this mode: {}!'.format(self.mode))

        if self.mode == 'demo':
            self.subset_root = args.demo_input_dir
        else:
            self.subset_root = os.path.join(args.data_root, dataset, self.mode)

        self.blur_list = []
        self.sharp_list = []

        self._scan()

    def set_modes(self):
        self.modes = ('train', 'val', 'test', 'demo')

    def _check_mode(self):
        """Should be called in the child class __init__() after super
        """
        if self.mode not in self.modes:
            raise NotImplementedError('mode error: not for {}'.format(self.mode))

        return

    def set_keys(self):
        self.blur_key = 'blur'      # to be overwritten by child class
        self.sharp_key = 'sharp'    # to be overwritten by child class

        self.non_blur_keys = []
        self.non_sharp_keys = []

        return

    def _scan(self, root=None):
        """Should be called in the child class __init__() after super
        """
        if root is None:
            root = self.subset_root

        if self.blur_key in self.non_blur_keys:
            self.non_blur_keys.remove(self.blur_key)
        if self.sharp_key in self.non_sharp_keys:
            self.non_sharp_keys.remove(self.sharp_key)

        def _key_check(path, true_key, false_keys):
            path = os.path.join(path, '')
            if path.find(true_key) >= 0:
                for false_key in false_keys:
                    if path.find(false_key) >= 0:
                        return False

                return True
            else:
                return False

        def _get_list_by_key(root, true_key, false_keys):
            data_list = []
            for sub, dirs, files in os.walk(root):
                if not dirs:
                    file_list = [os.path.join(sub, f) for f in files]
                    if _key_check(sub, true_key, false_keys):
                        data_list += file_list

            data_list.sort()

            return data_list

        def _rectify_keys():
            self.blur_key = os.path.join(self.blur_key, '')
            self.non_blur_keys = [os.path.join(non_blur_key, '') for non_blur_key in self.non_blur_keys]
            self.sharp_key = os.path.join(self.sharp_key, '')
            self.non_sharp_keys = [os.path.join(non_sharp_key, '') for non_sharp_key in self.non_sharp_keys]

        _rectify_keys()

        self.blur_list = _get_list_by_key(root, self.blur_key, self.non_blur_keys)
        self.sharp_list = _get_list_by_key(root, self.sharp_key, self.non_sharp_keys)

        if len(self.sharp_list) > 0:
            assert(len(self.blur_list) == len(self.sharp_list))

        return

    def __getitem__(self, idx):

        blur = imageio.imread(self.blur_list[idx], pilmode='RGB')
        if len(self.sharp_list) > 0:
            sharp = imageio.imread(self.sharp_list[idx], pilmode='RGB')
            imgs = [blur, sharp]
        else:
            imgs = [blur]

        pad_width = 0   # dummy value
        if self.mode == 'train':
            imgs = common.crop(*imgs, ps=self.args.patch_size)
            if self.args.augment:
                imgs = common.augment(*imgs, hflip=True, rot=True, shuffle=True, change_saturation=True, rgb_range=self.args.rgb_range)
                imgs[0] = common.add_noise(imgs[0], sigma_sigma=2, rgb_range=self.args.rgb_range)
        elif self.mode == 'demo':
            imgs[0], pad_width = common.pad(imgs[0], divisor=2**(self.args.n_scales-1))   # pad in case of non-divisible size
        else:
            pass    # deliver test image as is.

        if self.args.gaussian_pyramid:
            imgs = common.generate_pyramid(*imgs, n_scales=self.args.n_scales)

        imgs = common.np2tensor(*imgs)
        relpath = os.path.relpath(self.blur_list[idx], self.subset_root)

        blur = imgs[0]
        sharp = imgs[1] if len(imgs) > 1 else False

        return blur, sharp, pad_width, idx, relpath

    def __len__(self):
        return len(self.blur_list)
        # return 32





