from data.dataset import Dataset

from utils import interact

class REDS(Dataset):
    """REDS train, val, test subset class
    """
    def __init__(self, args, mode='train'):
        super(REDS, self).__init__(args, mode)

    def set_modes(self):
        self.modes = ('train', 'val', 'test')

    def set_keys(self):
        super(REDS, self).set_keys()
        # self.blur_key = 'blur'
        # self.sharp_key = 'sharp'

        self.non_blur_keys = ['blur', 'blur_comp', 'blur_bicubic']
        self.non_blur_keys.remove(self.blur_key)
        self.non_sharp_keys = ['sharp_bicubic', 'sharp']
        self.non_sharp_keys.remove(self.sharp_key)

    def __getitem__(self, idx):
        blur, sharp, pad_width, idx, relpath = super(REDS, self).__getitem__(idx)
        relpath = relpath.replace('{}/{}/'.format(self.mode, self.blur_key), '')

        return blur, sharp, pad_width, idx, relpath
