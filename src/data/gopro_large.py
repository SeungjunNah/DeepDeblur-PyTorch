from data.dataset import Dataset

from utils import interact

class GOPRO_Large(Dataset):
    """GOPRO_Large train, test subset class
    """
    def __init__(self, args, mode='train'):
        super(GOPRO_Large, self).__init__(args, mode)

    def set_modes(self):
        self.modes = ('train', 'test')

    def set_keys(self):
        super(GOPRO_Large, self).set_keys()
        self.blur_key = 'blur_gamma'
        # self.sharp_key = 'sharp'

    def __getitem__(self, idx):
        blur, sharp, pad_width, idx, relpath = super(GOPRO_Large, self).__getitem__(idx)
        relpath = relpath.replace('{}/'.format(self.blur_key), '')

        return blur, sharp, pad_width, idx, relpath
