from data.dataset import Dataset

from utils import interact

class Demo(Dataset):
    """Demo train, test subset class
    """
    def __init__(self, args, mode='demo'):
        super(Demo, self).__init__(args, mode)

    def set_modes(self):
        self.modes = ('demo')

    def set_keys(self):
        super(Demo, self).set_keys()
        self.blur_key = ''              # all the files
        self.non_sharp_keys = ['']      # no files

    def __getitem__(self, idx):
        blur, sharp, pad_width, idx, relpath = super(Demo, self).__getitem__(idx)

        return blur, sharp, pad_width, idx, relpath
