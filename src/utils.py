import readline
import rlcompleter
readline.parse_and_bind("tab: complete")
import code
import pdb

import time
import argparse
import os
import imageio
import torch
import torch.multiprocessing as mp

# debugging tools
def interact(local=None):
    """interactive console with autocomplete function. Useful for debugging.
    interact(locals())
    """
    if local is None:
        local=dict(globals(), **locals())

    readline.set_completer(rlcompleter.Completer(local).complete)
    code.interact(local=local)

def set_trace(local=None):
    """debugging with pdb
    """
    if local is None:
        local=dict(globals(), **locals())

    pdb.Pdb.complete = rlcompleter.Completer(local).complete
    pdb.set_trace()

# timer
class Timer():
    """Brought from https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


# argument parser type casting functions
def str2bool(val):
    """enable default constant true arguments"""
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(val, bool):
        return val
    elif val.lower() == 'true':
        return True
    elif val.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def int2str(val):
    """convert int to str for environment variable related arguments"""
    if isinstance(val, int):
        return str(val)
    elif isinstance(val, str):
        return val
    else:
        raise argparse.ArgumentTypeError('number value expected')


# image saver using multiprocessing queue
class MultiSaver():
    def __init__(self, result_dir=None):
        self.queue = None
        self.process = None
        self.result_dir = result_dir

    def begin_background(self):
        self.queue = mp.Queue()

        def t(queue):
            while True:
                if queue.empty():
                    continue
                img, name = queue.get()
                if name:
                    try:
                        basename, ext = os.path.splitext(name)
                        if ext != '.png':
                            name = '{}.png'.format(basename)
                        imageio.imwrite(name, img)
                    except Exception as e:
                        print(e)
                else:
                    return

        worker = lambda: mp.Process(target=t, args=(self.queue,), daemon=False)
        cpu_count = min(8, mp.cpu_count() - 1)
        self.process = [worker() for _ in range(cpu_count)]
        for p in self.process:
            p.start()

    def end_background(self):
        if self.queue is None:
            return

        for _ in self.process:
            self.queue.put((None, None))

    def join_background(self):
        if self.queue is None:
            return

        while not self.queue.empty():
            time.sleep(0.5)

        for p in self.process:
            p.join()

        self.queue = None

    def save_image(self, output, save_names, result_dir=None):
        result_dir = result_dir if self.result_dir is None else self.result_dir
        if result_dir is None:
            raise Exception('no result dir specified!')

        if self.queue is None:
            try:
                self.begin_background()
            except Exception as e:
                print(e)
                return

        # assume NCHW format
        if output.ndim == 2:
            output = output.expand([1, 1] + list(output.shape))
        elif output.ndim == 3:
            output = output.expand([1] + list(output.shape))

        for output_img, save_name in zip(output, save_names):
            # assume image range [0, 255]
            output_img = output_img.add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

            save_name = os.path.join(result_dir, save_name)
            save_dir = os.path.dirname(save_name)
            os.makedirs(save_dir, exist_ok=True)

            self.queue.put((output_img, save_name))

        return

class Map(dict):
    """
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    def toDict(self):
        return self.__dict__