""" distributed launcher adopted from torch.distributed.launch
    usage example: https://github.com/facebookresearch/maskrcnn-benchmark
    This enables using multiprocessing for each spawned process (as they are treated as main processes)
"""
import sys
import subprocess
from argparse import ArgumentParser, REMAINDER

from utils import str2bool, int2str

def parse_args():
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")


    parser.add_argument('--n_GPUs', type=int, default=1, help='the number of GPUs for training')

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

def main():
    args = parse_args()

    processes = []
    for rank in range(0, args.n_GPUs):
        cmd = [sys.executable]

        cmd.append(args.training_script)
        cmd.extend(args.training_script_args)

        cmd += ['--distributed', 'True']
        cmd += ['--launched', 'True']
        cmd += ['--n_GPUs', str(args.n_GPUs)]
        cmd += ['--rank', str(rank)]

        process = subprocess.Popen(cmd)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)

if __name__ == "__main__":
    main()
