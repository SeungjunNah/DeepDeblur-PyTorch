"""optionional argument parsing"""
# pylint: disable=C0103, C0301
import argparse
import datetime
import os
import re
import shutil
import time

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from utils import interact
from utils import str2bool, int2str

import template

# Training settings
parser = argparse.ArgumentParser(description='Dynamic Scene Deblurring')

# Device specifications
group_device = parser.add_argument_group('Device specs')
group_device.add_argument('--seed', type=int, default=-1, help='random seed')
group_device.add_argument('--num_workers', type=int, default=7, help='the number of dataloader workers')
group_device.add_argument('--device_type', type=str, choices=('cpu', 'cuda'), default='cuda', help='device to run models')
group_device.add_argument('--device_index', type=int, default=0, help='device id to run models')
group_device.add_argument('--n_GPUs', type=int, default=1, help='the number of GPUs for training')
group_device.add_argument('--distributed', type=str2bool, default=False, help='use DistributedDataParallel instead of DataParallel for better speed')
group_device.add_argument('--launched', type=str2bool, default=False, help='identify if main.py was executed from launch.py. Do not set this to be true using main.py.')

group_device.add_argument('--master_addr', type=str, default='127.0.0.1', help='master address for distributed')
group_device.add_argument('--master_port', type=int2str, default='8023', help='master port for distributed')
group_device.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
group_device.add_argument('--init_method', type=str, default='env://', help='distributed init method URL to discover peers')
group_device.add_argument('--rank', type=int, default=0, help='rank of the distributed process (gpu id). 0 is the master process.')
group_device.add_argument('--world_size', type=int, default=1, help='world_size for distributed training (number of GPUs)')

# Data
group_data = parser.add_argument_group('Data specs')
group_data.add_argument('--data_root', type=str, default='~/Research/dataset', help='dataset root location')
group_data.add_argument('--dataset', type=str, default=None, help='training/validation/test dataset name, has priority if not None')
group_data.add_argument('--data_train', type=str, default='GOPRO_Large', help='training dataset name')
group_data.add_argument('--data_val', type=str, default=None, help='validation dataset name')
group_data.add_argument('--data_test', type=str, default='GOPRO_Large', help='test dataset name')
group_data.add_argument('--blur_key', type=str, default='blur_gamma', choices=('blur', 'blur_gamma'), help='blur type from camera response function for GOPRO_Large dataset')
group_data.add_argument('--rgb_range', type=int, default=255, help='RGB pixel value ranging from 0')

# Model
group_model = parser.add_argument_group('Model specs')
group_model.add_argument('--model', type=str, default='MSResNet', help='model architecture')
group_model.add_argument('--pretrained', type=str, default='', help='pretrained model location')
group_model.add_argument('--n_scales', type=int, default=3, help='multi-scale deblurring level')
group_model.add_argument('--gaussian_pyramid', type=str2bool, default=True, help='gaussian pyramid input/target')
group_model.add_argument('--n_resblocks', type=int, default=19, help='number of residual blocks per scale')
group_model.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
group_model.add_argument('--kernel_size', type=int, default=5, help='size of conv kernel')
group_model.add_argument('--downsample', type=str, choices=('Gaussian', 'bicubic', 'stride'), default='Gaussian', help='input pyramid generation method')

group_model.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test(single | half)')

# amp
group_amp = parser.add_argument_group('AMP specs')
group_amp.add_argument('--amp', type=str2bool, default=False, help='use automatic mixed precision training')
group_amp.add_argument('--init_scale', type=float, default=1024., help='initial loss scale')

# Training
group_train = parser.add_argument_group('Training specs')
group_train.add_argument('--patch_size', type=int, default=256, help='training patch size')
group_train.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
group_train.add_argument('--split_batch', type=int, default=1, help='split a minibatch into smaller chunks')
group_train.add_argument('--augment', type=str2bool, default=True, help='train with data augmentation')

# Testing
group_test = parser.add_argument_group('Testing specs')
group_test.add_argument('--validate_every', type=int, default=10, help='do validation at every N epochs')
group_test.add_argument('--test_every', type=int, default=10, help='do test at every N epochs')
# group_test.add_argument('--chop', type=str2bool, default=False, help='memory-efficient forward')
# group_test.add_argument('--self_ensemble', type=str2bool, default=False, help='self-ensembled testing')

# Action
group_action = parser.add_argument_group('Source behavior')
group_action.add_argument('--do_train', type=str2bool, default=True, help='do train the model')
group_action.add_argument('--do_validate', type=str2bool, default=True, help='do validate the model')
group_action.add_argument('--do_test', type=str2bool, default=True, help='do test the model')
group_action.add_argument('--demo', type=str2bool, default=False, help='demo')
group_action.add_argument('--demo_input_dir', type=str, default='', help='demo input directory')
group_action.add_argument('--demo_output_dir', type=str, default='', help='demo output directory')

# Optimization
group_optim = parser.add_argument_group('Optimization specs')
group_optim.add_argument('--lr', type=float, default=1e-4, help='learning rate')
group_optim.add_argument('--milestones', type=int, nargs='+', default=[500, 750, 900], help='learning rate decay per N epochs')
group_optim.add_argument('--scheduler', default='step', choices=('step', 'plateau'), help='learning rate scheduler type')
group_optim.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
group_optim.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | RMSProp)')
group_optim.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
group_optim.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='ADAM betas')
group_optim.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon')
group_optim.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Loss
group_loss = parser.add_argument_group('Loss specs')
group_loss.add_argument('--loss', type=str, default='1*L1', help='loss function configuration')
group_loss.add_argument('--metric', type=str, default='PSNR,SSIM', help='metric function configuration. ex) None | PSNR | SSIM | PSNR,SSIM')

# Logging
group_log = parser.add_argument_group('Logging specs')
group_log.add_argument('--save_dir', type=str, default='', help='subdirectory to save experiment logs')
# group_log.add_argument('--load_dir', type=str, default='', help='subdirectory to load experiment logs')
group_log.add_argument('--start_epoch', type=int, default=-1, help='(re)starting epoch number')
group_log.add_argument('--end_epoch', type=int, default=1000, help='ending epoch number')
group_log.add_argument('--load_epoch', type=int, default=-1, help='epoch number to load model (start_epoch-1 for training, start_epoch for testing)')
group_log.add_argument('--save_every', type=int, default=10, help='save model/optimizer at every N epochs')
group_log.add_argument('--save_results', type=str, default='part', choices=('none', 'part', 'all'), help='save none/part/all of result images')

# Debugging
group_debug = parser.add_argument_group('Debug specs')
group_debug.add_argument('--stay', type=str2bool, default=False, help='stay at interactive console after trainer initialization')

parser.add_argument('--template', type=str, default='', help='argument template option')

args = parser.parse_args()
template.set_template(args)

args.data_root = os.path.expanduser(args.data_root)   # recognize home directory
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if args.save_dir == '':
    args.save_dir = now
args.save_dir = os.path.join('../experiment', args.save_dir)
os.makedirs(args.save_dir, exist_ok=True)

if args.start_epoch < 0: # start from scratch or continue from the last epoch
    # check if there are any models saved before
    model_dir = os.path.join(args.save_dir, 'models')
    model_prefix = 'model-'
    if os.path.exists(model_dir):
        model_list = [name for name in os.listdir(model_dir) if name.startswith(model_prefix)]
        last_epoch = 0
        for name in model_list:
            epochNumber = int(re.findall('\\d+', name)[0]) # model example name model-100.pt
            if last_epoch < epochNumber:
                last_epoch = epochNumber

        args.start_epoch = last_epoch + 1
    else:
        # train from scratch
        args.start_epoch = 1
elif args.start_epoch == 0:
    # remove existing directory and start over
    if args.rank == 0:  # maybe local rank
        shutil.rmtree(args.save_dir, ignore_errors=True)
    os.makedirs(args.save_dir, exist_ok=True)
    args.start_epoch = 1

if args.load_epoch < 0:  # load_epoch == start_epoch when doing a post-training test for a specific epoch
    args.load_epoch = args.start_epoch - 1

if args.pretrained:
    if args.start_epoch <= 1:
        args.pretrained = os.path.join('../experiment', args.pretrained)
    else:
        print('starting from epoch {}! ignoring pretrained model path..'.format(args.start_epoch))
        args.pretrained = ''

if args.model == 'MSResNet':
    args.gaussian_pyramid = True

argname = os.path.join(args.save_dir, 'args.pt')
argname_txt = os.path.join(args.save_dir, 'args.txt')
if args.start_epoch > 1:
    # load previous arguments and keep the necessary ones same

    if os.path.exists(argname):
        args_old = torch.load(argname)

        load_list = []  # list of arguments that are fixed
        # training
        load_list += ['patch_size']
        load_list += ['batch_size']
        # data format
        load_list += ['rgb_range']
        load_list += ['blur_key']
        # model architecture
        load_list += ['n_scales']
        load_list += ['n_resblocks']
        load_list += ['n_feats']

        for arg_part in load_list:
            vars(args)[arg_part] = vars(args_old)[arg_part]

if args.dataset is not None:
    args.data_train = args.dataset
    args.data_val = args.dataset if args.dataset != 'GOPRO_Large' else None
    args.data_test = args.dataset

if args.data_val is None:
    args.do_validate = False

if args.demo_input_dir:
    args.demo = True

if args.demo:
    assert os.path.basename(args.save_dir) != now, 'You should specify pretrained directory by setting --save_dir SAVE_DIR'

    args.data_train = ''
    args.data_val = ''
    args.data_test = ''

    args.do_train = False
    args.do_validate = False
    args.do_test = False

    assert len(args.demo_input_dir) > 0, 'Please specify demo_input_dir!'
    args.demo_input_dir = os.path.expanduser(args.demo_input_dir)
    if args.demo_output_dir:
        args.demo_output_dir = os.path.expanduser(args.demo_output_dir)

    args.save_results = 'all'

if args.amp:
    args.precision = 'single'   # model parameters should stay in fp32

if args.seed < 0:
    args.seed = int(time.time())

# save arguments
if args.rank == 0:
    torch.save(args, argname)
    with open(argname_txt, 'a') as file:
        file.write('execution at {}\n'.format(now))

        for key in args.__dict__:
            file.write(key + ': ' + str(args.__dict__[key]) + '\n')

        file.write('\n')

# device and type
if args.device_type == 'cuda' and not torch.cuda.is_available():
    raise Exception("GPU not available!")

if not args.distributed:
    args.rank = 0

def setup(args):
    cudnn.benchmark = True

    if args.distributed:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port

        args.device_index = args.rank
        args.world_size = args.n_GPUs   # consider single-node training

        # initialize the process group
        dist.init_process_group(args.dist_backend, init_method=args.init_method, rank=args.rank, world_size=args.world_size)

    args.device = torch.device(args.device_type, args.device_index)
    args.dtype = torch.float32
    args.dtype_eval = torch.float32 if args.precision == 'single' else torch.float16

    # set seed for processes (distributed: different seed for each process)
    # model parameters are synchronized explicitly at initial
    torch.manual_seed(args.seed)
    if args.device_type == 'cuda':
        torch.cuda.set_device(args.device)
        if args.rank == 0:
            torch.cuda.manual_seed_all(args.seed)

    return args

def cleanup(args):
    if args.distributed:
        dist.destroy_process_group()
