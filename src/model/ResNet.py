import torch.nn as nn

from . import common

def build_model(args):
    return ResNet(args)

class ResNet(nn.Module):
    def __init__(self, args, in_channels=None, out_channels=None, n_feats=None, kernel_size=None, n_resblocks=None):
        super(ResNet, self).__init__()

        self.in_channels = in_channels if isinstance(in_channels, int) else args.in_channels
        self.out_channels = out_channels if isinstance(out_channels, int) else args.out_channels
        self.n_feats = n_feats if isinstance(n_feats, int) else args.n_feats
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else args.kernel_size
        self.n_resblocks = n_resblocks if isinstance(n_resblocks, int) else args.n_resblocks


        # n_resblock = args.n_resblocks
        # n_feat = args.n_feats
        # kernel_size = args.kernel_size

        modules = []
        modules.append(common.default_conv(self.in_channels, self.n_feats, self.kernel_size))
        for _ in range(self.n_resblocks):
            modules.append(common.ResBlock(self.n_feats, self.kernel_size))
        modules.append(common.default_conv(self.n_feats, self.out_channels, self.kernel_size))

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        # x -= 0.5
        # x = self.body(x)
        # x += 0.5

        # return x

        return self.body(x)

