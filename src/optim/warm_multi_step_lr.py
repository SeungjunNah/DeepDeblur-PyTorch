import math
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

# MultiStep learning rate scheduler with warm restart
class WarmMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, scale=1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of increasing integers. Got {}',
                milestones
            )

        self.milestones = milestones
        self.gamma = gamma
        self.scale = scale

        self.warmup_epochs = 5
        self.gradual = (self.scale - 1) / self.warmup_epochs
        super(WarmMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (1 + self.last_epoch * self.gradual) / self.scale
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
