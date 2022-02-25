import math

from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithLinearWarmup(_LRScheduler):
    def __init__(
            self, optimizer, warmup_epoch,
            T_max, adjust_epoch=False, eta_min=0,
            last_epoch=-1, verbose=False):
        self.warmup_epoch = warmup_epoch
        self.T_max = T_max
        self.eta_min = eta_min
        self.adjust_epoch = adjust_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_epoch: # linear warmup
            rate = self.last_epoch / self.warmup_epoch
            return [lr * rate for lr in self.base_lrs]
        else: # cosine annealing
            cur_epoch = self.last_epoch - self.warmup_epoch
            if self.adjust_epoch:
                max_epoch = self.T_max - self.warmup_epoch
            else:
                max_epoch = self.T_max
            rate = (1 + math.cos(cur_epoch / max_epoch * math.pi))
            return [self.eta_min + 0.5 * (lr - self.eta_min) * rate for lr in self.base_lrs]


class MultiStepLRWithLinearWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_epoch, milestones, gamma, adjust_epoch=False, last_epoch=-1, verbose=False):
        self.warmup_epoch = warmup_epoch
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.adjust_epoch = adjust_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_epoch: # linear warmup
            rate = self.last_epoch / self.warmup_epoch
            return [lr * rate for lr in self.base_lrs]
        else: # multi step lr decay
            if self.adjust_epoch:
                cur_epoch = self.last_epoch - self.warmup_epoch
            else:
                cur_epoch = self.last_epoch
            if cur_epoch not in self.milestones:
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma ** self.milestones[cur_epoch]
                            for group in self.optimizer.param_groups]
