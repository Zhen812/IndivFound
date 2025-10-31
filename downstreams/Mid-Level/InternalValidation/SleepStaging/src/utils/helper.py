import os
import numpy as np
import torch
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    # torch.backends.cudnn.benchmark = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # 训练集变化不大时使训练加速
