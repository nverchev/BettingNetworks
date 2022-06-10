import numpy as np


class NoSchedule:

    def __call__(self, base_lr, epoch):
        return base_lr

    def __repr__(self):
        return "NoSchedule"


class ExponentialSchedule:
    def __init__(self, exp_decay=.975):
        self.exp_decay = exp_decay

    def __call__(self, base_lr, epoch):
        return base_lr * self.exp_decay ** epoch

    def __repr__(self):
        return "ExponentialSchedule"


class CosineSchedule:
    def __init__(self, decay_steps=60, min_decay=0.1):
        self.decay_steps = decay_steps
        self.min_decay = min_decay

    def __call__(self, base_lr, epoch):
        min_lr = self.min_decay * base_lr
        return min_lr + (base_lr - min_lr) * (1 + np.cos(np.pi * epoch / self.decay_steps) / 2)

    def __repr__(self):
        return "CosineSchedule"


