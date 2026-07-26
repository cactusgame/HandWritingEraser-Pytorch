from torch.optim.lr_scheduler import _LRScheduler, StepLR


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


class WarmupPolyLR(_LRScheduler):
    """Linear warmup followed by polynomial decay, stepped once per update."""

    def __init__(self, optimizer, max_iters, warmup_iters=500, power=0.9,
                 min_lr=1e-6, last_epoch=-1):
        self.max_iters = max(1, max_iters)
        self.warmup_iters = max(0, warmup_iters)
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        if self.warmup_iters and step < self.warmup_iters:
            factor = float(step + 1) / self.warmup_iters
        else:
            decay_steps = max(1, self.max_iters - self.warmup_iters)
            progress = min(1.0, (step - self.warmup_iters) / decay_steps)
            factor = (1.0 - progress) ** self.power
        return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]
