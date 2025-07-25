from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler


class InverseSquareRootScheduler(LRScheduler):
    def __init__(self, learning_rate: float, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate
        super(InverseSquareRootScheduler, self).__init__(learning_rate, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return self.base_lr
        scale_factor = (self.warmup_steps ** 0.5) / (step ** 0.5)
        return self.base_lr * scale_factor
