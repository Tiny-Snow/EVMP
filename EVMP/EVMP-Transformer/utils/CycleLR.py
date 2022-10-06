# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (Transformer)
# Time:     2022.5.18

from torch.optim.lr_scheduler import _LRScheduler


class CycleDecayLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs in a (cycle_size * step_size) cycle
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer       Optimizer       Wrapped optimizer.
        step_size       int             The epochs in a step
        cycle_size      int             The number of steps in a cycle
        gamma           float           Multiplicative factor of learning rate decay. (Default: 0.1)
        last_epoch      int             The index of last epoch. (Default: -1)
        verbose         bool            If ``True``, prints a message to stdout for each update. (Default: ``False``)

    Example:
        >>> # Assuming optimizer uses lr = 1 for all groups
        >>> # lr = 1        if epoch < 10
        >>> # lr = 0.1      if 10 <= epoch < 20
        >>> # lr = 0.01     if 20 <= epoch < 30
        >>> # lr = 1        if 30 <= epoch < 40
        >>> # lr = 0.1      if 40 <= epoch < 50
        >>> # ...
        >>> scheduler = MultiStepLR(optimizer, step_size = 10, cycle_size = 3, gamma = 0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, cycle_size, gamma = 0.1, last_epoch = -1, verbose = False):
        super(CycleDecayLR, self).__init__(optimizer, last_epoch, verbose)
        self.step_size = step_size
        self.cycle_size = cycle_size
        self.gamma = gamma

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        if self.last_epoch % (self.cycle_size * self.step_size) == 0:
            return [group['lr'] / self.gamma ** (self.cycle_size - 1) for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** ((self.last_epoch // self.step_size) % self.cycle)
                for base_lr in self.base_lrs]
