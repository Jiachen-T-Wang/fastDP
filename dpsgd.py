import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD, Adam, Adagrad, RMSprop

import numpy as np

# This implementation is adapted from https://github.com/ChrisWaites/pyvacy


def make_optimizer_class(optimizer):
    class DPOptimizerClass(optimizer):
        def __init__(self, l2_norm_clip, noise_multiplier, batch_size, minibatch_size, 
                     *args, **kwargs):
          
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.minibatch_size = minibatch_size
            self.batch_size = batch_size

            # Create accumulated gradients for each parameter group
            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_minibatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def minibatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.minibatch_size / self.batch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass

DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)
DPRMSprop = make_optimizer_class(RMSprop)

