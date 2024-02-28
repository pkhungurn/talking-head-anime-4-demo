from typing import Optional

import torch
from torch.nn import Module

from tha4.shion.core.module_accumulator import ModuleAccumulator


# Code from https://github.com/rosinality/style-based-gan-pytorch/blob/8437a8bbd106ad4a4691b798ce35d30b5111990b/train.py
def accumulate_modules(new_module: Module, accumulated_module: Module, beta=0.99):
    with torch.no_grad():
        new_module_params = dict(new_module.named_parameters())
        accumulated_module_params = dict(accumulated_module.named_parameters())
        for key in new_module_params.keys():
            accumulated_module_params[key].mul_(beta).add_(new_module_params[key] * (1 - beta))

        new_module_buffers = dict(new_module.named_buffers())
        accumulated_module_buffers = dict(accumulated_module.named_buffers())
        for key in new_module_buffers.keys():
            accumulated_module_buffers[key].copy_(new_module_buffers[key])


class DecayAccumulator(ModuleAccumulator):
    def __init__(self, decay: float = 0.999):
        self.decay = decay

    def accumulate(self, module: Module, output: Module, examples_seen_so_far: Optional[int] = None) -> Module:
        accumulate_modules(module, output, self.decay)
        return output
