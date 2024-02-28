from typing import Callable, Optional

import torch
from torch import zero_
from torch.nn import Module
from torch.nn.init import kaiming_normal_, xavier_normal_, normal_


class HeInitialization:
    def __init__(self, a: int = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
        self.nonlinearity = nonlinearity
        self.mode = mode
        self.a = a

    def __call__(self, module: Module) -> Module:
        with torch.no_grad():
            kaiming_normal_(module.weight, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity)
        return module


class NormalInitialization:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.std = std
        self.mean = mean

    def __call__(self, module: Module) -> Module:
        with torch.no_grad():
            normal_(module.weight, self.mean, self.std)
        return module


class XavierInitialization:
    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, module: Module) -> Module:
        with torch.no_grad():
            xavier_normal_(module.weight, self.gain)
        return module


class ZeroInitialization:
    def __call__(self, module: Module) -> Module:
        with torch.no_grad:
            zero_(module.weight)
        return module


class NoInitialization:
    def __call__(self, module: Module) -> Module:
        return module


def resolve_initialization_func(initialization: Optional[Callable[[Module], Module]]):
    if initialization is None:
        return NoInitialization()
    else:
        return initialization
