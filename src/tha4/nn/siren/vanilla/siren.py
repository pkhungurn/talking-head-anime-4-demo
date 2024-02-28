import math
from typing import Callable, Optional, List

import torch
from torch import Tensor
from torch.nn import Module, Conv2d, ModuleList

from tha4.shion.core.module_factory import ModuleFactory
from tha4.shion.nn00.initialization_funcs import HeInitialization


class SineLinearLayer(Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            is_first=False,
            omega_0=30.0):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_channels, 1.0 / in_channels)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6.0 / in_channels) / self.omega_0,
                    math.sqrt(6.0 / in_channels) / self.omega_0)

    def forward(self, x: Tensor):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenArgs:
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            intermediate_channels: int,
            num_sine_layers: int,
            use_tanh: bool = False,
            init_func: Optional[Callable[[Module], Module]] = None):
        if init_func is None:
            init_func = HeInitialization()
        self.init_func = init_func
        self.use_tanh = use_tanh
        assert num_sine_layers >= 1
        self.intermediate_channels = intermediate_channels
        self.num_sine_layers = num_sine_layers
        self.out_channels = out_channels
        self.in_channels = in_channels


class Siren(Module):
    def __init__(self, args: SirenArgs):
        super().__init__()
        self.args = args
        self.sine_layers = ModuleList()
        self.sine_layers.append(
            SineLinearLayer(
                in_channels=args.in_channels, out_channels=args.intermediate_channels, is_first=True))
        for i in range(args.num_sine_layers - 1):
            self.sine_layers.append(
                SineLinearLayer(
                    in_channels=args.intermediate_channels,
                    out_channels=args.intermediate_channels,
                    is_first=False))
        self.last_linear = args.init_func(Conv2d(
            args.intermediate_channels,
            args.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True))

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.args.num_sine_layers):
            x = self.sine_layers[i].forward(x)
        x = self.last_linear(x)
        if self.args.use_tanh:
            return torch.tanh(x)
        else:
            return x


class SirenFactory(ModuleFactory):
    def __init__(self, args: SirenArgs):
        super().__init__()
        self.args = args

    def create(self) -> Module:
        return Siren(self.args)
