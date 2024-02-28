from typing import Optional

import torch
from torch.nn import Module, Parameter, BatchNorm2d, InstanceNorm2d, GroupNorm
from torch.nn.functional import layer_norm
from torch.nn.init import normal_, constant_

from tha4.shion.nn00.normalization_layer_factory import NormalizationLayerFactory
from tha4.shion.nn00.pass_through import PassThrough


class Bias2d(Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.bias = Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return x + self.bias


class NoNorm2dFactory(NormalizationLayerFactory):
    def __init__(self):
        super().__init__()

    def create(self, num_features: int, affine: bool = True) -> Module:
        if affine:
            return Bias2d(num_features)
        else:
            return PassThrough()


class BatchNorm2dFactory(NormalizationLayerFactory):
    def __init__(self,
                 weight_mean: Optional[float] = None,
                 weight_std: Optional[float] = None,
                 bias: Optional[float] = None):
        super().__init__()
        self.bias = bias
        self.weight_std = weight_std
        self.weight_mean = weight_mean

    def get_weight_mean(self):
        if self.weight_mean is None:
            return 1.0
        else:
            return self.weight_mean

    def get_weight_std(self):
        if self.weight_std is None:
            return 0.02
        else:
            return self.weight_std

    def create(self, num_features: int, affine: bool = True) -> Module:
        module = BatchNorm2d(num_features=num_features, affine=affine)
        if affine:
            if self.weight_mean is not None or self.weight_std is not None:
                normal_(module.weight, self.get_weight_mean(), self.get_weight_std())
            if self.bias is not None:
                constant_(module.bias, self.bias)
        return module


class InstanceNorm2dFactory(NormalizationLayerFactory):
    def __init__(self):
        super().__init__()

    def create(self, num_features: int, affine: bool = True) -> Module:
        return InstanceNorm2d(num_features=num_features, affine=affine)


class LayerNorm2d(Module):
    def __init__(self, channels: int, affine: bool = True):
        super(LayerNorm2d, self).__init__()
        self.channels = channels
        self.affine = affine

        if self.affine:
            self.weight = Parameter(torch.ones(1, channels, 1, 1))
            self.bias = Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        shape = x.size()[1:]
        y = layer_norm(x, shape) * self.weight + self.bias
        return y


class LayerNorm2dFactory(NormalizationLayerFactory):
    def __init__(self):
        super().__init__()

    def create(self, num_features: int, affine: bool = True) -> Module:
        return LayerNorm2d(channels=num_features, affine=affine)


class GroupNormFactory(NormalizationLayerFactory):
    def __init__(self, num_groups: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

    def create(self, num_features: int, affine: bool = True) -> Module:
        return GroupNorm(num_channels=num_features, num_groups=self.num_groups, eps=self.eps, affine=affine)


def resolve_normalization_layer_factory(factory: Optional['NormalizationLayerFactory']) -> 'NormalizationLayerFactory':
    if factory is None:
        return InstanceNorm2dFactory()
    else:
        return factory
