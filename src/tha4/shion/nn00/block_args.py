from typing import Optional

from torch.nn import Module, Sequential

from tha4.shion.core.module_factory import ModuleFactory
from tha4.shion.nn00.linear_module_args import LinearModuleArgs
from tha4.shion.nn00.nonlinearity_factories import resolve_nonlinearity_factory
from tha4.shion.nn00.normalization_layer_factories import resolve_normalization_layer_factory
from tha4.shion.nn00.normalization_layer_factory import NormalizationLayerFactory


class BlockArgs:
    def __init__(
            self,
            linear_module_args: Optional[LinearModuleArgs] = None,
            normalization_layer_factory: Optional[NormalizationLayerFactory] = None,
            nonlinearity_factory: Optional[ModuleFactory] = None):
        if linear_module_args is None:
            linear_module_args = LinearModuleArgs()
        self.nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
        self.normalization_layer_factory = resolve_normalization_layer_factory(normalization_layer_factory)
        self.linear_module_args = linear_module_args