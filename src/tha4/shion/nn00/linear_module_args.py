from typing import Optional, Callable

from torch.nn import Module
from torch.nn.utils import spectral_norm

from tha4.shion.nn00.initialization_funcs import resolve_initialization_func


class LinearModuleArgs:
    def __init__(
            self,
            initialization_func: Optional[Callable[[Module], Module]] = None,
            use_spectral_norm: bool = False):
        self.use_spectral_norm = use_spectral_norm
        self.initialization_func = resolve_initialization_func(initialization_func)

    def wrap_linear_module(self, module: Module) -> Module:
        module = self.initialization_func(module)
        if self.use_spectral_norm:
            module = spectral_norm(module)
        return module


def wrap_linear_module(module: Module, linear_module_args: Optional[LinearModuleArgs] = None):
    if linear_module_args is None:
        linear_module_args = LinearModuleArgs()
    module = linear_module_args.initialization_func(module)
    if linear_module_args.use_spectral_norm:
        module = spectral_norm(module)
    return module
