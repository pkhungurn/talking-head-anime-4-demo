from abc import ABC, abstractmethod
from typing import Optional

from torch.nn import Module


class ModuleAccumulator(ABC):
    @abstractmethod
    def accumulate(self, module: Module, output: Module, examples_seen_so_far: Optional[int] = None) -> Module:
        pass
