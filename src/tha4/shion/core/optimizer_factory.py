from abc import ABC, abstractmethod
from typing import Iterable

from torch.nn import Parameter


class OptimizerFactory(ABC):
    @abstractmethod
    def create(self, parameters: Iterable[Parameter]):
        pass
