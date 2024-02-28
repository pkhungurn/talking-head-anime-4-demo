from abc import ABC, abstractmethod

from torch.nn import Module


class NormalizationLayerFactory(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self, num_features: int, affine: bool = True) -> Module:
        pass
