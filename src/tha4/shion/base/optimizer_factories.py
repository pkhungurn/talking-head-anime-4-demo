from typing import Tuple, Iterable

from torch.nn import Parameter
from torch.optim import Optimizer, Adam, AdamW, SparseAdam, RMSprop

from tha4.shion.core.optimizer_factory import OptimizerFactory


class AdamOptimizerFactory(OptimizerFactory):
    def __init__(self, betas: Tuple[float, float] = (0.9, 0.999), epsilon: float = 1e-8, weight_decay: float = 0.0):
        super().__init__()
        self.weight_decay = weight_decay
        self.betas = betas
        self.epsilon = epsilon

    def create(self, parameters: Iterable[Parameter]) -> Optimizer:
        return Adam(parameters, betas=self.betas, eps=self.epsilon, weight_decay=self.weight_decay)


class AdamWOptimizerFactory(OptimizerFactory):
    def __init__(self, betas: Tuple[float, float] = (0.9, 0.999), epsilon: float = 1e-8, weight_decay: float = 0.01):
        super().__init__()
        self.weight_decay = weight_decay
        self.betas = betas
        self.epsilon = epsilon

    def create(self, parameters: Iterable[Parameter]) -> Optimizer:
        return AdamW(parameters, betas=self.betas, eps=self.epsilon, weight_decay=self.weight_decay)


class SparseAdamOptimizerFactory(OptimizerFactory):
    def __init__(self, betas: Tuple[float, float] = (0.9, 0.999), epsilon: float = 1e-8):
        super().__init__()
        self.betas = betas
        self.epsilon = epsilon

    def create(self, parameters: Iterable[Parameter]) -> Optimizer:
        return SparseAdam(list(parameters), betas=self.betas, eps=self.epsilon)


class RMSpropOptimizerFactory(OptimizerFactory):
    def __init__(self):
        super().__init__()

    def create(self, parameters: Iterable[Parameter]) -> Optimizer:
        return RMSprop(parameters)
