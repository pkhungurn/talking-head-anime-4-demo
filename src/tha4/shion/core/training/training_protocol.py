from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any, Optional

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from tha4.shion.core.loss import Loss
from tha4.shion.core.optimizer_factory import OptimizerFactory


class TrainingProtocol(ABC):
    @abstractmethod
    def get_optimizer_factories(self) -> Dict[str, OptimizerFactory]:
        pass

    @abstractmethod
    def get_checkpoint_examples(self) -> List[int]:
        pass

    @abstractmethod
    def get_random_seed(self) -> int:
        pass

    @abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abstractmethod
    def get_learning_rate(self, examples_seen_so_far: int) -> Dict[str, float]:
        pass

    @abstractmethod
    def run_training_iteration(
            self,
            batch: Any,
            examples_seen_so_far: int,
            modules: Dict[str, Module],
            accumulated_modules: Dict[str, Module],
            optimizers: Dict[str, Optimizer],
            losses: Dict[str, Loss],
            create_log_func: Optional[Callable[[str, int], Callable[[str, float], None]]],
            device: torch.device):
        pass


class AbstractTrainingProtocol(TrainingProtocol, ABC):
    def __init__(self,
                 check_point_examples: List[int],
                 batch_size: int,
                 learning_rate: Callable[[int], Dict[str, float]],
                 optimizer_factories: Dict[str, OptimizerFactory],
                 random_seed: int):
        self.random_seed = random_seed
        self.optimizer_factories = optimizer_factories
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.check_point_examples = check_point_examples

    def get_optimizer_factories(self) -> Dict[str, OptimizerFactory]:
        return self.optimizer_factories

    def get_checkpoint_examples(self) -> List[int]:
        return self.check_point_examples

    def get_random_seed(self) -> int:
        return self.random_seed

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_learning_rate(self, examples_seen_so_far: int) -> Dict[str, float]:
        return self.learning_rate(examples_seen_so_far)
