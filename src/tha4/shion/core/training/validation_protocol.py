from abc import ABC, abstractmethod
from typing import Dict, Callable, Any

import torch
from torch.nn import Module

from tha4.shion.core.loss import Loss


class ValidationProtocol(ABC):
    @abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abstractmethod
    def get_examples_per_validation_iteration(self) -> int:
        pass

    @abstractmethod
    def run_validation_iteration(
            self,
            batch: Any,
            examples_seen_so_far: int,
            modules: Dict[str, Module],
            accumulated_modules: Dict[str, Module],
            losses: Dict[str, Loss],
            create_log_func: Callable[[str, int], Callable[[str, float], None]],
            device: torch.device):
        pass


class AbstractValidationProtocol(ValidationProtocol, ABC):
    def __init__(self,
                 example_per_validation_iteration: int,
                 batch_size: int):
        self.batch_size = batch_size
        self.example_per_validation_iteration = example_per_validation_iteration

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_examples_per_validation_iteration(self) -> int:
        return self.example_per_validation_iteration
