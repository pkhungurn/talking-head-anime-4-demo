from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
from torch.nn import Module
from torch.utils.data import Dataset


class SampleOutputProtocol(ABC):
    @abstractmethod
    def get_examples_per_sample_output(self) -> int:
        pass

    @abstractmethod
    def get_random_seed(self) -> int:
        pass

    @abstractmethod
    def get_sample_output_data(self, validation_dataset: Dataset, device: torch.device) -> Any:
        pass

    @abstractmethod
    def save_sample_output_data(
            self,
            modules: Dict[str, Module],
            accumulated_modules: Dict[str, Module],
            sample_output_data: Any,
            prefix: str,
            examples_seen_so_far: int,
            device: torch.device):
        pass


class AbstractSampleOutputProtocol(SampleOutputProtocol, ABC):
    def __init__(self, examples_per_sample_output: int, random_seed: int):
        self.random_seed = random_seed
        self.examples_per_sample_output = examples_per_sample_output

    def get_examples_per_sample_output(self) -> int:
        return self.examples_per_sample_output

    def get_random_seed(self) -> int:
        return self.random_seed
