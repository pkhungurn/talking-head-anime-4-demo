from abc import ABC, abstractmethod
from typing import Callable, Optional

from torch import Tensor

from tha4.shion.core.cached_computation import ComputationState


class Loss(ABC):
    @abstractmethod
    def compute(
            self,
            state: ComputationState,
            log_func: Optional[Callable[[str, float], None]] = None) -> Tensor:
        pass
