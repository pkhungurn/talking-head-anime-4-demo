from typing import Callable, Optional

from tha4.shion.core.cached_computation import TensorCachedComputationFunc, ComputationState
from tha4.shion.core.loss import Loss


class L2Loss(Loss):
    def __init__(self,
                 expected_func: TensorCachedComputationFunc,
                 actual_func: TensorCachedComputationFunc,
                 weight: float = 1.0):
        self.actual_func = actual_func
        self.expected_func = expected_func
        self.weight = weight

    def compute(
            self,
            state: ComputationState,
            log_func: Optional[Callable[[str, float], None]] = None):
        expected = self.expected_func(state)
        actual = self.actual_func(state)
        loss = self.weight * ((expected - actual) ** 2).mean()
        if log_func is not None:
            log_func("loss", loss.item())
        return loss
