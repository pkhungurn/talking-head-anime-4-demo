from typing import Callable, Optional

from tha4.shion.core.cached_computation import TensorCachedComputationFunc, ComputationState
from tha4.shion.core.loss import Loss


class ComputedScaledL2Loss(Loss):
    def __init__(self,
                 expected_func: TensorCachedComputationFunc,
                 actual_func: TensorCachedComputationFunc,
                 element_scale_func: TensorCachedComputationFunc,
                 weight: float = 1.0):
        self.element_scale_func = element_scale_func
        self.actual_func = actual_func
        self.expected_func = expected_func
        self.weight = weight

    def compute(
            self,
            state: ComputationState,
            log_func: Optional[Callable[[str, float], None]] = None):
        element_scale = self.element_scale_func(state)
        expected = self.expected_func(state)
        actual = self.actual_func(state)
        diff = (expected - actual) * element_scale
        loss = self.weight * (diff ** 2).mean()
        if log_func is not None:
            log_func("loss", loss.item())
        return loss
