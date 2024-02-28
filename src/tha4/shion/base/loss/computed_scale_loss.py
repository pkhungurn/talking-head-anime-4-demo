from typing import Optional, Callable

from tha4.shion.core.cached_computation import TensorCachedComputationFunc, ComputationState
from tha4.shion.core.loss import Loss


class ComputedScaleLoss(Loss):
    def __init__(self,
                 scale_func: TensorCachedComputationFunc,
                 loss: Loss,
                 weight: float = 1.0):
        self.weight = weight
        self.loss = loss
        self.scale_func = scale_func

    def compute(self, state: ComputationState, log_func: Optional[Callable[[str, float], None]] = None):
        loss = self.loss.compute(state)
        scale = self.scale_func(state)
        loss = self.weight * scale * loss
        if log_func is not None:
            log_func("loss", loss.item())
        return loss
