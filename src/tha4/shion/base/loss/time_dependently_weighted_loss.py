from typing import Callable, Optional

from torch import Tensor

from tha4.shion.core.cached_computation import ComputationState, CachedComputationFunc
from tha4.shion.core.loss import Loss


class TimeDependentlyWeightedLoss(Loss):
    def __init__(self,
                 base_loss: Loss,
                 examples_seen_so_far_func: CachedComputationFunc,
                 weight_func: Callable[[int], float]):
        self.weight_func = weight_func
        self.examples_seen_so_far_func = examples_seen_so_far_func
        self.base_loss = base_loss

    def compute(self,
                state: ComputationState,
                log_func: Optional[Callable[[str, float], None]] = None) -> Tensor:
        base_value = self.base_loss.compute(state)
        examples_seen_so_far = self.examples_seen_so_far_func(state)
        weight = self.weight_func(examples_seen_so_far)
        loss_value = base_value * weight

        if log_func is not None:
            log_func("loss", loss_value.item())

        return loss_value
