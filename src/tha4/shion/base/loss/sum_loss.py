from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor

from tha4.shion.core.cached_computation import ComputationState
from tha4.shion.core.loss import Loss


class SumLoss(Loss):
    def __init__(self, losses: List[Tuple[str, Loss]]):
        self.losses = losses

    def compute(self,
                state: ComputationState,
                log_func: Optional[Callable[[str, float], None]] = None) -> Tensor:
        device = state.batch[0].device
        loss_value = torch.zeros(1, device=device)
        for loss_spec in self.losses:
            loss_name = loss_spec[0]
            loss = loss_spec[1]
            if log_func is not None:
                def loss_log_func(name, value):
                    log_func(loss_name + "_" + name, value)
            else:
                loss_log_func = None
            loss_value = loss_value + loss.compute(state, loss_log_func)

        if log_func is not None:
            log_func("loss", loss_value.item())

        return loss_value
