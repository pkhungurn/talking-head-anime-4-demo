from typing import Callable, Optional

import torch

from tha4.shion.core.cached_computation import TensorCachedComputationFunc, ComputationState
from tha4.shion.core.loss import Loss


class L1Loss(Loss):
    def __init__(self,
                 expected_func: TensorCachedComputationFunc,
                 actual_func: TensorCachedComputationFunc,
                 weight: float = 1.0):
        self.actual_func = actual_func
        self.expected_func = expected_func
        self.weight = weight

    def compute(self, state: ComputationState, log_func: Optional[Callable[[str, float], None]] = None):
        expected = self.expected_func(state)
        actual = self.actual_func(state)
        loss = self.weight * (expected - actual).abs().mean()
        if log_func is not None:
            log_func("loss", loss.item())
        return loss


class ListL1Loss(Loss):
    def __init__(self,
                 expected_func: TensorCachedComputationFunc,
                 actual_func: TensorCachedComputationFunc,
                 weight: float = 1.0):
        self.actual_func = actual_func
        self.expected_func = expected_func
        self.weight = weight

    def compute(self, state: ComputationState, log_func: Optional[Callable[[str, float], None]] = None):
        expected = self.expected_func(state)
        actual = self.actual_func(state)
        assert len(expected) == len(actual)
        loss = torch.zeros(1, device=expected[0].device)
        for i in range(len(expected)):
            loss += (expected[i] - actual[i]).abs().mean()
        loss = self.weight * loss
        if log_func is not None:
            log_func("loss", loss.item())
        return loss


class MaskedL1Loss(Loss):
    def __init__(self,
                 expected_func: TensorCachedComputationFunc,
                 actual_func: TensorCachedComputationFunc,
                 mask_func: TensorCachedComputationFunc,
                 weight: float = 1.0):
        self.mask_func = mask_func
        self.actual_func = actual_func
        self.expected_func = expected_func
        self.weight = weight

    def compute(self, state: ComputationState, log_func: Optional[Callable[[str, float], None]] = None):
        mask = self.mask_func(state)
        expected = self.expected_func(state)
        actual = self.actual_func(state)
        loss = self.weight * ((expected - actual) * mask).abs().mean()
        if log_func is not None:
            log_func("loss", loss.item())
        return loss