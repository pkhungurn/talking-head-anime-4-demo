from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional

import torch
from torch import Tensor
from torch.nn import Module


class ComputationState:
    def __init__(self,
                 modules: Dict[str, Module],
                 accumulated_modules: Dict[str, Module],
                 batch: Any,
                 outputs: Optional[Dict[str, Any]] = None):
        if outputs is None:
            outputs = {}
        self.outputs = outputs
        self.batch = batch
        self.accumulated_modules = accumulated_modules
        self.modules = modules


CachedComputationFunc = Callable[[ComputationState], Any]
TensorCachedComputationFunc = Callable[[ComputationState], Tensor]


def create_get_item_func(func: CachedComputationFunc, index):
    def _f(state: ComputationState):
        output = func(state)
        return output[index]

    return _f


def create_batch_element_func(index: int) -> TensorCachedComputationFunc:
    def _f(state: ComputationState) -> Tensor:
        return state.batch[index]

    return _f


class CachedComputationProtocol(ABC):
    def get_output(self, key: str, state: ComputationState) -> Any:
        if key in state.outputs:
            return state.outputs[key]
        else:
            output = self.compute_output(key, state)
            state.outputs[key] = output
            return state.outputs[key]

    @abstractmethod
    def compute_output(self, key: str, state: ComputationState) -> Any:
        pass

    def get_output_func(self, key: str) -> CachedComputationFunc:
        def func(state: ComputationState):
            return self.get_output(key, state)

        return func


ComposableCachedComputationStep = Callable[[CachedComputationProtocol, ComputationState], Any]


class ComposableCachedComputationProtocol(CachedComputationProtocol):
    def __init__(self, computation_steps: Optional[Dict[str, ComposableCachedComputationStep]] = None):
        if computation_steps is None:
            computation_steps = {}
        self.computation_steps = computation_steps

    def compute_output(self, key: str, state: ComputationState) -> Any:
        if key in self.computation_steps:
            return self.computation_steps[key](self, state)
        else:
            raise RuntimeError("Computing output for key " + key + " is not supported!")


def batch_indexing_func(index: int):
    def _f(protocol: CachedComputationProtocol, state: ComputationState):
        return state.batch[index]

    return _f


def proxy_func(key: str):
    def _f(protocol: CachedComputationProtocol, state: ComputationState):
        return protocol.get_output(key, state)

    return _f


def output_array_indexing_func(key: str, index: int):
    def _f(protocol: CachedComputationProtocol, state: ComputationState):
        return protocol.get_output(key, state)[index]

    return _f


def add_step(step_dict: Dict[str, ComposableCachedComputationStep], name: str):
    def _f(func):
        step_dict[name] = func
        return func

    return _f


def zeros_like_func(key: str):
    def _f(protocol: CachedComputationProtocol, state: ComputationState):
        prototype = protocol.get_output(key, state)
        return torch.zeros_like(prototype)

    return _f
