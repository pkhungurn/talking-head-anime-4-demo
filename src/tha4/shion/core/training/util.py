from typing import Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer


def optimizer_to_device(optim: Optimizer, device: torch.device):
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def zero_module(module: Module):
    parameters = dict(module.named_parameters())
    for k in parameters.keys():
        parameters[k].data.zero_()


def get_least_greater_multiple(x: int, m: int) -> int:
    """
    :param x: a non-negative integer
    :param m: a positive integer
    :return: the next multiple of m that is greater than x
    """
    assert x >= 0
    assert m > 0
    return (x // m + 1) * m


def create_log_func(summary_writer, prefix: str, examples_seen_so_far: int) -> Callable[[str, float], None]:
    def log_func(tag: str, value: float):
        summary_writer.add_scalar(prefix + "_" + tag, value, examples_seen_so_far)

    return log_func


def set_learning_rate(module, lr):
    for param_group in module.param_groups:
        param_group['lr'] = lr
