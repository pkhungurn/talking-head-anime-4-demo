from typing import List, Dict, Callable, Any, Optional

import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer

from tha4.shion.core.cached_computation import ComputationState
from tha4.shion.core.loss import Loss
from tha4.shion.core.optimizer_factory import OptimizerFactory
from tha4.shion.core.training.training_protocol import TrainingProtocol


class TwoNetworksWithMinibatchTrainingProtocol(TrainingProtocol):
    def __init__(self,
                 check_point_examples: List[int],
                 batch_size: int,
                 learning_rate: Callable[[int], Dict[str, float]],
                 optimizer_factories: Dict[str, OptimizerFactory],
                 key_network_0: str,
                 key_network_1: str,
                 train_network_0: bool = False,
                 random_seed: int = 39549059840,
                 max_grad_norm: Optional[float] = None,
                 minibatch_size: Optional[int] = None):
        super().__init__()
        if minibatch_size is None:
            minibatch_size = batch_size
        assert batch_size % minibatch_size == 0
        self.train_network_0 = train_network_0
        self.key_network_1 = key_network_1
        self.key_network_0 = key_network_0
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.optimizer_factories = optimizer_factories
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.check_point_examples = check_point_examples

    def get_optimizer_factories(self) -> Dict[str, OptimizerFactory]:
        return self.optimizer_factories

    def get_checkpoint_examples(self) -> List[int]:
        return self.check_point_examples

    def get_random_seed(self) -> int:
        return self.random_seed

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_learning_rate(self, examples_seen_so_far: int) -> Dict[str, float]:
        return self.learning_rate(examples_seen_so_far)

    def run_training_iteration(
            self,
            batch: Any,
            examples_seen_so_far: int,
            modules: Dict[str, Module],
            accumulated_modules: Dict[str, Module],
            optimizers: Dict[str, Optimizer],
            losses: Dict[str, Loss],
            create_log_func: Optional[Callable[[str, int], Callable[[str, float], None]]],
            device: torch.device):
        network_0 = modules[self.key_network_0]
        network_0.train(self.train_network_0)

        network_1 = modules[self.key_network_1]
        network_1.train(True)

        if self.train_network_0:
            optimizers[self.key_network_0].zero_grad(set_to_none=True)
        optimizers[self.key_network_1].zero_grad(set_to_none=True)

        if create_log_func is not None:
            network_0_log_func = create_log_func("training_" + self.key_network_0, examples_seen_so_far)
            network_1_log_func = create_log_func("training_" + self.key_network_1, examples_seen_so_far)
        else:
            network_0_log_func = None
            network_1_log_func = None

        num_minibatch = self.batch_size // self.minibatch_size
        for minibatch_index in range(num_minibatch):
            minibatch = []
            for item in batch:
                minibatch.append(
                    item[minibatch_index * self.minibatch_size:(minibatch_index + 1) * self.minibatch_size])
            loss = losses[self.key_network_1].compute(
                ComputationState(modules, accumulated_modules, minibatch),
                network_1_log_func if minibatch_index == 0 else None)
            if self.train_network_0 and self.key_network_0 in losses:
                loss = loss + losses[self.key_network_0].compute(
                    ComputationState(modules, accumulated_modules, minibatch),
                    network_0_log_func if minibatch_index == 0 else None)
            loss = loss / num_minibatch
            loss.backward()

        if self.max_grad_norm is not None:
            clip_grad_norm_(network_1.parameters(), self.max_grad_norm)
            if self.train_network_0:
                clip_grad_norm_(network_0.parameters(), self.max_grad_norm)

        optimizers[self.key_network_1].step()
        if self.train_network_0:
            optimizers[self.key_network_0].step()
