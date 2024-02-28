import argparse
import logging
import os.path
import time
from datetime import datetime
from typing import Dict, Optional, Callable, Any

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tha4.shion.core.load_save import torch_save, torch_load
from tha4.shion.core.loss import Loss
from tha4.shion.core.module_accumulator import ModuleAccumulator
from tha4.shion.core.module_factory import ModuleFactory
from tha4.shion.core.training.distrib.device_mapper import SimpleCudaDeviceMapper
from tha4.shion.core.training.distrib.distributed_training_states import DistributedTrainingState
from tha4.shion.core.training.sample_output_protocol import SampleOutputProtocol
from tha4.shion.core.training.training_protocol import TrainingProtocol
from tha4.shion.core.training.util import set_learning_rate, create_log_func, get_least_greater_multiple
from tha4.shion.core.training.validation_protocol import ValidationProtocol

KEY_CHECKPOINT = 'checkpoint'
KEY_SNAPSHOT = 'snapshot'
KEY_VALIDATION = 'validation'
KEY_SAMPLE_OUTPUT = 'sample_output'


class DistributedTrainer:
    def __init__(self,
                 prefix: str,
                 module_factories: Dict[str, ModuleFactory],
                 accumulators: Dict[str, ModuleAccumulator],
                 losses: Dict[str, Loss],
                 training_dataset: Dataset,
                 validation_dataset: Optional[Dataset],
                 training_protocol: TrainingProtocol,
                 validation_protocol: Optional[ValidationProtocol],
                 sample_output_protocol: Optional[SampleOutputProtocol],
                 pretrained_module_file_names: Dict[str, str],
                 example_per_snapshot: int,
                 num_data_loader_workers: int = 8,
                 distrib_backend: str = 'gloo'):
        self.distrib_backend = distrib_backend
        self.num_data_loader_workers = num_data_loader_workers
        self.accumulators = accumulators
        self.sample_output_protocol = sample_output_protocol
        self.example_per_snapshot = example_per_snapshot
        self.pretrained_module_file_names = pretrained_module_file_names
        self.losses = losses
        self.validation_protocol = validation_protocol
        self.training_protocol = training_protocol
        self.module_factories = module_factories
        self.prefix = prefix
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.checkpoint_examples = self.training_protocol.get_checkpoint_examples()
        assert len(self.checkpoint_examples) >= 1
        assert self.checkpoint_examples[0] > 0
        self.checkpoint_examples = [0] + self.checkpoint_examples

        self.module_names = self.module_factories.keys()
        assert len(self.module_names) > 0

        self.training_data_loader = None
        self.training_data_loader_iter = None
        self.training_data_loader_batch_size = None
        self.training_data_sampler = None

        self.validation_data_loader = None
        self.validation_data_loader_iter = None
        self.validation_data_loader_batch_size = None

        self.sample_output_data = None
        self.summary_writer = None
        self.log_dir = None
        self.training_state = None

    def get_sample_output_data_file_name(self):
        return self.prefix + "/sample_output_data.pt"

    def save_sample_output_data(self, rank: int, device: torch.device):
        if rank != 0:
            return
        if os.path.exists(self.get_sample_output_data_file_name()):
            return
        if self.sample_output_protocol is not None:
            torch.manual_seed(self.sample_output_protocol.get_random_seed())
            sample_output_data = self.sample_output_protocol.get_sample_output_data(self.validation_dataset, device)
            torch_save(sample_output_data, self.get_sample_output_data_file_name())
        else:
            torch_save({}, self.get_sample_output_data_file_name())

    def load_sample_output_data(self, rank: int, device: torch.device):
        if rank != 0:
            return None
        else:
            self.save_sample_output_data(rank, device)
            return torch_load(self.get_sample_output_data_file_name())

    def get_snapshot_prefix(self) -> str:
        return self.prefix + "/snapshot"

    def can_load_training_state(self, prefix: str, world_size: int) -> bool:
        return DistributedTrainingState.can_load(
            prefix,
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories(),
            world_size)

    def load_training_state(self, prefix, rank: int, local_rank: int, device: torch.device) -> DistributedTrainingState:
        return DistributedTrainingState.load(
            prefix,
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories(),
            rank,
            local_rank,
            device)

    @staticmethod
    def checkpoint_prefix(prefix: str, checkpoint_index: int) -> str:
        return "%s/checkpoint/%04d" % (prefix, checkpoint_index)

    def get_checkpoint_prefix(self, checkpoint_index) -> str:
        return DistributedTrainer.checkpoint_prefix(self.prefix, checkpoint_index)

    def get_initial_training_state(self, rank: int, local_rank: int, device: torch.device) -> DistributedTrainingState:
        training_state = DistributedTrainingState.new(
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories(),
            self.training_protocol.get_random_seed(),
            rank,
            local_rank,
            device,
            self.pretrained_module_file_names)
        logging.info("Created a new initial training state.")
        return training_state

    def load_previous_training_state(self,
                                     target_checkpoint_examples: int,
                                     world_size: int,
                                     rank: int,
                                     local_rank: int,
                                     device: torch.device) -> DistributedTrainingState:
        if self.can_load_training_state(self.get_snapshot_prefix(), world_size):
            examples_seen_so_far = DistributedTrainingState.get_examples_seen_so_far(self.get_snapshot_prefix())
            diff = examples_seen_so_far - target_checkpoint_examples
            if diff < self.training_protocol.get_batch_size():
                return self.load_training_state(self.get_snapshot_prefix(), rank, local_rank, device)
        num_checkpoints = len(self.checkpoint_examples)
        for checkpoint_index in range(num_checkpoints - 1, -1, -1):
            if self.can_load_training_state(self.get_checkpoint_prefix(checkpoint_index), world_size):
                examples_seen_so_far = DistributedTrainingState.get_examples_seen_so_far(
                    self.get_checkpoint_prefix(checkpoint_index))
                diff = examples_seen_so_far - target_checkpoint_examples
                if diff < self.training_protocol.get_batch_size():
                    return self.load_training_state(
                        self.get_checkpoint_prefix(checkpoint_index), rank, local_rank, device)

        training_state = self.get_initial_training_state(rank, local_rank, device)
        training_state.save(self.get_checkpoint_prefix(0), rank, lambda: self.barrier(local_rank))
        training_state = self.load_training_state(self.get_checkpoint_prefix(0), rank, local_rank, device)
        return training_state

    def get_log_dir(self):
        if self.log_dir is None:
            now = datetime.now()
            self.log_dir = self.prefix + "/log/" + now.strftime("%Y_%m_%d__%H_%M_%S")
        return self.log_dir

    def get_summary_writer(self, rank: int) -> Optional[SummaryWriter]:
        if rank != 0:
            return None
        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(log_dir=self.get_log_dir())
        return self.summary_writer

    def get_effective_training_epoch_size(self, world_size: int):
        batch_size = self.training_protocol.get_batch_size()
        N = len(self.training_dataset)
        N = (N // world_size) * world_size
        N = (N // batch_size) * batch_size
        return N

    def get_training_epoch_index(self, examples_seen_so_far: int, world_size: int):
        epoch_size = self.get_effective_training_epoch_size(world_size)
        batch_size = self.training_protocol.get_batch_size()
        return (examples_seen_so_far + batch_size * world_size) // epoch_size

    def get_next_training_batch(self, examples_seen_so_far: int, world_size: int, device: torch.device):
        batch_size = self.training_protocol.get_batch_size()
        dataset = self.training_dataset
        if self.training_data_loader is None:
            self.training_data_sampler = DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=True)
            self.training_data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=self.training_data_sampler,
                shuffle=False,
                num_workers=self.num_data_loader_workers,
                drop_last=True)
        if self.training_data_loader_iter is None:
            epoch_index = self.get_training_epoch_index(examples_seen_so_far, world_size)
            logging.info(f"Started a new epoch: index = {epoch_index}, examples_seen_so_far = {examples_seen_so_far}")
            self.training_data_sampler.set_epoch(epoch_index)
            self.training_data_loader_iter = iter(self.training_data_loader)
        try:
            batch = next(self.training_data_loader_iter)
        except StopIteration:
            epoch_index = self.get_training_epoch_index(examples_seen_so_far, world_size)
            logging.info(f"Started a new epoch: index = {epoch_index}, examples_seen_so_far = {examples_seen_so_far}")
            self.training_data_sampler.set_epoch(epoch_index)
            self.training_data_loader_iter = iter(self.training_data_loader)
            batch = next(self.training_data_loader_iter)
        return [x.to(device) for x in batch]

    def get_next_checkpoint_num_examples(self, examples_seen_so_far) -> int:
        next_index = next(
            (i for i in range(len(self.checkpoint_examples)) if self.checkpoint_examples[i] > examples_seen_so_far),
            -1)
        return self.checkpoint_examples[next_index]

    def get_next_snapshot_num_examples(self, examples_seen_so_far) -> int:
        return get_least_greater_multiple(examples_seen_so_far, self.example_per_snapshot)

    def get_next_validation_num_examples(self, examples_seen_so_far) -> int:
        if self.validation_protocol is None:
            return -1
        return get_least_greater_multiple(examples_seen_so_far,
                                          self.validation_protocol.get_examples_per_validation_iteration())

    def get_next_sample_output_num_examples(self, examples_seen_so_far) -> int:
        if self.sample_output_protocol is None:
            return -1
        return get_least_greater_multiple(examples_seen_so_far,
                                          self.sample_output_protocol.get_examples_per_sample_output())

    def get_next_num_examples(self, examples_seen_so_far) -> Dict[str, int]:
        return {
            KEY_CHECKPOINT: self.get_next_checkpoint_num_examples(examples_seen_so_far),
            KEY_SNAPSHOT: self.get_next_snapshot_num_examples(examples_seen_so_far),
            KEY_VALIDATION: self.get_next_validation_num_examples(examples_seen_so_far),
            KEY_SAMPLE_OUTPUT: self.get_next_sample_output_num_examples(examples_seen_so_far)
        }

    def get_next_validation_batch(self, device: torch.device):
        if self.validation_dataset is None:
            return None
        if self.validation_data_loader is None:
            self.validation_data_loader = DataLoader(
                self.validation_dataset,
                batch_size=self.validation_protocol.get_batch_size(),
                shuffle=True,
                num_workers=1,
                drop_last=True)
        if self.validation_data_loader_iter is None:
            self.validation_data_loader_iter = iter(self.validation_data_loader)
        try:
            batch = next(self.validation_data_loader_iter)
        except StopIteration:
            self.validation_data_loader_iter = iter(self.validation_data_loader)
            batch = next(self.validation_data_loader_iter)
        return [x.to(device) for x in batch]

    def get_checkpoint_index_to_save(self, examples_seen_so_far: int) -> int:
        checkpoint_index = 0
        for i in range(len(self.checkpoint_examples)):
            if self.checkpoint_examples[i] <= examples_seen_so_far:
                checkpoint_index = i
        return checkpoint_index

    def barrier(self, local_rank: int):
        if self.distrib_backend == 'nccl':
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()

    def train(self,
              world_size: int,
              rank: int,
              local_rank: int,
              target_checkpoint_examples: Optional[int] = None,
              device_mapper: Optional[Callable[[int, int], torch.device]] = None):
        if target_checkpoint_examples is None:
            target_checkpoint_examples = self.checkpoint_examples[-1]

        if device_mapper is None:
            device_mapper = SimpleCudaDeviceMapper()
        device = device_mapper(rank, local_rank)

        sample_output_data = self.load_sample_output_data(rank, device)
        training_state = self.load_previous_training_state(
            target_checkpoint_examples, world_size, rank, local_rank, device)
        summary_writer = self.get_summary_writer(rank)
        if summary_writer is not None:
            log_func_factory = lambda name, num: create_log_func(summary_writer, name, num)
        else:
            log_func_factory = None
        last_time = time.time()

        while training_state.examples_seen_so_far < target_checkpoint_examples:
            # Set the learning rate
            learning_rate_by_module_name = self.training_protocol.get_learning_rate(training_state.examples_seen_so_far)
            for module_name in self.module_factories.keys():
                if module_name not in learning_rate_by_module_name or module_name not in training_state.optimizers:
                    continue
                lr = learning_rate_by_module_name[module_name]
                set_learning_rate(training_state.optimizers[module_name], lr)
                if summary_writer is not None:
                    summary_writer.add_scalar(
                        module_name + "_learning_rate", lr, training_state.examples_seen_so_far)

            # One training iteration
            training_batch = self.get_next_training_batch(training_state.examples_seen_so_far, world_size, device)
            self.training_protocol.run_training_iteration(
                training_batch,
                training_state.examples_seen_so_far,
                training_state.modules,
                training_state.accumulated_modules,
                training_state.optimizers,
                self.losses,
                log_func_factory,
                device)

            # Accumulate model data
            for module_name in self.accumulators:
                new_module = training_state.modules[module_name]
                if isinstance(new_module, DistributedDataParallel):
                    new_module = new_module.module
                buffer_module = training_state.accumulated_modules[module_name]
                self.accumulators[module_name].accumulate(
                    new_module, buffer_module, examples_seen_so_far=training_state.examples_seen_so_far)

            # Advance the number of examples seen so far
            next_num_examples = self.get_next_num_examples(training_state.examples_seen_so_far)
            training_state.examples_seen_so_far += self.training_protocol.get_batch_size() * world_size

            # Validation iteration
            if self.validation_protocol is not None \
                    and training_state.examples_seen_so_far >= next_num_examples[KEY_VALIDATION] \
                    and rank == 0:
                validation_batch = self.get_next_validation_batch(device)
                self.validation_protocol.run_validation_iteration(
                    validation_batch,
                    training_state.examples_seen_so_far,
                    training_state.modules,
                    training_state.accumulated_modules,
                    self.losses,
                    log_func_factory,
                    device)

            # Save sample output
            if self.sample_output_protocol is not None \
                    and training_state.examples_seen_so_far >= next_num_examples[KEY_SAMPLE_OUTPUT]:
                if rank == 0:
                    self.sample_output_protocol.save_sample_output_data(
                        training_state.modules,
                        training_state.accumulated_modules,
                        sample_output_data,
                        self.prefix + "/sample_outputs",
                        training_state.examples_seen_so_far,
                        device)
                self.barrier(local_rank)

            # Save checkpoint
            if training_state.examples_seen_so_far >= next_num_examples[KEY_CHECKPOINT]:
                checkpoint_index = self.get_checkpoint_index_to_save(training_state.examples_seen_so_far)
                training_state.save(
                    self.get_checkpoint_prefix(checkpoint_index), rank, lambda: self.barrier(local_rank))
                if next_num_examples[KEY_CHECKPOINT] != next_num_examples[KEY_SNAPSHOT]:
                    training_state.save(self.get_snapshot_prefix(), rank, lambda: self.barrier(local_rank))

            # Save snapshot
            if training_state.examples_seen_so_far >= next_num_examples[KEY_SNAPSHOT]:
                training_state.save(self.get_snapshot_prefix(), rank, lambda: self.barrier(local_rank))

            now = time.time()
            if now - last_time > 10:
                logging.info("Showed %d training examples." % training_state.examples_seen_so_far)
                last_time = now

    @staticmethod
    def get_default_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='Training script.')
        parser.add_argument("--target_checkpoint_examples", type=int)
        return parser

    @staticmethod
    def run_with_args(trainer_factory: Callable[[int, str], 'DistributedTrainer'],
                      args,
                      backend: str = 'gloo',
                      device_mapper: Optional[Callable[[int, int], torch.device]] = None):
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])

        torch.distributed.init_process_group(backend)
        trainer = trainer_factory(world_size, backend)
        trainer.train(world_size, rank, local_rank, args.target_checkpoint_examples, device_mapper)

    @staticmethod
    def run(trainer_factory: Callable[[int, str], 'DistributedTrainer'],
            backend: str = 'gloo',
            device_mapper: Optional[Callable[[int, int], torch.device]] = None,
            args: Optional[Any] = None):
        if args is None:
            parser = DistributedTrainer.get_default_arg_parser()
            args = parser.parse_args()

        DistributedTrainer.run_with_args(trainer_factory, args, backend, device_mapper)
