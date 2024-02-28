import logging
import time
from datetime import datetime
from typing import Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tha4.pytasuku.workspace import Workspace
from tha4.shion.core.load_save import torch_save, torch_load
from tha4.shion.core.loss import Loss
from tha4.shion.core.module_accumulator import ModuleAccumulator
from tha4.shion.core.module_factory import ModuleFactory
from tha4.shion.core.training.sample_output_protocol import SampleOutputProtocol
from tha4.shion.core.training.single.training_states import TrainingState
from tha4.shion.core.training.training_protocol import TrainingProtocol
from tha4.shion.core.training.util import get_least_greater_multiple, create_log_func, set_learning_rate
from tha4.shion.core.training.validation_protocol import ValidationProtocol

KEY_CHECKPOINT = 'checkpoint'
KEY_SNAPSHOT = 'snapshot'
KEY_VALIDATION = 'validation'
KEY_SAMPLE_OUTPUT = 'sample_output'


class TrainingTasks:
    def __init__(
            self,
            workspace: Workspace,
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
            device: torch.device,
            num_data_loader_workers: int = 8,
            dependencies: Optional[List[str]] = None):
        super().__init__()
        self.num_data_loader_workers = num_data_loader_workers
        self.accumulators = accumulators
        self.device = device
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
        self.validation_data_loader = None
        self.validation_data_loader_iter = None
        self.validation_data_loader_batch_size = None
        self.sample_output_data = None
        self.summary_writer = None
        self.log_dir = None
        self.training_state = None

        if dependencies is None:
            dependencies = []
        self.sample_output_data_task = workspace.create_file_task(
            self.get_sample_output_data_file_name(), dependencies, self.save_sample_output_data)

        module_file_dependencies = [self.sample_output_data_task.name]
        for module_name in pretrained_module_file_names:
            module_file_dependencies.append(self.pretrained_module_file_names[module_name])

        def create_train_func(target_examples: int):
            return lambda: self.train(target_examples)

        train_tasks = []
        for checkpoint_index in range(1, len(self.checkpoint_examples)):
            for module_name in self.module_names:
                module_file_name = TrainingState.get_module_file_name(
                    self.get_checkpoint_prefix(checkpoint_index),
                    module_name)
                workspace.create_file_task(
                    module_file_name,
                    module_file_dependencies,
                    create_train_func(self.checkpoint_examples[checkpoint_index]))
            for module_name in self.accumulators:
                accumulated_module_file_name = TrainingState.get_accumulated_module_file_name(
                    self.get_checkpoint_prefix(checkpoint_index),
                    module_name)
                workspace.create_file_task(
                    accumulated_module_file_name,
                    module_file_dependencies,
                    create_train_func(self.checkpoint_examples[checkpoint_index]))
            workspace.create_command_task(
                self.get_checkpoint_prefix(checkpoint_index) + "/train",
                module_file_dependencies,
                create_train_func(self.checkpoint_examples[checkpoint_index]))
            train_tasks.append(self.get_checkpoint_prefix(checkpoint_index) + "/train")

        self.train_task = workspace.create_file_task(
            self.get_train_command_name(),
            module_file_dependencies,
            create_train_func(self.checkpoint_examples[-1]))

    def get_sample_output_data_file_name(self):
        return self.prefix + "/sample_output_data.pt"

    def save_sample_output_data(self):
        if self.sample_output_protocol is not None:
            torch.manual_seed(self.sample_output_protocol.get_random_seed())
            sample_output_data = self.sample_output_protocol.get_sample_output_data(self.validation_dataset,
                                                                                    self.device)
            torch_save(sample_output_data, self.get_sample_output_data_file_name())
        else:
            torch_save({}, self.get_sample_output_data_file_name())

    def get_module_file_name(self, checkpoint_index, module_name):
        return TrainingState.get_module_file_name(self.get_checkpoint_prefix(checkpoint_index), module_name)

    def get_last_module_file_name(self, module_name):
        return self.get_module_file_name(len(self.checkpoint_examples) - 1, module_name)

    def get_log_dir(self):
        if self.log_dir is None:
            now = datetime.now()
            self.log_dir = self.prefix + "/log/" + now.strftime("%Y_%m_%d__%H_%M_%S")
        return self.log_dir

    def get_summary_writer(self) -> SummaryWriter:
        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(log_dir=self.get_log_dir())
        return self.summary_writer

    def get_train_command_name(self) -> str:
        return self.prefix + "/train"

    def get_snapshot_prefix(self) -> str:
        return self.prefix + "/snapshot"

    def get_checkpoint_prefix(self, checkpoint_index) -> str:
        return "%s/checkpoint/%04d" % (self.prefix, checkpoint_index)

    def can_load_training_state(self, prefix) -> bool:
        return TrainingState.can_load(
            prefix,
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories())

    def load_training_state(self, prefix) -> TrainingState:
        return TrainingState.load(
            prefix,
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories(),
            self.device)

    def get_initial_training_state(self) -> TrainingState:
        training_state = TrainingState.new(
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories(),
            self.training_protocol.get_random_seed(),
            self.device,
            self.pretrained_module_file_names)
        logging.info("Created a new initial training state.")
        return training_state

    def load_previous_training_state(self, target_checkpoint_examples: int) -> TrainingState:
        if self.can_load_training_state(self.get_snapshot_prefix()):
            examples_seen_so_far = TrainingState.get_examples_seen_so_far(self.get_snapshot_prefix())
            diff = examples_seen_so_far - target_checkpoint_examples
            if diff < self.training_protocol.get_batch_size():
                return self.load_training_state(self.get_snapshot_prefix())
        num_checkpoints = len(self.checkpoint_examples)
        for checkpoint_index in range(num_checkpoints - 1, -1, -1):
            if self.can_load_training_state(self.get_checkpoint_prefix(checkpoint_index)):
                examples_seen_so_far = TrainingState.get_examples_seen_so_far(
                    self.get_checkpoint_prefix(checkpoint_index))
                diff = examples_seen_so_far - target_checkpoint_examples
                if diff < self.training_protocol.get_batch_size():
                    return self.load_training_state(self.get_checkpoint_prefix(checkpoint_index))
        return self.get_initial_training_state()

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

    def get_checkpoint_index_to_save(self, examples_seen_so_far: int) -> int:
        checkpoint_index = 0
        for i in range(len(self.checkpoint_examples)):
            if self.checkpoint_examples[i] <= examples_seen_so_far:
                checkpoint_index = i
        return checkpoint_index

    def get_next_training_batch(self):
        if self.training_data_loader is None:
            self.training_data_loader = DataLoader(
                self.training_dataset,
                batch_size=self.training_protocol.get_batch_size(),
                shuffle=True,
                num_workers=self.num_data_loader_workers,
                drop_last=True)
        if self.training_data_loader_iter is None:
            self.training_data_loader_iter = iter(self.training_data_loader)
        try:
            batch = next(self.training_data_loader_iter)
        except StopIteration:
            self.training_data_loader_iter = iter(self.training_data_loader)
            batch = next(self.training_data_loader_iter)
        return [x.to(self.device) for x in batch]

    def get_next_validation_batch(self):
        if self.validation_dataset is None:
            return None
        if self.validation_data_loader is None:
            self.validation_data_loader = DataLoader(
                self.validation_dataset,
                batch_size=self.validation_protocol.get_batch_size(),
                shuffle=True,
                num_workers=self.num_data_loader_workers,
                drop_last=True)
        if self.validation_data_loader_iter is None:
            self.validation_data_loader_iter = iter(self.validation_data_loader)
        try:
            batch = next(self.validation_data_loader_iter)
        except StopIteration:
            self.validation_data_loader_iter = iter(self.validation_data_loader)
            batch = next(self.validation_data_loader_iter)
        return [x.to(self.device) for x in batch]

    def get_checkpoint_index(self, target_checkpoint_examples: int):
        return self.checkpoint_examples.index(target_checkpoint_examples)

    def train(self, target_checkpoint_examples: Optional[int] = None):
        if target_checkpoint_examples is None:
            target_checkpoint_examples = self.checkpoint_examples[-1]

        sample_output_data = torch_load(self.get_sample_output_data_file_name())
        logging.info("Loaded sampled output data from %s", self.get_sample_output_data_file_name())
        training_state = self.load_previous_training_state(target_checkpoint_examples)
        summary_writer = self.get_summary_writer()
        last_time = time.time()

        while training_state.examples_seen_so_far < target_checkpoint_examples:
            # One training iteration
            learning_rate = self.training_protocol.get_learning_rate(training_state.examples_seen_so_far)
            for module_name in self.module_factories.keys():
                if module_name not in learning_rate or module_name not in training_state.optimizers:
                    continue
                lr = learning_rate[module_name]
                set_learning_rate(training_state.optimizers[module_name], lr)
                self.get_summary_writer().add_scalar(
                    module_name + "_learning_rate", lr, training_state.examples_seen_so_far)
            training_batch = self.get_next_training_batch()
            self.training_protocol.run_training_iteration(
                training_batch,
                training_state.examples_seen_so_far,
                training_state.modules,
                training_state.accumulated_modules,
                training_state.optimizers,
                self.losses,
                lambda name, num: create_log_func(summary_writer, name, num),
                self.device)

            # Accumulate model data
            for module_name in self.accumulators:
                new_module = training_state.modules[module_name]
                buffer_module = training_state.accumulated_modules[module_name]
                self.accumulators[module_name].accumulate(
                    new_module,
                    buffer_module,
                    training_state.examples_seen_so_far)

            # Advance the number of examples seen so far
            next_num_examples = self.get_next_num_examples(training_state.examples_seen_so_far)
            training_state.examples_seen_so_far += self.training_protocol.get_batch_size()

            # Validation iteration
            if self.validation_protocol is not None \
                    and training_state.examples_seen_so_far >= next_num_examples[KEY_VALIDATION]:
                validation_batch = self.get_next_validation_batch()
                self.validation_protocol.run_validation_iteration(
                    validation_batch,
                    training_state.examples_seen_so_far,
                    training_state.modules,
                    training_state.accumulated_modules,
                    self.losses,
                    lambda name, num: create_log_func(summary_writer, name, num),
                    self.device)

            # Save sample output
            if self.sample_output_protocol is not None \
                    and training_state.examples_seen_so_far >= next_num_examples[KEY_SAMPLE_OUTPUT]:
                self.sample_output_protocol.save_sample_output_data(
                    training_state.modules,
                    training_state.accumulated_modules,
                    sample_output_data,
                    self.prefix + "/sample_outputs",
                    training_state.examples_seen_so_far,
                    self.device)

            # Save checkpoint
            if training_state.examples_seen_so_far >= next_num_examples[KEY_CHECKPOINT]:
                checkpoint_index = self.get_checkpoint_index_to_save(training_state.examples_seen_so_far)
                training_state.save(self.get_checkpoint_prefix(checkpoint_index))
                if next_num_examples[KEY_CHECKPOINT] != next_num_examples[KEY_SNAPSHOT]:
                    training_state.save(self.get_snapshot_prefix())

            # Save snapshot
            if training_state.examples_seen_so_far >= next_num_examples[KEY_SNAPSHOT]:
                training_state.save(self.get_snapshot_prefix())

            now = time.time()
            if now - last_time > 10:
                logging.info("Showed %d training examples." % training_state.examples_seen_so_far)
                last_time = now
