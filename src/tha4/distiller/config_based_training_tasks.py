import logging
import os
import sys
from typing import Callable, List, Optional

from tha4.pytasuku.workspace import Workspace
from tha4.shion.core.training.distrib.distributed_trainer import DistributedTrainer
from tha4.shion.core.training.distrib.distributed_training_states import DistributedTrainingState


def get_torchrun_executable():
    return os.path.dirname(sys.executable) + os.path.sep + "torchrun"


class RdzvConfig:
    def __init__(self, id: int, port: int):
        self.port = port
        self.id = id


def run_standalone_config_based_training_script(
        training_script_file_name: str,
        config_file_name: str,
        num_proc_per_node: int,
        target_checkpoint_examples: Optional[int] = None,
        rdzv_config: Optional[RdzvConfig] = None):
    command = f"{get_torchrun_executable()} " \
              f"--nnodes=1 " \
              f"--nproc_per_node={num_proc_per_node} "
    if rdzv_config is not None:
        command += f"--rdzv_endpoint=localhost:{rdzv_config.port} "
        command += "--rdzv_backend=c10d "
        command += f"--rdzv_id={rdzv_config.id} "
    else:
        command += "--standalone "
    command += f"{training_script_file_name} "
    if target_checkpoint_examples is not None:
        command += f"--target_checkpoint_examples {target_checkpoint_examples} "
        command += f"--config_file={config_file_name} "
    logging.info(f"Executing -- {command}")
    os.system(command)


def define_standalone_config_based_training_tasks(
        workspace: Workspace,
        distributed_trainer_func: Callable[[], DistributedTrainer],
        training_script_file_name: str,
        config_file_name: str,
        num_proc_per_node: int,
        dependencies: Optional[List[str]] = None,
        rdzv_config: Optional[RdzvConfig] = None):
    trainer = distributed_trainer_func()
    checkpoint_examples = trainer.training_protocol.get_checkpoint_examples()
    assert len(checkpoint_examples) >= 1
    assert checkpoint_examples[0] > 0
    checkpoint_examples = [0] + checkpoint_examples

    if dependencies is None:
        dependencies = []
    module_file_dependencies = dependencies[:]
    for module_name in trainer.pretrained_module_file_names:
        module_file_dependencies.append(trainer.pretrained_module_file_names[module_name])

    def create_train_func(target_checkpoint_examples: int):
        return lambda: run_standalone_config_based_training_script(
            training_script_file_name,
            config_file_name,
            num_proc_per_node,
            target_checkpoint_examples,
            rdzv_config=rdzv_config)

    train_tasks = []
    for checkpoint_index in range(0, len(checkpoint_examples)):
        for module_name in trainer.module_names:
            module_file_name = DistributedTrainingState.get_module_file_name(
                trainer.get_checkpoint_prefix(checkpoint_index),
                module_name)
            workspace.create_file_task(
                module_file_name,
                module_file_dependencies,
                create_train_func(trainer.checkpoint_examples[checkpoint_index]))
        for module_name in trainer.accumulators:
            accumulated_module_file_name = DistributedTrainingState.get_accumulated_module_file_name(
                trainer.get_checkpoint_prefix(checkpoint_index),
                module_name)
            workspace.create_file_task(
                accumulated_module_file_name,
                module_file_dependencies,
                create_train_func(checkpoint_examples[checkpoint_index]))
        workspace.create_command_task(
            trainer.get_checkpoint_prefix(checkpoint_index) + "/train_standalone",
            module_file_dependencies,
            create_train_func(checkpoint_examples[checkpoint_index]))
        train_tasks.append(trainer.get_checkpoint_prefix(checkpoint_index) + "/train_standlone")
    workspace.create_file_task(
        trainer.prefix + "/train_standalone",
        module_file_dependencies,
        create_train_func(checkpoint_examples[-1]))
