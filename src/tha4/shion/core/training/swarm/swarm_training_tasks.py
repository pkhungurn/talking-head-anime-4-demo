from typing import Callable, Optional, List

from tha4.pytasuku.workspace import Workspace
from tha4.shion.core.training.distrib.distributed_training_tasks import RdzvConfig, \
    run_standalone_distributed_training_script
from tha4.shion.core.training.single.training_states import TrainingState
from tha4.shion.core.training.swarm.swarm_unit_trainer import SwarmUnitTrainer


def define_standalone_swarm_training_tasks(
        workspace: Workspace,
        swarm_unit_trainer_func: Callable[[], SwarmUnitTrainer],
        training_script_file_name: str,
        num_proc_per_node: int,
        dependencies: Optional[List[str]] = None,
        rdzv_config: Optional[RdzvConfig] = None):
    trainer = swarm_unit_trainer_func()
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
        return lambda: run_standalone_distributed_training_script(
            training_script_file_name,
            num_proc_per_node,
            target_checkpoint_examples,
            rdzv_config=rdzv_config)

    train_tasks = []
    for checkpoint_index in range(0, len(checkpoint_examples)):
        for module_name in trainer.module_names:
            module_file_name = TrainingState.get_module_file_name(
                trainer.get_checkpoint_prefix(checkpoint_index),
                module_name)
            workspace.create_file_task(
                module_file_name,
                module_file_dependencies,
                create_train_func(trainer.checkpoint_examples[checkpoint_index]))
        for module_name in trainer.accumulators:
            accumulated_module_file_name = TrainingState.get_accumulated_module_file_name(
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
