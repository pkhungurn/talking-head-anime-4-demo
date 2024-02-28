import abc
from typing import Iterable, List

from tha4.pytasuku.workspace import Workspace
from tha4.pytasuku.indexed.indexed_tasks import IndexedTasks
from tha4.pytasuku.workspace import do_nothing


class BundledIndexedTasks:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def indexed_tasks_command_names(self) -> Iterable[str]:
        pass

    @abc.abstractmethod
    def get_indexed_tasks(self, command_name) -> IndexedTasks:
        pass


def define_all_tasks_from_list(workspace: Workspace, prefix: str, tasks: List[BundledIndexedTasks]):
    for command_name in tasks[0].indexed_tasks_command_names:
        workspace.create_command_task(
            prefix + "/" + command_name,
            [x.get_indexed_tasks(command_name).run_command for x in tasks],
            do_nothing)
        workspace.create_command_task(
            prefix + "/" + command_name + "_clean",
            [x.get_indexed_tasks(command_name).clean_command for x in tasks],
            do_nothing)
