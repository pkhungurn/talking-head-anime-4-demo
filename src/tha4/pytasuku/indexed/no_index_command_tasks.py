import abc
from typing import List

from tha4.pytasuku.workspace import Workspace
from tha4.pytasuku.indexed.indexed_tasks import IndexedTasks


class NoIndexCommandTasks(IndexedTasks, abc.ABC):
    def __init__(self, workspace: Workspace, prefix: str, command_name: str, define_tasks_immediately: bool = True):
        super().__init__(workspace, prefix)
        self.command_name = command_name
        if define_tasks_immediately:
            self.define_tasks()

    @property
    def run_command(self):
        return self.prefix + "/" + self.command_name

    @property
    def clean_command(self):
        return self.prefix + "/" + self.command_name + "_clean"

    @property
    def arity(self) -> int:
        return 0

    @property
    def shape(self) -> List[int]:
        return []

    @abc.abstractmethod
    def execute_run_command(self):
        pass

    @abc.abstractmethod
    def execute_clean_command(self):
        pass

    def define_tasks(self):
        self.workspace.create_command_task(self.run_command, [], self.execute_run_command)
        self.workspace.create_command_task(self.clean_command, [], self.execute_clean_command)
