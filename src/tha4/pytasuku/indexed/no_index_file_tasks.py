import abc
from typing import List

from tha4.pytasuku.workspace import Workspace
from tha4.pytasuku.indexed.indexed_file_tasks import IndexedFileTasks
from tha4.pytasuku.indexed.util import delete_file


class NoIndexFileTasks(IndexedFileTasks, abc.ABC):
    def __init__(self, workspace: Workspace, prefix: str, command_name: str, define_tasks_immediately: bool = True):
        super().__init__(workspace, prefix)
        self.command_name = command_name
        if define_tasks_immediately:
            self.define_tasks()

    @property
    @abc.abstractmethod
    def file_name(self):
        pass

    @abc.abstractmethod
    def create_file_task(self):
        pass

    def get_file_name(self, *indices: int) -> str:
        if len(indices) > 0:
            raise IndexError("NoIndexFileTasks has arity 0, but get_file_name is called with an index.")
        return self.file_name

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

    @property
    def file_list(self) -> List[str]:
        return [self.file_name]

    def clean(self):
        delete_file(self.file_name)

    def define_tasks(self):
        self.create_file_task()
        self.workspace.create_command_task(self.run_command, [self.file_name])
        self.workspace.create_command_task(self.clean_command, [], lambda: self.clean())