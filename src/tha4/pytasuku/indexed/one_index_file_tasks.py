import abc

from typing import List

from tha4.pytasuku.workspace import Workspace
from tha4.pytasuku.indexed.indexed_file_tasks import IndexedFileTasks
from tha4.pytasuku.indexed.util import delete_file


class OneIndexFileTasks(IndexedFileTasks, abc.ABC):
    def __init__(self, workspace: Workspace, prefix: str, command_name: str, count: int,
                 define_tasks_immediately: bool = True):
        super().__init__(workspace, prefix)
        self.command_name = command_name
        self.count = count
        self.file_list_ = []
        if define_tasks_immediately:
            self.define_tasks()

    @property
    def run_command(self) -> str:
        return self.prefix + "/" + self.command_name

    @property
    def clean_command(self) -> str:
        return self.prefix + "/" + self.command_name + "_clean"

    @property
    def shape(self) -> List[int]:
        return [self.count]

    @property
    def arity(self) -> int:
        return 1

    @abc.abstractmethod
    def file_name(self, index):
        pass

    @abc.abstractmethod
    def create_file_tasks(self, index):
        pass

    def get_file_name(self, *indices: int) -> str:
        if len(indices) != 1:
            raise IndexError("OneIndexFileTasks has arity 1, but "
                             "get_file_name does not get the appropriate number of arguments.")
        return self.file_name(indices[0])

    @property
    def file_list(self):
        if len(self.file_list_) == 0:
            for i in range(self.count):
                self.file_list_.append(self.file_name(i))
        return self.file_list_

    def clean(self):
        for file in self.file_list:
            delete_file(file)

    def define_tasks(self):
        for index in range(self.count):
            self.create_file_tasks(index)
        dependencies = self.file_list
        self.workspace.create_command_task(self.run_command, dependencies)
        self.workspace.create_command_task(self.clean_command, [], lambda: self.clean())
