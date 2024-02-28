import abc
from typing import List

from tha4.pytasuku.workspace import Workspace
from tha4.pytasuku.indexed.indexed_file_tasks import IndexedFileTasks
from tha4.pytasuku.indexed.util import delete_file


class TwoIndicesFileTasks(IndexedFileTasks, abc.ABC):
    def __init__(self, workspace: Workspace, prefix: str, command_name: str,
                 count0: int, count1: int, define_tasks_immediately: bool = True):
        super().__init__(workspace, prefix)
        self.count1 = count1
        self.count0 = count0
        self.command_name = command_name
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
        return [self.count0, self.count1]

    @property
    def arity(self) -> int:
        return 2

    @abc.abstractmethod
    def file_name(self, index0: int, index1: int) -> str:
        pass

    @property
    def file_list(self) -> List[str]:
        if len(self.file_list_) == 0:
            for i in range(self.count0):
                for j in range(self.count1):
                    self.file_list_.append(self.file_name(i, j))
        return self.file_list_

    @abc.abstractmethod
    def create_file_tasks(self, index0: int, index1: int):
        pass

    def get_file_name(self, *indices: int) -> str:
        if len(indices) != 2:
            raise IndexError("TwoIndicesFileTasks.get_file_name require two indices, " +
                             "but not exactly 2 indices were provide")
        return self.file_name(indices[0], indices[1])

    def clean(self):
        for file in self.file_list:
            delete_file(file)

    def define_tasks(self):
        for index0 in range(self.count0):
            for index1 in range(self.count1):
                self.create_file_tasks(index0, index1)
        self.workspace.create_command_task(self.run_command, self.file_list)
        self.workspace.create_command_task(self.clean_command, [], lambda: self.clean())
