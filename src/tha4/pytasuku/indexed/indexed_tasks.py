import abc
from typing import List

from tha4.pytasuku.workspace import Workspace


class IndexedTasks(abc.ABC):
    def __init__(self, workspace: Workspace, prefix: str):
        self.prefix = prefix
        self.workspace = workspace

    @property
    @abc.abstractmethod
    def run_command(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def clean_command(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> List[int]:
        pass

    @property
    @abc.abstractmethod
    def arity(self) -> int:
        pass

    @abc.abstractmethod
    def define_tasks(self):
        pass
