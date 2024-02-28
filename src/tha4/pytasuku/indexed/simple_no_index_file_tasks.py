from typing import Callable, List, Optional

from tha4.pytasuku.workspace import Workspace
from tha4.pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks


class SimpleNoIndexFileTasks(NoIndexFileTasks):
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 command_name: str,
                 file_name: str,
                 run_func: Callable[[], None],
                 dependencies: Optional[List[str]] = None):
        super().__init__(workspace, prefix, command_name, define_tasks_immediately=False)
        if dependencies is None:
            dependencies = []
        self.run_func = run_func
        self._file_name = file_name
        self.dependencies = dependencies
        self.define_tasks()

    @property
    def file_name(self):
        return self._file_name

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, self.dependencies, self.run_func)
