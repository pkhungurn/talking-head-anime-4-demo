import os
import logging
from typing import List


class Task:
    def __init__(self, workspace: 'Workspace', name: str, dependencies: List[str]):
        self._workspace = workspace
        self._name = name
        self._dependencies = dependencies
        self._workspace.add_task(self)

    def run(self):
        pass

    @property
    def can_run(self) -> bool:
        return True

    @property
    def needs_to_be_run(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> List[str]:
        return self._dependencies

    @property
    def workspace(self) -> 'Workspace':
        return self._workspace

    @property
    def timestamp(self) -> float:
        return float("inf")


class CommandTask(Task):
    def __init__(self, workspace, name, dependencies):
        super().__init__(workspace, name, dependencies)

    @property
    def needs_to_be_run(self):
        return True


class PlaceholderTask(Task):
    def __init__(self, workspace, name):
        super().__init__(workspace, name, [])

    @property
    def can_run(self):
        return False

    def run(self):
        raise Exception("A  placeholder task cannot be run! (%s)" % self.name)

    @property
    def needs_to_be_run(self):
        return not os.path.isfile(self.name)

    @property
    def timestamp(self) -> float:
        if not os.path.isfile(self.name):
            return float("inf")
        else:
            return os.path.getmtime(self.name)


class FileTask(Task):
    def __init__(self, workspace, name, dependencies):
        super().__init__(workspace, name, dependencies)

    @property
    def timestamp(self):
        return os.path.getmtime(self.name)

    @property
    def needs_to_be_run(self):
        if not os.path.isfile(self.name):
            logging.info("Task %s will be run because the corresponding file does not exist." % self.name)
            return True
        for dep in self.dependencies:
            if self.workspace.needs_to_run(dep):
                logging.info("Task %s will be run because dependency %s also needs to be run." % (self.name, dep))
                return True
            else:
                self_timestamp = self.timestamp
                dep_task = self.workspace.get_task(dep)
                if dep_task.timestamp > self_timestamp:
                    if isinstance(dep_task, FileTask) or isinstance(dep_task, PlaceholderTask):
                        logging.info("Task %s needs to be run because task %s has later timestamp." %
                                     (self.name, dep))
                    elif isinstance(dep_task, CommandTask):
                        logging.info("Task %s needs to be run because task %s is a command." % (self.name, dep))
                    return True
        return False
