from contextlib import contextmanager
from enum import Enum
from typing import List

from tha4.pytasuku.task import Task, CommandTask, FileTask, PlaceholderTask


class WorkspaceState(Enum):
    OUT_OF_SESSION = 1
    IN_SESSION = 2


class NodeState(Enum):
    IN_STACK = 1
    VISITED = 2


class FuncCommandTask(CommandTask):
    def __init__(self, workspace, name, dependencies, func):
        super().__init__(workspace, name, dependencies)
        self._func = func

    def run(self):
        self._func()


class FuncFileTask(FileTask):
    def __init__(self, workspace, name, dependencies, func):
        super().__init__(workspace, name, dependencies)
        self._func = func

    def run(self):
        self._func()


def do_nothing():
    pass


class Workspace:
    def __init__(self):
        self._tasks = dict()
        self._name_to_done = None
        self._state = WorkspaceState.OUT_OF_SESSION
        self._modified = False

    @property
    def modified(self) -> bool:
        return self._modified

    @property
    def state(self) -> WorkspaceState:
        return self._state

    @property
    def in_session(self) -> bool:
        return self._state == WorkspaceState.IN_SESSION

    def task_exists(self, name: str) -> bool:
        return name in self._tasks

    def task_exists_and_not_placeholder(self, name: str) -> bool:
        return self.task_exists(name) and not isinstance(self.get_task(name), PlaceholderTask)

    def get_task(self, name: str) -> Task:
        return self._tasks[name]

    def add_task(self, task):
        if self.in_session:
            raise RuntimeError("New tasks can only be created when the workspace is out of session.")
        if isinstance(task, PlaceholderTask):
            if not self.task_exists(task.name):
                self._tasks[task.name] = task
                self._modified = True
        else:
            self._tasks[task.name] = task
            for dep in task.dependencies:
                PlaceholderTask(self, dep)
            self._modified = True

    def start_session(self):
        if self.in_session:
            raise RuntimeError("A session can only be started when the workspace is out of session.")
        if self.modified:
            self.check_cycle()
        self._state = WorkspaceState.IN_SESSION
        self._name_to_done = dict()
        self._modified = False

    def end_session(self):
        if not self.in_session:
            raise RuntimeError("A session can only be ended when the workspace is in session.")
        self._state = WorkspaceState.OUT_OF_SESSION
        self._name_to_done = None

    @contextmanager
    def session(self):
        try:
            self.start_session()
            yield
        finally:
            self.end_session()

    def check_cycle(self):
        node_states = dict()
        for name in self._tasks:
            if name not in node_states:
                self.dfs(name, node_states)

    def dfs(self, name, node_states):
        node_states[name] = NodeState.IN_STACK
        task = self.get_task(name)
        for dep in task.dependencies:
            if dep not in node_states:
                self.dfs(dep, node_states)
            else:
                state = node_states[dep]
                if state == NodeState.IN_STACK:
                    raise RuntimeError("Dicovered cyclic dependency!")
        node_states[name] = NodeState.VISITED

    def run(self, name):
        if not self.in_session:
            raise RuntimeError("A task can only be run when the workspace is in session.")
        if not self.task_exists(name):
            raise RuntimeError("Task %s does not exists" % name)
        self.run_helper(name)

    def run_helper(self, name):
        task = self.get_task(name)
        for dep in task.dependencies:
            if self.needs_to_run(dep):
                self.run_helper(dep)
        if self.needs_to_run(name):
            task.run()
            self._name_to_done[name] = True

    def needs_to_run(self, name):
        if not self.in_session:
            raise RuntimeError("You can only check whether a task needs to run when the workspace is in session.")
        if name in self._name_to_done:
            return not self._name_to_done[name]
        task = self.get_task(name)
        need_to_run_value = task.needs_to_be_run
        self._name_to_done[name] = not need_to_run_value
        return need_to_run_value

    def create_command_task(self, name, dependencies, func=do_nothing):
        return FuncCommandTask(self, name, dependencies, func)

    def create_file_task(self, name, dependencies, func):
        return FuncFileTask(self, name, dependencies, func)


def command_task(workspace: Workspace, name: str, dependencies: List[str]):
    def func(f):
        workspace.create_command_task(name, dependencies, f)
        return f

    return func


def file_task(workspace: Workspace, name: str, dependencies: List[str]):
    def func(f):
        workspace.create_file_task(name, dependencies, f)
        return f

    return func
