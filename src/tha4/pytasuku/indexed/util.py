import os
from typing import Iterable, Dict, Callable, List

from tha4.pytasuku.workspace import Workspace
from tha4.pytasuku.indexed.all_tasks import AllTasks
from tha4.pytasuku.indexed.indexed_tasks import IndexedTasks


def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        print("[delete] " + file_name)
    else:
        print("[not exist] " + file_name)


def all_tasks_from_named_tasks_map(
        workspace: Workspace,
        prefix: str,
        tasks: Iterable[Dict[str, IndexedTasks]],
        define_all_tasks: bool = True) \
        -> Dict[str, IndexedTasks]:
    subtasks = [x for x in tasks]
    name_to_subtask_list = {}
    for a_subtasks in subtasks:
        for name in a_subtasks:
            if not define_all_tasks and name == "all":
                continue
            if name not in name_to_subtask_list:
                name_to_subtask_list[name] = []
            name_to_subtask_list[name].append(a_subtasks[name])
    output = {}
    for name in name_to_subtask_list:
        output[name] = AllTasks(workspace, prefix, name_to_subtask_list[name], name)
    return output


def create_tasks_hierarchy_helper(
        workspace: Workspace,
        prefix: str,
        tasks_func: Callable[[Workspace, str, List[str]], Dict[str, IndexedTasks]],
        branches: List[List[str]],
        path: List[str]):
    if len(branches) == 0:
        return tasks_func(workspace, prefix, path)
    else:
        tasks = {}
        for branch in branches[0]:
            output_tasks = create_tasks_hierarchy_helper(
                workspace,
                f"{prefix}/{branch}",
                tasks_func,
                branches[1:],
                path + [branch])
            if output_tasks is not None:
                tasks[branch] = output_tasks
        return all_tasks_from_named_tasks_map(workspace, prefix, tasks.values())


def create_task_hierarchy(
        workspace: Workspace,
        prefix: str,
        tasks_func: Callable[[Workspace, str, List[str]], Dict[str, IndexedTasks]],
        branches: List[List[str]]) -> Dict[str, IndexedTasks]:
    return create_tasks_hierarchy_helper(workspace, prefix, tasks_func, branches, [])


def write_done_file(file_name: str):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wt") as fout:
        fout.write("DONE!!!")