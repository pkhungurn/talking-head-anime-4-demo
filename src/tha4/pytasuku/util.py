import os.path
from typing import List
import logging

from tha4.pytasuku.workspace import Workspace


def create_delete_all_task(workspace: Workspace, name: str, files: List[str]):
    def delete_all():
        for file in files:
            if os.path.exists(file):
                logging.info("Removing %s ..." % file)
                os.remove(file)

    workspace.create_command_task(name, [], delete_all)
