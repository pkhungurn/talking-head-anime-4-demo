import argparse
import logging

from tha4.distiller.distiller_config import DistillerConfig
from tha4.pytasuku.workspace import Workspace


def run_config(config_file_name: str):
    config = DistillerConfig.load(config_file_name)

    logging.basicConfig(level=logging.INFO, force=True)
    workspace = Workspace()
    config.define_tasks(workspace)

    workspace.start_session()
    workspace.run(f"{config.prefix}/all")
    workspace.end_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument("--config_file", type=str, required=True,
                        help="The name of the config file for the distillation process.")
    args = parser.parse_args()
    run_config(args.config_file)
