import logging

from tha4.shion.core.training.distrib.distributed_trainer import DistributedTrainer
from tha4.distiller.distiller_config import DistillerConfig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = DistributedTrainer.get_default_arg_parser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()

    config_file_name = args.config_file
    config = DistillerConfig.load(config_file_name)

    DistributedTrainer.run_with_args(config.get_face_morpher_trainer, args)
