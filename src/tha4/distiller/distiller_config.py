import os.path
import shutil
import PIL.Image
from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf
from tha4.charmodel.character_model import CharacterModel
from tha4.pytasuku.workspace import Workspace, file_task
from tha4.distiller.config_based_training_tasks import define_standalone_config_based_training_tasks
from tha4.nn.siren.face_morpher.siren_face_morpher_00_trainer import SirenFaceMorpher00TrainerArgs
from tha4.nn.siren.morpher.siren_morpher_03_trainer import SirenMorpher03TrainerArgs, TrainingPhases, TrainingPhase, \
    LossWeights, LossTerm
from tha4.shion.base.image_util import pil_image_has_transparency

POSE_DATASET_FILE_NAME = 'data/pose_dataset.pt'


def copy_file(source_file_name: str, dest_file_name):
    os.makedirs(os.path.dirname(dest_file_name), exist_ok=True)
    shutil.copyfile(source_file_name, dest_file_name)


@dataclass
class DistillerConfig:
    prefix: str
    character_image_file_name: str
    face_mask_image_file_name: str

    face_morpher_random_seed_0: int = 12771885812175595441
    face_morpher_random_seed_1: int = 14367217090963479175
    face_morpher_num_training_examples_per_sample_output: Optional[int] = 10_000
    face_morpher_batch_size: int = 8

    body_morpher_random_seed_0: int = 2892221210020292507
    body_morpher_random_seed_1: int = 9998918537095922080
    body_morpher_num_training_examples_per_sample_output: Optional[int] = 10_000
    body_morpher_batch_size: int = 8

    num_cpu_workers: int = 1
    num_gpus: int = 1

    def check(self):
        DistillerConfig.check_prefix(self.prefix)
        DistillerConfig.check_character_image_file_name(self.character_image_file_name)
        DistillerConfig.check_face_mask_image_file_name(self.face_mask_image_file_name)

        DistillerConfig.check_num_cpu_workers(self.num_cpu_workers)
        DistillerConfig.check_num_gpus(self.num_gpus)

        DistillerConfig.check_random_seed(self.face_morpher_random_seed_0, "face_morpher_random_seed_0")
        DistillerConfig.check_random_seed(self.face_morpher_random_seed_1, "face_morpher_random_seed_1")
        DistillerConfig.check_batch_size(self.face_morpher_batch_size, "face_morpher_batch_size")
        DistillerConfig.check_num_training_examples_per_sample_output(
            self.face_morpher_num_training_examples_per_sample_output,
            "face_morpher_num_training_examples_per_sample_output")

        DistillerConfig.check_random_seed(self.body_morpher_random_seed_0, "body_morpher_random_seed_0")
        DistillerConfig.check_random_seed(self.body_morpher_random_seed_1, "body_morpher_random_seed_1")
        DistillerConfig.check_batch_size(self.body_morpher_batch_size, "body_morpher_batch_size")
        DistillerConfig.check_num_training_examples_per_sample_output(
            self.body_morpher_num_training_examples_per_sample_output,
            "body_morpher_num_training_examples_per_sample_output")

    @staticmethod
    def check_prefix(prefix):
        assert os.path.isdir(prefix), "The 'prefix' must be a directory."
        assert os.path.exists(prefix), f"The {prefix} directory does not exist."

    @staticmethod
    def check_character_image_file_name(file_name):
        _, ext = os.path.splitext(file_name)
        assert os.path.isfile(file_name), \
            f"The specified character image file name, {file_name}, does not point to a file."
        assert ext.lower() == ".png", "The character image file name must have extension '.png'."

        image = PIL.Image.open(file_name)
        assert pil_image_has_transparency(image), "The character image must have an alpha channel."
        assert image.width == 512 and image.height == 512, "The character image must be 512x512."
        image.close()

    @staticmethod
    def check_face_mask_image_file_name(file_name):
        _, ext = os.path.splitext(file_name)
        assert os.path.isfile(file_name), \
            f"The specified face mask image file name, {file_name}, does not point to a file."
        assert ext.lower() == ".png", "The face mask image file name must have extension '.png'."

        image = PIL.Image.open(file_name)
        assert image.width == 512 and image.height == 512, "The face mask image must be 512x512."
        assert image.mode == "RGB", "The face mask image must be an RGB image."
        for x in range(512):
            for y in range(512):
                r, g, b = image.getpixel((x, y))
                assert (r == 0) or (r == 255), "The R channel of the face mask image must be 0 or 255"
                assert (g == 0) or (g == 255), "The G channel of the face mask image must be 0 or 255"
                assert (b == 0) or (b == 255), "The B channel of the face mask image must be 0 or 255"
        image.close()

    @staticmethod
    def check_batch_size(value, field_name: str):
        assert isinstance(value, int), f"The {field_name} must be an integer."
        assert value >= 1, f"The {field_name} must be at least 1."
        assert value <= 8, f"The {field_name} must be at most 8."

    @staticmethod
    def check_num_cpu_workers(value):
        assert value >= 1, "The value of 'num_cpu_workers must be at least 1."

    @staticmethod
    def check_num_gpus(value):
        assert value >= 1, "The value of 'num_gpus' must be at least 1."

    @staticmethod
    def check_random_seed(value, field_name: str):
        assert isinstance(value, int), f"The {field_name} must be an integer."
        assert value >= 0 and value <= 0x_ffff_ffff_ffff_ffff, "A random seed must be between 0 and 2**64-1."

    @staticmethod
    def check_num_training_examples_per_sample_output(value, field_name):
        assert value in [10_000, 100_000, 1_000_000,
                         None], f"The {field_name} must be 10_000, 100_00, 1_000_000_000, or None."

    def save(self, file_name: str):
        conf = OmegaConf.structured(self)
        os.makedirs(self.prefix, exist_ok=True)
        with open(file_name, "wt") as fout:
            fout.write(OmegaConf.to_yaml(conf))

    def config_yaml_file_name(self):
        return f"{self.prefix}/config.yaml"

    def create_config_yaml_file(self):
        if os.path.exists(self.config_yaml_file_name()):
            return
        self.save(self.config_yaml_file_name())

    @staticmethod
    def load(file_name: str) -> 'DistillerConfig':
        conf = OmegaConf.to_container(OmegaConf.load(file_name))
        args = DistillerConfig(**conf)
        args.check()
        return args

    def face_morpher_prefix(self):
        return f"{self.prefix}/face_morpher"

    def get_face_morpher_trainer(self, world_size: Optional[int] = None, backend: str = 'gloo'):
        if world_size is None:
            world_size = self.num_gpus
        args = SirenFaceMorpher00TrainerArgs(
            character_file_name=self.character_image_file_name,
            face_mask_file_name=self.face_mask_image_file_name,
            pose_dataset_file_name=POSE_DATASET_FILE_NAME,
            total_worker=self.num_cpu_workers,
            num_training_examples_per_sample_output=self.face_morpher_num_training_examples_per_sample_output,
            total_batch_size=self.face_morpher_batch_size,
            training_random_seed=self.face_morpher_random_seed_0,
            sample_output_random_seed=self.face_morpher_random_seed_1)
        return args.create_trainer(self.face_morpher_prefix(), world_size, backend)

    def body_morpher_prefix(self):
        return f"{self.prefix}/body_morpher"

    def get_body_morpher_trainer(self, world_size: Optional[int] = None, backend: str = 'gloo'):
        if world_size is None:
            world_size = self.num_gpus
        args = SirenMorpher03TrainerArgs(
            character_file_name=self.character_image_file_name,
            pose_dataset_file_name=POSE_DATASET_FILE_NAME,
            total_worker=self.num_cpu_workers,
            num_training_examples_per_sample_output=self.body_morpher_num_training_examples_per_sample_output,
            training_random_seed=self.body_morpher_random_seed_0,
            sample_output_random_seed=self.body_morpher_random_seed_1,
            total_batch_size=self.body_morpher_batch_size,
            sample_output_batch_size=1,
            training_phases=TrainingPhases([
                TrainingPhase(
                    num_examples_upper_bound=200_000,
                    learning_rate=1e-4,
                    loss_weights=LossWeights(weights={
                        LossTerm.full_blended: 0.25,
                        LossTerm.full_warped: 0.25,
                        LossTerm.full_grid_change: 0.5,
                        LossTerm.full_color_change: 2.0,
                    })),
                TrainingPhase(
                    num_examples_upper_bound=400_000,
                    learning_rate=3e-5,
                    loss_weights=LossWeights(weights={
                        LossTerm.full_blended: 0.25,
                        LossTerm.full_warped: 0.25,
                        LossTerm.full_grid_change: 0.5,
                        LossTerm.full_color_change: 2.0,
                    })),
                TrainingPhase(
                    num_examples_upper_bound=600_000,
                    learning_rate=3e-5,
                    loss_weights=LossWeights(weights={
                        LossTerm.full_blended: 1.0,
                        LossTerm.full_warped: 2.5,
                        LossTerm.full_grid_change: 5.0,
                        LossTerm.full_color_change: 1.0,
                    })),
                TrainingPhase(
                    num_examples_upper_bound=800_000,
                    learning_rate=1e-5,
                    loss_weights=LossWeights(weights={
                        LossTerm.full_blended: 1.0,
                        LossTerm.full_warped: 2.5,
                        LossTerm.full_grid_change: 5.0,
                        LossTerm.full_color_change: 1.0,
                    })),
                TrainingPhase(
                    num_examples_upper_bound=1_300_000,
                    learning_rate=1e-5,
                    loss_weights=LossWeights(weights={
                        LossTerm.full_blended: 10.0,
                        LossTerm.full_warped: 1.0,
                        LossTerm.full_grid_change: 1.0,
                        LossTerm.full_color_change: 1.0,
                    })),
                TrainingPhase(
                    num_examples_upper_bound=1_500_000,
                    learning_rate=3e-6,
                    loss_weights=LossWeights(weights={
                        LossTerm.full_blended: 10.0,
                        LossTerm.full_warped: 1.0,
                        LossTerm.full_grid_change: 1.0,
                        LossTerm.full_color_change: 1.0,
                    })),
            ]))
        return args.create_trainer(self.body_morpher_prefix(), world_size, backend)

    def character_model_prefix(self):
        return f"{self.prefix}/character_model"

    def character_model_face_morpher_file_name(self):
        return f"{self.character_model_prefix()}/face_morpher.pt"

    def character_model_body_morpher_file_name(self):
        return f"{self.character_model_prefix()}/body_morpher.pt"

    def character_model_character_png_file_name(self):
        return f"{self.character_model_prefix()}/character.png"

    def character_model_yaml_file_name(self):
        return f"{self.character_model_prefix()}/character_model.yaml"

    def define_tasks(self, workspace: Workspace):
        workspace.create_file_task(self.config_yaml_file_name(), [], self.create_config_yaml_file)

        define_standalone_config_based_training_tasks(
            workspace,
            self.get_face_morpher_trainer,
            "src/tha4/distiller/distill_face_morpher.py",
            self.config_yaml_file_name(),
            num_proc_per_node=self.num_gpus,
            dependencies=[
                self.config_yaml_file_name(),
            ])

        define_standalone_config_based_training_tasks(
            workspace,
            self.get_body_morpher_trainer,
            "src/tha4/distiller/distill_body_morpher.py",
            self.config_yaml_file_name(),
            num_proc_per_node=self.num_gpus,
            dependencies=[
                self.config_yaml_file_name(),
            ])

        @file_task(workspace, self.character_model_character_png_file_name(), [self.character_image_file_name])
        def copy_character_image_file_name():
            copy_file(self.character_image_file_name, self.character_model_character_png_file_name())

        @file_task(workspace, self.character_model_face_morpher_file_name(), [
            f"{self.face_morpher_prefix()}/checkpoint/0010/module_module.pt",
        ])
        def copy_face_morpher():
            copy_file(
                f"{self.face_morpher_prefix()}/checkpoint/0010/module_module.pt",
                self.character_model_face_morpher_file_name())

        @file_task(workspace, self.character_model_body_morpher_file_name(), [
            f"{self.body_morpher_prefix()}/checkpoint/0015/module_module.pt",
        ])
        def copy_face_morpher():
            copy_file(
                f"{self.body_morpher_prefix()}/checkpoint/0015/module_module.pt",
                self.character_model_body_morpher_file_name())

        @file_task(workspace, self.character_model_yaml_file_name(), [])
        def create_character_model_yaml_file():
            character_model = CharacterModel(
                self.character_model_character_png_file_name(),
                self.character_model_face_morpher_file_name(),
                self.character_model_body_morpher_file_name())
            character_model.save(self.character_model_yaml_file_name())

        workspace.create_command_task(
            f"{self.prefix}/all",
            [
                f"{self.face_morpher_prefix()}/train_standalone",
                f"{self.body_morpher_prefix()}/train_standalone",
                self.character_model_character_png_file_name(),
                self.character_model_face_morpher_file_name(),
                self.character_model_body_morpher_file_name(),
                self.character_model_yaml_file_name(),
            ])
