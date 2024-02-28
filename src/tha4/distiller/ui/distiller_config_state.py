import os.path
from contextlib import contextmanager
from pathlib import PurePath, Path
from typing import Callable, Any, Optional

from tha4.distiller.distiller_config import DistillerConfig


class DistillerConfigState:
    def __init__(self):
        self.config = DistillerConfig(prefix="", character_image_file_name="", face_mask_image_file_name="")
        self.last_saved_timestamp = None
        self.dirty = False

    def load(self, file_name):
        self.config = DistillerConfig.load(file_name)
        if os.path.exists(self.config.config_yaml_file_name()):
            self.last_saved_timestamp = os.path.getmtime(self.config.config_yaml_file_name())
        else:
            self.last_saved_timestamp = None
        self.dirty = False

    def need_to_check_overwrite(self):
        if self.last_saved_timestamp is None:
            return True
        if not os.path.exists(self.config.config_yaml_file_name()):
            return False
        if self.last_saved_timestamp < os.path.getmtime(self.config.config_yaml_file_name()):
            return True
        return False

    def save(self):
        self.config.save(self.config.config_yaml_file_name())
        self.dirty = False
        self.last_saved_timestamp = os.path.getmtime(self.config.config_yaml_file_name())

    @contextmanager
    def updating_value(self, value_func: Callable[[], Any]):
        old_value = value_func()
        yield
        new_value = value_func()
        if new_value != old_value:
            self.dirty = True

    def set_prefix(self, new_value):
        with self.updating_value(lambda: self.config.prefix):
            new_relative_path = self.get_relative_path_to_cwd(
                new_value,
                "The prefix directory must be a subdirectory of the talking-head-anime-4-demo's source code directory.")
            DistillerConfig.check_prefix(new_relative_path)
            self.config.prefix = new_relative_path

    def set_character_image_file_name(self, new_value):
        with self.updating_value(lambda: self.config.character_image_file_name):
            new_relative_path = self.get_relative_path_to_cwd(
                new_value,
                "The character image file must be under talking-head-anime-4-demo's source code directory.")
            DistillerConfig.check_character_image_file_name(new_relative_path)
            self.config.character_image_file_name = new_relative_path

    def set_face_mask_image_file_name(self, new_value):
        with self.updating_value(lambda: self.config.face_mask_image_file_name):
            new_relative_path = self.get_relative_path_to_cwd(
                new_value,
                "The face mask image file must be under talking-head-anime-4-demo's source code directory.")
            DistillerConfig.check_face_mask_image_file_name(new_relative_path)
            self.config.face_mask_image_file_name = new_relative_path

    def set_num_cpu_workers(self, new_value: int):
        with self.updating_value(lambda: self.config.num_cpu_workers):
            DistillerConfig.check_num_cpu_workers(new_value)
            self.config.num_cpu_workers = new_value

    def set_num_gpus(self, new_value: int):
        with self.updating_value(lambda: self.config.num_gpus):
            DistillerConfig.check_num_cpu_workers(new_value)
            self.config.num_gpus = new_value

    def set_face_morpher_random_seed_0(self, new_value: int):
        with self.updating_value(lambda: self.config.face_morpher_random_seed_0):
            DistillerConfig.check_random_seed(new_value, "face_morpher_random_seed_0")
            self.config.face_morpher_random_seed_0 = new_value

    def set_face_morpher_random_seed_1(self, new_value: int):
        with self.updating_value(lambda: self.config.face_morpher_random_seed_1):
            DistillerConfig.check_random_seed(new_value, "face_morpher_random_seed_1")
            self.config.face_morpher_random_seed_1 = new_value

    def set_face_morpher_num_training_examples_per_sample_output(self, new_value: Optional[int]):
        with self.updating_value(lambda: self.config.face_morpher_num_training_examples_per_sample_output):
            DistillerConfig.check_num_training_examples_per_sample_output(
                new_value, "face_morpher_num_training_examples_per_sample_output")
            self.config.face_morpher_num_training_examples_per_sample_output = new_value

    def set_face_morpher_batch_size(self, new_value: int):
        with self.updating_value(lambda: self.config.face_morpher_batch_size):
            DistillerConfig.check_batch_size(new_value, "face_morpher_batch_size")
            self.config.face_morpher_batch_size = new_value

    def set_body_morpher_random_seed_0(self, new_value: int):
        with self.updating_value(lambda: self.config.body_morpher_random_seed_0):
            DistillerConfig.check_random_seed(new_value, "body_morpher_random_seed_0")
            self.config.body_morpher_random_seed_0 = new_value

    def set_body_morpher_random_seed_1(self, new_value: int):
        with self.updating_value(lambda: self.config.body_morpher_random_seed_1):
            DistillerConfig.check_random_seed(new_value, "body_morpher_random_seed_1")
            self.config.body_morpher_random_seed_1 = new_value

    def set_body_morpher_num_training_examples_per_sample_output(self, new_value: Optional[int]):
        with self.updating_value(lambda: self.config.body_morpher_num_training_examples_per_sample_output):
            DistillerConfig.check_num_training_examples_per_sample_output(
                new_value, "body_morpher_num_training_examples_per_sample_output")
            self.config.body_morpher_num_training_examples_per_sample_output = new_value

    def set_body_morpher_batch_size(self, new_value: int):
        with self.updating_value(lambda: self.config.body_morpher_batch_size):
            DistillerConfig.check_batch_size(new_value, "body_morpher_batch_size")
            self.config.body_morpher_batch_size = new_value

    def get_relative_path_to_cwd(self, file_name: str, message: str):
        cwd = os.getcwd()
        assert os.path.commonprefix([cwd, file_name]) == cwd, message
        cwd_path = Path(cwd).as_posix()
        new_path = Path(file_name).as_posix()
        new_relative_path = os.path.relpath(str(new_path), cwd_path)
        new_relative_path = str(Path(new_relative_path).as_posix())
        return new_relative_path

    def can_show_character_image(self):
        return os.path.isfile(self.config.character_image_file_name)

    def can_show_face_mask_image(self):
        return os.path.isfile(self.config.face_mask_image_file_name)

    def can_show_mask_on_face_image(self):
        return self.can_show_character_image() and self.can_show_face_mask_image()

    def can_save(self):
        return os.path.isdir(self.config.prefix) \
            and os.path.isfile(self.config.character_image_file_name) \
            and os.path.isfile(self.config.face_mask_image_file_name)
