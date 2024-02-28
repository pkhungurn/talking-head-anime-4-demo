from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from tha4.shion.core.cached_computation import CachedComputationProtocol, ComputationState
from tha4.shion.core.load_save import torch_load
from tha4.nn.siren.face_morpher.siren_face_morpher_00 import SirenFaceMorpher00Args, SirenFaceMorpher00
from tha4.nn.siren.morpher.siren_morpher_03 import SirenMorpher03, SirenMorpher03Args, SirenMorpherLevelArgs
from tha4.nn.siren.vanilla.siren import SirenArgs
from tha4.poser.general_poser_02 import GeneralPoser02
from tha4.poser.modes.pose_parameters import get_pose_parameters
from torch import Tensor

KEY_FACE_MORPHER = "face_morpher"
KEY_BODY_MORPHER = "body_morpher"


@dataclass
class Keys:
    face_morpher: str = KEY_FACE_MORPHER
    face_morpher_output: str = "face_morpher_output"

    face_morpher_input_image: str = "face_morpher_input_image"
    face_morpher_input_pose: str = "face_morpher_input_pose"

    body_morpher_input_image: str = "body_morpher_input_image"

    body_morpher: str = KEY_BODY_MORPHER
    body_morpher_output: str = "body_morpher_output"

    all_outputs: str = "all_outputs"


@dataclass
class Indices:
    original_image: int = 0
    original_pose: int = 1


class TwoStepPoserComputationProtocol(CachedComputationProtocol):
    def __init__(self, keys: Optional[Keys] = None, indices: Optional[Indices] = None):
        super().__init__()

        if keys is None:
            keys = Keys()
        if indices is None:
            indices = Indices()

        self.keys = keys
        self.indices = indices

    def compute_func(self):
        def func(state: ComputationState) -> List[Tensor]:
            return self.get_output(self.keys.all_outputs, state)

        return func

    def compute_output(self, key: str, state: ComputationState) -> Any:
        if key == self.keys.face_morpher_input_image:
            image = state.batch[self.indices.original_image]
            center_x = 256
            center_y = 128 + 16
            return image[:, :, center_y - 64:center_y + 64, center_x - 64:center_x + 64]
        elif key == self.keys.face_morpher_input_pose:
            pose = state.batch[self.indices.original_pose]
            return pose[:, 0:39]
        elif key == self.keys.face_morpher_output:
            module = state.modules[self.keys.face_morpher]
            pose = self.get_output(self.keys.face_morpher_input_pose, state)
            with torch.no_grad():
                return module.forward(pose)
        elif key == self.keys.body_morpher_input_image:
            image = state.batch[self.indices.original_image].clone()
            center_x = 256
            center_y = 128 + 16
            face_morphed_image = self.get_output(self.keys.face_morpher_output, state)
            image[:, :, center_y - 64:center_y + 64, center_x - 64:center_x + 64] = face_morphed_image
            return image
        elif key == self.keys.body_morpher_output:
            image = self.get_output(self.keys.body_morpher_input_image, state)
            pose = state.batch[self.indices.original_pose]
            body_morpher = state.modules[self.keys.body_morpher]
            with torch.no_grad():
                return body_morpher.forward(image, pose)
        elif key == self.keys.all_outputs:
            body_morpher_output = self.get_output(self.keys.body_morpher_output, state)
            face_morpher_output = self.get_output(self.keys.face_morpher_output, state)
            return body_morpher_output + [face_morpher_output]
        else:
            raise RuntimeError("Unsupported key: " + key)


def load_face_morpher(file_name: Optional[str] = None):
    module = SirenFaceMorpher00(
        SirenFaceMorpher00Args(
            image_size=128,
            image_channels=4,
            pose_size=39,
            siren_args=SirenArgs(
                in_channels=39 + 2,
                out_channels=4,
                intermediate_channels=128,
                num_sine_layers=8)))
    if file_name is not None:
        module.load_state_dict(torch_load(file_name))
    return module


def load_body_morpher(file_name: Optional[str] = None):
    module = SirenMorpher03(
        SirenMorpher03Args(
            image_size=512,
            image_channels=4,
            pose_size=45,
            level_args=[
                SirenMorpherLevelArgs(
                    image_size=128,
                    intermediate_channels=360,
                    num_sine_layers=3),
                SirenMorpherLevelArgs(
                    image_size=256,
                    intermediate_channels=180,
                    num_sine_layers=3),
                SirenMorpherLevelArgs(
                    image_size=512,
                    intermediate_channels=90,
                    num_sine_layers=3),
            ]))
    if file_name is not None:
        module.load_state_dict(torch_load(file_name))
    return module


def create_poser(
        device: torch.device,
        module_file_names: Optional[Dict[str, str]] = None,
        default_output_index: int = 0) -> GeneralPoser02:
    if module_file_names is None:
        module_file_names = {}
    if KEY_FACE_MORPHER not in module_file_names:
        file_name = "data/character_models/lambda_00/face_morpher.pt"
        module_file_names[KEY_FACE_MORPHER] = file_name
    if KEY_BODY_MORPHER not in module_file_names:
        file_name = "data/character_models/lambda_00/body_morpher.pt"
        module_file_names[KEY_BODY_MORPHER] = file_name

    loaders = {
        KEY_FACE_MORPHER:
            lambda: load_face_morpher(module_file_names[KEY_FACE_MORPHER]),
        KEY_BODY_MORPHER:
            lambda: load_body_morpher(module_file_names[KEY_BODY_MORPHER]),
    }

    return GeneralPoser02(
        image_size=512,
        module_loaders=loaders,
        pose_parameters=get_pose_parameters().get_pose_parameter_groups(),
        output_list_func=TwoStepPoserComputationProtocol().compute_func(),
        subrect=None,
        device=device,
        output_length=5 + 1,
        default_output_index=default_output_index)
