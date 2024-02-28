from enum import Enum
from typing import List, Dict, Optional, Any

import torch
from tha4.shion.core.cached_computation import CachedComputationProtocol, ComputationState
from tha4.shion.core.load_save import torch_load
from tha4.nn.eyebrow_decomposer.eyebrow_decomposer_00 import EyebrowDecomposer00, \
    EyebrowDecomposer00Factory, EyebrowDecomposer00Args
from tha4.nn.eyebrow_morphing_combiner.eyebrow_morphing_combiner_00 import \
    EyebrowMorphingCombiner00Factory, EyebrowMorphingCombiner00Args, EyebrowMorphingCombiner00
from tha4.nn.face_morpher.face_morpher_08 import FaceMorpher08Args, FaceMorpher08Factory
from tha4.nn.nonlinearity_factory import ReLUFactory
from tha4.nn.normalization import InstanceNorm2dFactory
from tha4.nn.util import BlockArgs
from tha4.poser.general_poser_02 import GeneralPoser02
from tha4.poser.modes.pose_parameters import get_pose_parameters
from torch import Tensor


class Network(Enum):
    eyebrow_decomposer = 1
    eyebrow_morphing_combiner = 2
    face_morpher = 3

    @property
    def outputs_key(self):
        return f"{self.name}_outputs"


class Branch(Enum):
    face_morphed_half = 1
    face_morphed_full = 2
    all_outputs = 3


NUM_EYEBROW_PARAMS = 12
NUM_FACE_PARAMS = 27
NUM_ROTATION_PARAMS = 6


class FiveStepPoserComputationProtocol(CachedComputationProtocol):
    def __init__(self, eyebrow_morphed_image_index: int):
        super().__init__()
        self.eyebrow_morphed_image_index = eyebrow_morphed_image_index
        self.cached_batch_0 = None
        self.cached_eyebrow_decomposer_output = None

    def compute_func(self):
        def func(state: ComputationState) -> List[Tensor]:
            if self.cached_batch_0 is None:
                new_batch_0 = True
            elif state.batch[0].shape[0] != self.cached_batch_0.shape[0]:
                new_batch_0 = True
            else:
                new_batch_0 = torch.max((state.batch[0] - self.cached_batch_0).abs()).item() > 0
            if not new_batch_0:
                state.outputs[Network.eyebrow_decomposer.outputs_key] = self.cached_eyebrow_decomposer_output
            output = self.get_output(Branch.all_outputs.name, state)
            if new_batch_0:
                self.cached_batch_0 = state.batch[0]
                self.cached_eyebrow_decomposer_output = state.outputs[Network.eyebrow_decomposer.outputs_key]
            return output

        return func

    def compute_output(self, key: str, state: ComputationState) -> Any:
        if key == Network.eyebrow_decomposer.outputs_key:
            input_image = state.batch[0][:, :, 64:192, 64 + 128:192 + 128]
            return state.modules[Network.eyebrow_decomposer.name].forward(input_image)
        elif key == Network.eyebrow_morphing_combiner.outputs_key:
            eyebrow_decomposer_output = self.get_output(Network.eyebrow_decomposer.outputs_key, state)
            background_layer = eyebrow_decomposer_output[EyebrowDecomposer00.BACKGROUND_LAYER_INDEX]
            eyebrow_layer = eyebrow_decomposer_output[EyebrowDecomposer00.EYEBROW_LAYER_INDEX]
            eyebrow_pose = state.batch[1][:, :NUM_EYEBROW_PARAMS]
            return state.modules[Network.eyebrow_morphing_combiner.name].forward(
                background_layer,
                eyebrow_layer,
                eyebrow_pose)
        elif key == Network.face_morpher.outputs_key:
            eyebrow_morphing_combiner_output = self.get_output(
                Network.eyebrow_morphing_combiner.outputs_key, state)
            eyebrow_morphed_image = eyebrow_morphing_combiner_output[self.eyebrow_morphed_image_index]
            input_image = state.batch[0][:, :, 32:32 + 192, (32 + 128):(32 + 192 + 128)].clone()
            input_image[:, :, 32:32 + 128, 32:32 + 128] = eyebrow_morphed_image
            face_pose = state.batch[1][:, NUM_EYEBROW_PARAMS:NUM_EYEBROW_PARAMS + NUM_FACE_PARAMS]
            return state.modules[Network.face_morpher.name].forward(input_image, face_pose)
        elif key == Branch.all_outputs.name:
            face_morpher_output = self.get_output(Network.face_morpher.outputs_key, state)
            eyebrow_morphing_combiner_output = self.get_output(Network.eyebrow_morphing_combiner.outputs_key, state)
            eyebrow_decomposer_output = self.get_output(Network.eyebrow_decomposer.outputs_key, state)
            output = face_morpher_output \
                     + eyebrow_morphing_combiner_output \
                     + eyebrow_decomposer_output
            return output
        else:
            raise RuntimeError("Unsupported key: " + key)


def load_eyebrow_decomposer(file_name: str):
    factory = EyebrowDecomposer00Factory(
        EyebrowDecomposer00Args(
            image_size=128,
            image_channels=4,
            start_channels=64,
            bottleneck_image_size=16,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))))
    print("Loading the eyebrow decomposer ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    print("DONE!!!")
    return module


def load_eyebrow_morphing_combiner(file_name: str):
    factory = EyebrowMorphingCombiner00Factory(
        EyebrowMorphingCombiner00Args(
            image_size=128,
            image_channels=4,
            start_channels=64,
            num_pose_params=12,
            bottleneck_image_size=16,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))))
    print("Loading the eyebrow morphing conbiner ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    print("DONE!!!")
    return module


def load_face_morpher(file_name: str):
    factory = FaceMorpher08Factory(
        FaceMorpher08Args(
            image_size=192,
            image_channels=4,
            num_expression_params=27,
            start_channels=64,
            bottleneck_image_size=24,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=False)),
            output_iris_mouth_grid_change=True))
    print("Loading the face morpher ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    print("DONE!!!")
    return module


def apply_color_change(alpha, color_change, image: Tensor) -> Tensor:
    return color_change * alpha + image * (1 - alpha)


def create_poser(
        device: torch.device,
        module_file_names: Optional[Dict[str, str]] = None,
        eyebrow_morphed_image_index: int = EyebrowMorphingCombiner00.EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX,
        default_output_index: int = 0) -> GeneralPoser02:
    if module_file_names is None:
        module_file_names = {}
    if Network.eyebrow_decomposer.name not in module_file_names:
        file_name = "data/tha4/eyebrow_decomposer.pt"
        module_file_names[Network.eyebrow_decomposer.name] = file_name
    if Network.eyebrow_morphing_combiner.name not in module_file_names:
        file_name = "data/tha4/eyebrow_morphing_combiner.pt"
        module_file_names[Network.eyebrow_morphing_combiner.name] = file_name
    if Network.face_morpher.name not in module_file_names:
        file_name = "data/tha4/face_morpher.pt"
        module_file_names[Network.face_morpher.name] = file_name

    loaders = {
        Network.eyebrow_decomposer.name:
            lambda: load_eyebrow_decomposer(module_file_names[Network.eyebrow_decomposer.name]),
        Network.eyebrow_morphing_combiner.name:
            lambda: load_eyebrow_morphing_combiner(module_file_names[Network.eyebrow_morphing_combiner.name]),
        Network.face_morpher.name:
            lambda: load_face_morpher(module_file_names[Network.face_morpher.name]),
    }
    return GeneralPoser02(
        image_size=512,
        module_loaders=loaders,
        pose_parameters=get_pose_parameters().get_pose_parameter_groups(),
        output_list_func=FiveStepPoserComputationProtocol(eyebrow_morphed_image_index).compute_func(),
        subrect=None,
        device=device,
        output_length=5 + 5 + 8,
        default_output_index=default_output_index)
