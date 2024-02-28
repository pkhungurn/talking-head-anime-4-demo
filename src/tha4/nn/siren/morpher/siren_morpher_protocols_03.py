from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any

import torch
from tha4.shion.base.dataset.util import get_indexed_batch

from tha4.shion.core.cached_computation import output_array_indexing_func, add_step, ComputationState, \
    CachedComputationProtocol, ComposableCachedComputationProtocol, batch_indexing_func, proxy_func
from tha4.shion.core.loss import Loss
from tha4.shion.core.optimizer_factory import OptimizerFactory
from tha4.shion.core.training.sample_output_protocol import SampleOutputProtocol
from tha4.shion.core.training.training_protocol import AbstractTrainingProtocol
from tha4.nn.image_processing_util import GridChangeApplier
from tha4.nn.siren.morpher.siren_morpher_03 import SirenMorpher03
from tha4.poser.general_poser_02 import GeneralPoser02
from tha4.sampleoutput.sample_image_creator import SampleImageSpec, ImageSource, ImageType, SampleImageSaver
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset

KEY_MODULE = "module"
KEY_POSER = "poser"
KEY_EXAMPLES_SEEN_SO_FAR = "examples_seen_so_far"


@dataclass
class SirenMorpherProtocol03Keys:
    module: str = KEY_MODULE
    module_output: str = "module_output"

    poser: str = KEY_POSER
    poser_output: str = "poser_output"

    image: str = "image"
    pose: str = "pose"
    face_mask: str = 'face_mask'

    groundtruth_posed_image: str = 'groundtruth_posed_image'
    groundtruth_grid_change: str = 'groundtruth_grid_change'
    groundtruth_posed_face_mask: str = 'groundtruth_posed_face_mask'
    predicted_posed_image: str = 'predicted_posed_image'

    module_input_image: str = "module_input_image"

    predicted_grid_change: str = "predicted_grid_change"
    predicted_color_change: str = "predicted_color_change"
    predicted_warped_image: str = "predicted_warped_image"
    predicted_alpha: str = "predicted_alpha"

    groundtruth_alpha: str = "groundtruth_alpha"
    groundtruth_warped_image: str = "groundtruth_warped_image"

    zero: str = "zero"


@dataclass
class SirenMorpherProtocol03Indices:
    batch_image: int = 0
    batch_face_mask: int = 1
    batch_pose: int = 2
    poser_posed_image: int = 0
    poser_grid_change: int = 3

    poser_output_module_input_image_index: int = 5
    poser_alpha: int = 1
    poser_warped_image: int = 2

    module_blended_image: int = SirenMorpher03.INDEX_BLENDED_IMAGE
    module_grid_change: int = SirenMorpher03.INDEX_GRID_CHANGE
    module_color_change: int = SirenMorpher03.INDEX_COLOR_CHANGE
    module_warped_image: int = SirenMorpher03.INDEX_WARPED_IMAGE
    module_alpha: int = SirenMorpher03.INDEX_ALPHA


class SirenMorpherComputationProtocol03(ComposableCachedComputationProtocol):
    def __init__(self,
                 keys: Optional[SirenMorpherProtocol03Keys] = None,
                 indices: Optional[SirenMorpherProtocol03Indices] = None):
        super().__init__()

        if keys is None:
            keys = SirenMorpherProtocol03Keys()
        if indices is None:
            indices = SirenMorpherProtocol03Indices()

        self.keys = keys
        self.indices = indices

        self.computation_steps[keys.image] = batch_indexing_func(indices.batch_image)
        self.computation_steps[keys.pose] = batch_indexing_func(indices.batch_pose)
        self.computation_steps[keys.face_mask] = batch_indexing_func(indices.batch_face_mask)
        self.grid_change_applier = GridChangeApplier()

        @add_step(self.computation_steps, keys.module_output)
        def get_module_output(protocol: CachedComputationProtocol, state: ComputationState):
            pose = protocol.get_output(keys.pose, state)
            module = state.modules[self.keys.module]
            return module.forward(pose)

        self.computation_steps[keys.predicted_posed_image] = proxy_func(keys.module_output)

        @add_step(self.computation_steps, keys.poser_output)
        def get_poser_output(protocol: CachedComputationProtocol, state: ComputationState):
            with torch.no_grad():
                poser = state.modules[keys.poser]
                pose = protocol.get_output(keys.pose, state)
                image = protocol.get_output(keys.image, state)
                return poser.get_posing_outputs(image, pose)

        @add_step(self.computation_steps, keys.groundtruth_posed_image)
        def get_groundtruth_posed_image(protocol: CachedComputationProtocol, state: ComputationState):
            return protocol.get_output(keys.poser_output, state)[indices.poser_posed_image]

        @add_step(self.computation_steps, keys.groundtruth_grid_change)
        def get_groundtruth_posed_image(protocol: CachedComputationProtocol, state: ComputationState):
            return protocol.get_output(keys.poser_output, state)[indices.poser_grid_change]

        @add_step(self.computation_steps, keys.groundtruth_posed_face_mask)
        def get_groundtruth_posed_face_mask(protocol: CachedComputationProtocol, state: ComputationState):
            face_mask = protocol.get_output(keys.face_mask, state)
            groundtruth_grid_change = protocol.get_output(keys.groundtruth_grid_change, state)
            with torch.no_grad():
                return self.grid_change_applier.apply(groundtruth_grid_change, face_mask)

        @add_step(self.computation_steps, keys.module_input_image)
        def get_module_input_image(protocol: CachedComputationProtocol, state: ComputationState):
            poser_output = protocol.get_output(keys.poser_output, state)
            return poser_output[indices.poser_output_module_input_image_index]

        @add_step(self.computation_steps, keys.module_output)
        def get_module_output(protocol: CachedComputationProtocol, state: ComputationState):
            image = protocol.get_output(keys.module_input_image, state)
            pose = protocol.get_output(keys.pose, state)
            module = state.modules[self.keys.module]
            return module.forward(image, pose)

        self.computation_steps[keys.predicted_posed_image] = output_array_indexing_func(
            keys.module_output, indices.module_blended_image)
        self.computation_steps[keys.predicted_grid_change] = output_array_indexing_func(
            keys.module_output, indices.module_grid_change)
        self.computation_steps[keys.predicted_color_change] = output_array_indexing_func(
            keys.module_output, indices.module_color_change)
        self.computation_steps[keys.predicted_warped_image] = output_array_indexing_func(
            keys.module_output, indices.module_warped_image)
        self.computation_steps[keys.predicted_alpha] = output_array_indexing_func(
            keys.module_output, indices.module_alpha)

        self.computation_steps[keys.groundtruth_alpha] = output_array_indexing_func(
            keys.poser_output, indices.poser_alpha)
        self.computation_steps[keys.groundtruth_warped_image] = output_array_indexing_func(
            keys.poser_output, indices.poser_warped_image)

        @add_step(self.computation_steps, keys.zero)
        def get_zero(protocol: CachedComputationProtocol, state: ComputationState):
            pose = protocol.get_output(keys.pose, state)
            device = pose.device
            return torch.zeros(1, device=device)


class SirenMorpherTrainingProtocol03(AbstractTrainingProtocol):
    def __init__(self,
                 check_point_examples: List[int],
                 batch_size: int,
                 learning_rate: Callable[[int], Dict[str, float]],
                 optimizer_factories: Dict[str, OptimizerFactory],
                 random_seed: int,
                 poser_func: Callable[[], GeneralPoser02],
                 key_module: str,
                 key_poser: str = KEY_POSER,
                 key_examples_seen_so_far: str = KEY_EXAMPLES_SEEN_SO_FAR):
        super().__init__(check_point_examples, batch_size, learning_rate, optimizer_factories, random_seed)
        self.key_examples_seen_so_far = key_examples_seen_so_far
        self.key_poser = key_poser
        self.key_module = key_module
        self.poser_func = poser_func
        self.poser = None

    def run_training_iteration(
            self,
            batch: Any,
            examples_seen_so_far: int,
            modules: Dict[str, Module],
            accumulated_modules: Dict[str, Module],
            optimizers: Dict[str, Optimizer],
            losses: Dict[str, Loss],
            create_log_func: Optional[Callable[[str, int], Callable[[str, float], None]]],
            device: torch.device):
        if self.poser is None:
            self.poser = self.poser_func()
            self.poser.to(device)

        module = modules[self.key_module]
        module.train(True)
        module_optimizer = optimizers[self.key_module]
        module_optimizer.zero_grad(set_to_none=True)

        loss = losses[self.key_module]
        if create_log_func is not None:
            log_func = create_log_func(f"training_{self.key_module}", examples_seen_so_far)
        else:
            log_func = None
        state = ComputationState(
            modules={
                **modules,
                self.key_poser: self.poser,
            },
            accumulated_modules=accumulated_modules,
            batch=batch,
            outputs={
                self.key_examples_seen_so_far: examples_seen_so_far,
            })
        loss_value = loss.compute(state, log_func)
        loss_value.backward()
        module_optimizer.step()


class SirenMorpherSampleOutputProtocol(SampleOutputProtocol):
    def __init__(self,
                 num_images: int,
                 image_size: int,
                 examples_per_sample_output: int,
                 computation_protocol,
                 poser_func: Callable[[], GeneralPoser02],
                 random_seed: int = 54859395058,
                 batch_image_index: int = 0,
                 batch_pose_index: int = 2,
                 batch_size: Optional[int] = None,
                 sample_image_specs: Optional[List[SampleImageSpec]] = None,
                 cell_size: Optional[int] = None):
        if batch_size is None:
            batch_size = num_images

        if sample_image_specs is None:
            sample_image_specs = [
                SampleImageSpec(
                    ImageSource.BATCH,
                    computation_protocol.indices.poser_posed_image,
                    ImageType.COLOR),
                SampleImageSpec(
                    ImageSource.OUTPUT,
                    computation_protocol.indices.module_blended_image,
                    ImageType.COLOR),
                SampleImageSpec(
                    ImageSource.OUTPUT,
                    computation_protocol.indices.module_alpha,
                    ImageType.ALPHA),
                SampleImageSpec(
                    ImageSource.OUTPUT,
                    computation_protocol.indices.module_color_change,
                    ImageType.COLOR),
                SampleImageSpec(
                    ImageSource.OUTPUT,
                    computation_protocol.indices.module_warped_image,
                    ImageType.COLOR),
                SampleImageSpec(
                    ImageSource.BATCH,
                    computation_protocol.indices.poser_grid_change,
                    ImageType.GRID_CHANGE),
                SampleImageSpec(
                    ImageSource.OUTPUT,
                    computation_protocol.indices.module_grid_change,
                    ImageType.GRID_CHANGE),
            ]

        if cell_size is None:
            cell_size = image_size

        self.batch_size = batch_size
        self.batch_pose_index = batch_pose_index
        self.batch_image_index = batch_image_index
        self.poser_func = poser_func
        self.random_seed = random_seed
        self.examples_per_sample_output = examples_per_sample_output
        self.image_size = image_size
        self.num_images = num_images
        self.computation_protocol = computation_protocol
        self.cell_size = cell_size

        self.sample_image_saver = SampleImageSaver(image_size, cell_size, 4, sample_image_specs)

    def get_examples_per_sample_output(self) -> int:
        return self.examples_per_sample_output

    def get_random_seed(self) -> int:
        return self.random_seed

    def get_sample_output_data(self, validation_dataset: Dataset, device: torch.device) -> dict:
        example_indices = torch.randint(0, len(validation_dataset), (self.num_images,))
        example_indices = [example_indices[i].item() for i in range(self.num_images)]
        batch = get_indexed_batch(validation_dataset, example_indices, device)
        poser = self.poser_func()
        poser.to(device)
        with torch.no_grad():
            ground_truth = poser.get_posing_outputs(batch[self.batch_image_index], batch[self.batch_pose_index])
        return {
            'batch': batch,
            'ground_truth': ground_truth
        }

    def save_sample_output_data(self,
                                modules: Dict[str, Module],
                                accumulated_modules: Dict[str, Module],
                                sample_output_data: Any,
                                prefix: str,
                                examples_seen_so_far: int,
                                device: torch.device):
        batch = sample_output_data['batch']
        ground_truth = sample_output_data['ground_truth']

        module = modules[self.computation_protocol.keys.module]
        module.train(False)
        if self.batch_size == self.num_images:
            with torch.no_grad():
                state = ComputationState(
                    modules=modules,
                    accumulated_modules=accumulated_modules,
                    batch=batch,
                    outputs={
                        self.computation_protocol.keys.poser_output: ground_truth,
                    })
                module_outputs = self.computation_protocol.get_output(
                    self.computation_protocol.keys.module_output, state)
        else:
            module_outputs_list = []
            start = 0
            while start < self.num_images:
                end = start + self.batch_size
                end = min(self.num_images, end)
                minibatch = [batch[i][start:end] for i in range(len(batch))]
                ground_truth_batch = [ground_truth[i][start:end] for i in range(len(ground_truth))]
                state = ComputationState(
                    modules=modules,
                    accumulated_modules=accumulated_modules,
                    batch=minibatch,
                    outputs={
                        self.computation_protocol.keys.poser_output: ground_truth_batch
                    })
                with torch.no_grad():
                    module_outputs = self.computation_protocol.get_output(
                        self.computation_protocol.keys.module_output, state)
                module_outputs_list.append(module_outputs)
                start = end

            module_outputs = []
            for i in range(len(module_outputs_list[0])):
                tensor_list = []
                for j in range(len(module_outputs_list)):
                    tensor_list.append(module_outputs_list[j][i])
                module_output = torch.cat(tensor_list, dim=0)
                module_outputs.append(module_output)

        self.sample_image_saver.save_sample_output_data(ground_truth, module_outputs, prefix, examples_seen_so_far)
