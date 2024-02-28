import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable

import PIL.Image
import numpy
import torch
from tha4.shion.base.dataset.util import get_indexed_batch
from tha4.shion.base.image_util import pytorch_rgba_to_numpy_image
from tha4.shion.core.cached_computation import CachedComputationProtocol, ComputationState, \
    ComposableCachedComputationProtocol, batch_indexing_func, add_step
from tha4.shion.core.training.sample_output_protocol import SampleOutputProtocol
from tha4.poser.general_poser_02 import GeneralPoser02
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

KEY_MODULE = "module"
KEY_POSER = "poser"


@dataclass
class SirenMorpherProtocol00Keys:
    module: str = KEY_MODULE
    module_output: str = "module_output"

    poser: str = KEY_POSER
    poser_output: str = "poser_output"

    original_image: str = "original_image"
    original_pose: str = "original_pose"

    module_input_image: str = "module_input_image"
    module_input_pose: str = "module_input_pose"

    groundtruth_posed_image: str = 'groundtruth_posed_image'
    predicted_posed_image: str = 'predicted_posed_image'

    eye_mouth_mask: str = 'eye_mouth_mask'


@dataclass
class SirenMorpherProtocol00Indices:
    batch_original_image: int = 0
    batch_pose: int = 1
    batch_eye_mouth_mask: int = 2
    poser_posed_image: int = 0


class SirenFaceMorpherComputationProtocol00(ComposableCachedComputationProtocol):
    def __init__(self,
                 transform_pose_to_module_input_func: Callable[[Tensor], Tensor],
                 transform_original_image_to_module_input_func: Callable[[Tensor], Tensor],
                 transform_poser_posed_image_to_groundtruth_func: Callable[[Tensor], Tensor],
                 keys: Optional[SirenMorpherProtocol00Keys] = None,
                 indices: Optional[SirenMorpherProtocol00Indices] = None):
        super().__init__()

        if keys is None:
            keys = SirenMorpherProtocol00Keys()
        if indices is None:
            indices = SirenMorpherProtocol00Indices()

        self.keys = keys
        self.indices = indices
        self.transform_image_to_module_input_func = transform_original_image_to_module_input_func
        self.transform_pose_to_module_input_func = transform_pose_to_module_input_func
        self.transform_poser_posed_image_to_groundtruth_func = transform_poser_posed_image_to_groundtruth_func

        self.computation_steps[keys.original_image] = batch_indexing_func(indices.batch_original_image)
        self.computation_steps[keys.original_pose] = batch_indexing_func(indices.batch_pose)

        @add_step(self.computation_steps, keys.module_input_pose)
        def get_module_input_pose(protocol: CachedComputationProtocol, state: ComputationState):
            original_pose = protocol.get_output(keys.original_pose, state)
            return transform_pose_to_module_input_func(original_pose)

        @add_step(self.computation_steps, keys.module_input_image)
        def get_module_input_image(protocol: CachedComputationProtocol, state: ComputationState):
            original_image = protocol.get_output(keys.original_image, state)
            return transform_original_image_to_module_input_func(original_image)

        @add_step(self.computation_steps, keys.poser_output)
        def get_poser_output(protocol: CachedComputationProtocol, state: ComputationState):
            with torch.no_grad():
                poser = state.modules[keys.poser]
                pose = protocol.get_output(keys.original_pose, state)
                image = protocol.get_output(keys.original_image, state)
                return poser.get_posing_outputs(image, pose)

        @add_step(self.computation_steps, keys.groundtruth_posed_image)
        def get_groundtruth_posed_image(protocol: CachedComputationProtocol, state: ComputationState):
            poser_output = protocol.get_output(keys.poser_output, state)
            poser_posed_image = poser_output[indices.poser_posed_image]
            return transform_poser_posed_image_to_groundtruth_func(poser_posed_image)

        @add_step(self.computation_steps, keys.module_output)
        def get_module_output(protocol: CachedComputationProtocol, state: ComputationState):
            module_input_pose = protocol.get_output(keys.module_input_pose, state)
            module = state.modules[keys.module]
            return module.forward(module_input_pose)

        @add_step(self.computation_steps, keys.predicted_posed_image)
        def get_predicted_image(protocol: CachedComputationProtocol, state: ComputationState):
            return protocol.get_output(keys.module_output, state)

        self.computation_steps[keys.eye_mouth_mask] = batch_indexing_func(indices.batch_eye_mouth_mask)


class SirenFaceMorpherSampleOutputProtocol00(SampleOutputProtocol):
    def __init__(self,
                 num_images: int,
                 image_size: int,
                 images_per_row: int,
                 examples_per_sample_output: int,
                 computation_protocol: SirenFaceMorpherComputationProtocol00,
                 poser_func: Callable[[], GeneralPoser02],
                 random_seed: int = 54859395058,
                 batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = num_images

        self.batch_size = batch_size
        self.poser_func = poser_func
        self.random_seed = random_seed
        self.examples_per_sample_output = examples_per_sample_output
        self.images_per_row = images_per_row
        self.image_size = image_size
        self.num_images = num_images
        self.computation_protocol = computation_protocol

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
            ground_truth = poser.pose(
                batch[self.computation_protocol.indices.batch_original_image],
                batch[self.computation_protocol.indices.batch_pose])
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
        ground_truth = self.computation_protocol.transform_poser_posed_image_to_groundtruth_func(ground_truth)

        module = modules[self.computation_protocol.keys.module]
        module.train(False)

        if self.batch_size == self.num_images:
            with torch.no_grad():
                state = ComputationState(
                    modules=modules,
                    accumulated_modules=accumulated_modules,
                    batch=batch,
                    outputs={})
                poser_output_images = self.computation_protocol.get_output(
                    self.computation_protocol.keys.predicted_posed_image, state)
        else:
            poser_output_images_list = []
            start = 0
            while start < self.num_images:
                end = start + self.batch_size
                end = min(self.num_images, end)
                minibatch = [batch[i][start:end] for i in range(len(batch))]
                state = ComputationState(
                    modules=modules,
                    accumulated_modules=accumulated_modules,
                    batch=minibatch,
                    outputs={})
                with torch.no_grad():
                    poser_output_images = self.computation_protocol.get_output(
                        self.computation_protocol.keys.predicted_posed_image, state)
                poser_output_images_list.append(poser_output_images)
                start = end
            poser_output_images = torch.cat(poser_output_images_list, dim=0)

        num_rows = self.num_images // self.images_per_row
        if self.num_images % self.images_per_row > 0:
            num_rows += 1
        num_cols = 2 * self.images_per_row

        image_channels = 4
        output_image = numpy.zeros([self.image_size * num_rows, self.image_size * num_cols, image_channels])

        for image_index in range(self.num_images):
            row = image_index // self.images_per_row
            start_row = row * self.image_size

            col = 2 * (image_index % self.images_per_row)
            start_col = col * self.image_size
            output_image[start_row:start_row + self.image_size, start_col:start_col + self.image_size, :] \
                = pytorch_rgba_to_numpy_image(ground_truth[image_index].detach().cpu())

            start_col += self.image_size
            output_image[start_row:start_row + self.image_size, start_col:start_col + self.image_size, :] \
                = pytorch_rgba_to_numpy_image(poser_output_images[image_index].detach().cpu())

        file_name = "%s/sample_output_%010d.png" % (prefix, examples_seen_so_far)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(output_image * 255.0)), mode='RGBA')
        pil_image.save(file_name)
        print("Saved %s" % file_name)
