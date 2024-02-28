from typing import Optional, List, Dict

import torch
from torch.nn import Module
from torch.utils.data import Dataset

from tha4.shion.base.dataset.util import get_indexed_batch
from tha4.shion.core.cached_computation import CachedComputationFunc, ComputationState
from tha4.shion.core.training.sample_output_protocol import SampleOutputProtocol
from tha4.sampleoutput.sample_image_creator import SampleImageSpec, ImageSource, ImageType, SampleImageSaver


class PoserSampleOutputProtocol(SampleOutputProtocol):
    def __init__(self,
                 output_list_func: Optional[CachedComputationFunc] = None,
                 num_images: int = 8,
                 image_size: int = 256,
                 cell_size: int = 256,
                 image_channels: int = 4,
                 examples_per_sample_output: int = 5000,
                 sample_image_specs: Optional[List[SampleImageSpec]] = None,
                 random_seed: int = 1203040687):
        super().__init__()
        self.num_images = num_images
        self.random_seed = random_seed
        self.examples_per_sample_output = examples_per_sample_output
        self.output_list_func = output_list_func

        if sample_image_specs is None:
            sample_image_specs = [
                SampleImageSpec(ImageSource.BATCH, 0, ImageType.COLOR),
                SampleImageSpec(ImageSource.BATCH, 2, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 0, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 1, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 2, ImageType.ALPHA),
                SampleImageSpec(ImageSource.BATCH, 3, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 3, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 4, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 5, ImageType.ALPHA),
                SampleImageSpec(ImageSource.OUTPUT, 6, ImageType.COLOR),
                SampleImageSpec(ImageSource.BATCH, 4, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 7, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 8, ImageType.COLOR),
                SampleImageSpec(ImageSource.OUTPUT, 9, ImageType.ALPHA),
                SampleImageSpec(ImageSource.OUTPUT, 10, ImageType.COLOR),
            ]

        self.sample_image_saver = SampleImageSaver(image_size, cell_size, image_channels, sample_image_specs)

    def get_examples_per_sample_output(self) -> int:
        return self.examples_per_sample_output

    def get_random_seed(self) -> int:
        return self.random_seed

    def get_sample_output_data(self, validation_dataset: Dataset, device: torch.device) -> dict:
        example_indices = torch.randint(0, len(validation_dataset), (self.num_images,))
        example_indices = [example_indices[i].item() for i in range(self.num_images)]
        batch = get_indexed_batch(validation_dataset, example_indices, device)
        return {'batch': batch}

    def save_sample_output_data(self,
                                modules: Dict[str, Module],
                                accumulated_modules: Dict[str, Module],
                                sample_output_data: dict, prefix: str,
                                examples_seen_so_far: int,
                                device: torch.device):
        for key in modules:
            modules[key].train(False)
        batch = sample_output_data['batch']
        with torch.no_grad():
            outputs = self.output_list_func(ComputationState(modules, accumulated_modules, batch))
        self.sample_image_saver.save_sample_output_data(batch, outputs, prefix, examples_seen_so_far)
