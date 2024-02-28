from typing import Dict, List, Optional, Callable

import torch
from tha4.shion.base.dataset.lazy_tensor_dataset import LazyTensorDataset
from tha4.shion.base.image_util import extract_pytorch_image_from_filelike
from tha4.shion.base.loss.l1_loss import L1Loss, MaskedL1Loss
from tha4.shion.base.loss.sum_loss import SumLoss
from tha4.shion.base.optimizer_factories import AdamOptimizerFactory
from tha4.shion.core.training.distrib.distributed_trainer import DistributedTrainer
from tha4.dataset.image_poses_and_aother_images_dataset import ImagePosesAndOtherImagesDataset
from tha4.nn.siren.face_morpher.siren_face_morpher_00 import SirenFaceMorpher00Factory, SirenFaceMorpher00Args
from tha4.nn.siren.face_morpher.siren_face_morpher_protocols_00 import SirenFaceMorpherComputationProtocol00, \
    SirenFaceMorpherSampleOutputProtocol00
from tha4.nn.siren.morpher.siren_morpher_protocols_03 import SirenMorpherTrainingProtocol03
from tha4.nn.siren.vanilla.siren import SirenArgs
from tha4.poser.poser import Poser
from torch import Tensor

KEY_MODULE = "module"
KEY_POSER = "poser"


def get_poser():
    import tha4.poser.modes.mode_12
    poser = tha4.poser.modes.mode_12.create_poser(torch.device('cpu'))
    return poser


class SirenFaceMorpher00TrainerArgs:
    def __init__(self,
                 character_file_name: str,
                 face_mask_file_name: str,
                 pose_dataset_file_name: str,
                 num_training_total_examples: int = 1_000_000,
                 num_training_examples_per_checkpoint: int = 100_000,
                 num_training_examples_lr_boundaries: Optional[List[int]] = None,
                 num_training_examples_per_sample_output: Optional[int] = 5_000,
                 num_training_examples_per_snapshot: int = 10_000,
                 total_batch_size: int = 8,
                 training_random_seed: int = 2965603729,
                 sample_output_random_seed: int = 3522651501,
                 total_worker: int = 16,
                 poser_func: Optional[Callable[[], Poser]] = None,
                 base_learning_rate: float = 1e-4):
        assert num_training_total_examples % num_training_examples_per_checkpoint == 0

        if num_training_examples_lr_boundaries is None:
            num_training_examples_lr_boundaries = [
                int(num_training_examples_per_checkpoint * 2),
                int(num_training_examples_per_checkpoint * 5),
                int(num_training_examples_per_checkpoint * 8),
            ]

        for x in num_training_examples_lr_boundaries:
            assert x % num_training_examples_per_snapshot == 0

        if poser_func is None:
            poser_func = get_poser

        self.face_mask_file_name = face_mask_file_name
        self.base_learning_rate = base_learning_rate
        self.poser_func = poser_func
        self.total_worker = total_worker
        self.num_training_examples_per_snapshot = num_training_examples_per_snapshot
        self.num_training_examples_per_sample_output = num_training_examples_per_sample_output
        self.sample_output_random_seed = sample_output_random_seed
        self.training_random_seed = training_random_seed
        self.total_batch_size = total_batch_size
        self.num_training_total_examples = num_training_total_examples
        self.num_training_examples_per_checkpoint = num_training_examples_per_checkpoint
        self.num_training_examples_lr_boundaries = num_training_examples_lr_boundaries
        self.pose_dataset_file_name = pose_dataset_file_name
        self.character_file_name = character_file_name

    def get_character_image(self):
        return extract_pytorch_image_from_filelike(
            self.character_file_name,
            scale=2.0,
            offset=-1.0,
            premultiply_alpha=True,
            perform_srgb_to_linear=True)

    def get_face_mask_image(self):
        loaded_image = extract_pytorch_image_from_filelike(
            self.face_mask_file_name,
            scale=1.0,
            offset=0.0,
            premultiply_alpha=True,
            perform_srgb_to_linear=True)
        output_image = torch.zeros(4, 128, 128)
        center_x = 256
        center_y = 128 + 16
        for i in range(4):
            output_image[i, :, :] = loaded_image[0, center_y - 64:center_y + 64, center_x - 64:center_x + 64]
        return output_image

    def get_training_dataset(self):
        return ImagePosesAndOtherImagesDataset(
            main_image_func=self.get_character_image,
            other_image_funcs=[self.get_face_mask_image],
            pose_dataset=LazyTensorDataset(self.pose_dataset_file_name))

    def get_module_factory(self):
        return SirenFaceMorpher00Factory(
            SirenFaceMorpher00Args(
                image_size=128,
                image_channels=4,
                pose_size=39,
                siren_args=SirenArgs(
                    in_channels=39 + 2,
                    out_channels=4,
                    intermediate_channels=128,
                    num_sine_layers=8)))

    def transform_pose_to_module_input(self, pose: Tensor):
        return pose[:, 0:39]

    def transform_original_image_to_module_input(self, image: Tensor):
        center_x = 256
        center_y = 128 + 16
        return image[:, :, center_y - 64:center_y + 64, center_x - 64:center_x + 64]

    def transform_poser_posed_image_to_groundtruth(self, image: Tensor):
        center_x = 96
        center_y = 96 + 16
        return image[:, :, center_y - 64:center_y + 64, center_x - 64:center_x + 64]

    def get_training_computation_protocol(self):
        return SirenFaceMorpherComputationProtocol00(
            transform_pose_to_module_input_func=self.transform_pose_to_module_input,
            transform_original_image_to_module_input_func=self.transform_original_image_to_module_input,
            transform_poser_posed_image_to_groundtruth_func=self.transform_poser_posed_image_to_groundtruth)

    def get_learning_rate(self, examples_seen_so_far) -> Dict[str, float]:
        if examples_seen_so_far < self.num_training_examples_lr_boundaries[0]:
            return {
                KEY_MODULE: self.base_learning_rate,
            }
        elif examples_seen_so_far < self.num_training_examples_lr_boundaries[1]:
            return {
                KEY_MODULE: self.base_learning_rate / 3.0,
            }
        elif examples_seen_so_far < self.num_training_examples_lr_boundaries[2]:
            return {
                KEY_MODULE: self.base_learning_rate / 10.0,
            }
        else:
            return {
                KEY_MODULE: self.base_learning_rate / 30.0,
            }

    def get_optimizer_factories(self):
        return {
            KEY_MODULE: AdamOptimizerFactory(betas=(0.9, 0.999)),
        }

    def get_poser(self):
        return self.poser_func()

    def get_training_protocol(self, world_size: int):
        total_examples = self.num_training_total_examples
        per_checkpoint_examples = self.num_training_examples_per_checkpoint
        num_checkpoints = total_examples // per_checkpoint_examples
        batch_size = self.total_batch_size // world_size
        return SirenMorpherTrainingProtocol03(
            check_point_examples=[per_checkpoint_examples * (i + 1) for i in range(num_checkpoints)],
            batch_size=batch_size,
            learning_rate=self.get_learning_rate,
            optimizer_factories=self.get_optimizer_factories(),
            random_seed=self.training_random_seed,
            poser_func=self.get_poser,
            key_module=KEY_MODULE,
            key_poser=KEY_POSER)

    def get_sample_output_protocol(self):
        return SirenFaceMorpherSampleOutputProtocol00(
            num_images=8,
            image_size=128,
            images_per_row=2,
            examples_per_sample_output=self.num_training_examples_per_sample_output,
            computation_protocol=self.get_training_computation_protocol(),
            poser_func=self.get_poser,
            random_seed=self.sample_output_random_seed)

    def get_loss(self):
        protocol = self.get_training_computation_protocol()
        return SumLoss([
            (
                'full',
                L1Loss(
                    expected_func=protocol.get_output_func(protocol.keys.groundtruth_posed_image),
                    actual_func=protocol.get_output_func(protocol.keys.predicted_posed_image),
                    weight=1.0)
            ),
            (
                'eye_mouth',
                MaskedL1Loss(
                    expected_func=protocol.get_output_func(protocol.keys.groundtruth_posed_image),
                    actual_func=protocol.get_output_func(protocol.keys.predicted_posed_image),
                    mask_func=protocol.get_output_func(protocol.keys.eye_mouth_mask),
                    weight=20.0)
            ),
        ])

    def create_trainer(self, prefix: str, world_size: int, distrib_backend: str = 'gloo'):
        if self.num_training_examples_per_sample_output is not None:
            sample_output_protocol = self.get_sample_output_protocol()
        else:
            sample_output_protocol = None

        return DistributedTrainer(
            prefix=prefix,
            module_factories={
                KEY_MODULE: self.get_module_factory(),
            },
            accumulators={},
            losses={
                KEY_MODULE: self.get_loss(),
            },
            training_dataset=self.get_training_dataset(),
            validation_dataset=self.get_training_dataset(),
            training_protocol=self.get_training_protocol(world_size),
            validation_protocol=None,
            sample_output_protocol=sample_output_protocol,
            pretrained_module_file_names={},
            example_per_snapshot=self.num_training_examples_per_snapshot,
            num_data_loader_workers=max(1, self.total_worker // world_size),
            distrib_backend=distrib_backend)
