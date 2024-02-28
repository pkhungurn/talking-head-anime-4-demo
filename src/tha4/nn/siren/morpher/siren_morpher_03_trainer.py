from enum import Enum
from typing import Dict, List, Optional, Callable

import torch
from tha4.shion.base.dataset.lazy_tensor_dataset import LazyTensorDataset
from tha4.shion.base.image_util import extract_pytorch_image_from_filelike
from tha4.shion.base.loss.l1_loss import L1Loss
from tha4.shion.base.loss.sum_loss import SumLoss
from tha4.shion.base.loss.time_dependently_weighted_loss import TimeDependentlyWeightedLoss
from tha4.shion.base.optimizer_factories import AdamOptimizerFactory
from tha4.shion.core.training.distrib.distributed_trainer import DistributedTrainer
from tha4.dataset.image_poses_and_aother_images_dataset import ImagePosesAndOtherImagesDataset
from tha4.nn.siren.morpher.siren_morpher_03 import SirenMorpherLevelArgs, SirenMorpher03Factory, SirenMorpher03Args
from tha4.nn.siren.morpher.siren_morpher_protocols_03 import SirenMorpherComputationProtocol03, \
    SirenMorpherProtocol03Indices, KEY_MODULE, KEY_POSER, KEY_EXAMPLES_SEEN_SO_FAR, SirenMorpherTrainingProtocol03, \
    SirenMorpherSampleOutputProtocol
from tha4.poser.poser import Poser


def get_poser():
    import tha4.poser.modes.mode_07
    poser = tha4.poser.modes.mode_07.create_poser(torch.device('cpu'))
    return poser


class LossTerm(Enum):
    full_blended = 1
    full_warped = 2
    full_grid_change = 3
    full_color_change = 4

    def get_loss(self, protocol: SirenMorpherComputationProtocol03):
        if self == LossTerm.full_blended:
            return L1Loss(
                expected_func=protocol.get_output_func(protocol.keys.groundtruth_posed_image),
                actual_func=protocol.get_output_func(protocol.keys.predicted_posed_image))
        elif self == LossTerm.full_warped:
            return L1Loss(
                expected_func=protocol.get_output_func(protocol.keys.groundtruth_warped_image),
                actual_func=protocol.get_output_func(protocol.keys.predicted_warped_image))
        elif self == LossTerm.full_grid_change:
            return L1Loss(
                expected_func=protocol.get_output_func(protocol.keys.groundtruth_grid_change),
                actual_func=protocol.get_output_func(protocol.keys.predicted_grid_change))
        elif self == LossTerm.full_color_change:
            return L1Loss(
                expected_func=protocol.get_output_func(protocol.keys.groundtruth_posed_image),
                actual_func=protocol.get_output_func(protocol.keys.predicted_color_change))
        else:
            raise RuntimeError(f"Unsupported loss term {self}")


class LossWeights:
    def __init__(self, weights: Optional[Dict[LossTerm, float]] = None):
        self.weights = {}
        for term in LossTerm:
            self.weights[term] = 0.0
        if weights is not None:
            for term in LossTerm:
                if term in weights:
                    self.weights[term] = weights[term]


class TrainingPhase:
    def __init__(self,
                 num_examples_upper_bound: int,
                 learning_rate: float,
                 loss_weights: LossWeights):
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.num_examples_upper_bound = num_examples_upper_bound


class LearningRateFunc:
    def __init__(self, phases: List[TrainingPhase], keys: List[str]):
        self.phases = phases
        self.keys = keys

    def make_learning_rate_dict(self, keys: List[str], value: float):
        output = {}
        for key in keys:
            output[key] = value
        return output

    def __call__(self, examples_seen_so_far: int) -> Dict[str, float]:
        for i in range(len(self.phases) - 1):
            if examples_seen_so_far < self.phases[i].num_examples_upper_bound:
                return self.make_learning_rate_dict(self.keys, self.phases[i].learning_rate)
        return self.make_learning_rate_dict(self.keys, self.phases[-1].learning_rate)


class LossWeightFunc:
    def __init__(self, phases: List[TrainingPhase], term: LossTerm):
        self.term = term
        self.phases = phases

    def __call__(self, examples_seen_so_far: int) -> float:
        for i in range(len(self.phases) - 1):
            if examples_seen_so_far < self.phases[i].num_examples_upper_bound:
                return self.phases[i].loss_weights.weights[self.term]
        return self.phases[-1].loss_weights.weights[self.term]


class TrainingPhases:
    def __init__(self, phases: List[TrainingPhase]):
        assert len(phases) > 0
        for i in range(1, len(phases)):
            assert phases[i - 1].num_examples_upper_bound < phases[i].num_examples_upper_bound

        self.phases = phases

    def make_learning_rate_dict(self, keys: List[str], value: float):
        output = {}
        for key in keys:
            output[key] = value
        return output

    def get_learning_rate_func(self, keys: List[str]):
        return LearningRateFunc(self.phases, keys)

    def get_loss_weight_func(self, term: LossTerm) -> Callable[[int], float]:
        return LossWeightFunc(self.phases, term)


class SirenMorpher03TrainerArgs:
    def __init__(self,
                 character_file_name: str,
                 pose_dataset_file_name: str,
                 training_phases: TrainingPhases,
                 num_training_examples_per_checkpoint: int = 100_000,
                 num_training_examples_per_sample_output: Optional[int] = 10_000,
                 num_training_examples_per_snapshot: int = 10_000,
                 total_batch_size: int = 8,
                 training_random_seed: int = 2965603729,
                 sample_output_random_seed: int = 3522651501,
                 total_worker: int = 8,
                 poser_func: Optional[Callable[[], Poser]] = None,
                 sample_output_batch_size: Optional[int] = None,
                 pretrained_module_file_name: Optional[str] = None):
        for phase in training_phases.phases:
            assert phase.num_examples_upper_bound % num_training_examples_per_checkpoint == 0

        if poser_func is None:
            poser_func = get_poser

        self.training_phases = training_phases
        self.pretrained_module_file_name = pretrained_module_file_name
        self.sample_output_batch_size = sample_output_batch_size
        self.poser_func = poser_func
        self.total_worker = total_worker
        self.num_training_examples_per_snapshot = num_training_examples_per_snapshot
        self.num_training_examples_per_sample_output = num_training_examples_per_sample_output
        self.sample_output_random_seed = sample_output_random_seed
        self.training_random_seed = training_random_seed
        self.total_batch_size = total_batch_size
        self.num_training_examples_per_checkpoint = num_training_examples_per_checkpoint
        self.pose_dataset_file_name = pose_dataset_file_name
        self.character_file_name = character_file_name

    def get_character_image(self):
        return extract_pytorch_image_from_filelike(
            self.character_file_name,
            scale=2.0,
            offset=-1.0,
            premultiply_alpha=True,
            perform_srgb_to_linear=True)

    def get_training_dataset(self):
        return ImagePosesAndOtherImagesDataset(
            main_image_func=self.get_character_image,
            pose_dataset=LazyTensorDataset(self.pose_dataset_file_name),
            other_image_funcs=[])

    def get_module_factory(self):
        return SirenMorpher03Factory(
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

    def get_training_computation_protocol(self):
        return SirenMorpherComputationProtocol03(
            indices=SirenMorpherProtocol03Indices(
                batch_image=0,
                batch_pose=1,
                batch_face_mask=2))

    def get_optimizer_factories(self):
        return {
            KEY_MODULE: AdamOptimizerFactory(betas=(0.9, 0.999)),
        }

    def get_poser(self):
        return self.poser_func()

    def get_training_protocol(self, world_size: int):
        total_examples = self.training_phases.phases[-1].num_examples_upper_bound
        per_checkpoint_examples = self.num_training_examples_per_checkpoint
        num_checkpoints = total_examples // per_checkpoint_examples
        batch_size = self.total_batch_size // world_size
        return SirenMorpherTrainingProtocol03(
            check_point_examples=[per_checkpoint_examples * (i + 1) for i in range(num_checkpoints)],
            batch_size=batch_size,
            learning_rate=self.training_phases.get_learning_rate_func([KEY_MODULE]),
            optimizer_factories=self.get_optimizer_factories(),
            random_seed=self.training_random_seed,
            poser_func=self.get_poser,
            key_module=KEY_MODULE,
            key_poser=KEY_POSER)

    def get_sample_output_protocol(self):
        return SirenMorpherSampleOutputProtocol(
            num_images=4,
            image_size=512,
            examples_per_sample_output=self.num_training_examples_per_sample_output,
            computation_protocol=self.get_training_computation_protocol(),
            poser_func=self.get_poser,
            random_seed=self.sample_output_random_seed,
            batch_size=self.sample_output_batch_size,
            batch_pose_index=1,
            batch_image_index=0)

    def get_loss(self):
        protocol = self.get_training_computation_protocol()
        losses = []
        for term in LossTerm:
            base_loss = term.get_loss(protocol)
            loss = TimeDependentlyWeightedLoss(
                base_loss,
                examples_seen_so_far_func=lambda state: state.outputs[KEY_EXAMPLES_SEEN_SO_FAR],
                weight_func=self.training_phases.get_loss_weight_func(term))
            losses.append((term.name, loss))
        return SumLoss(losses)

    def create_trainer(self, prefix: str, world_size: int, distrib_backend: str = 'gloo'):
        if self.num_training_examples_per_sample_output is not None:
            sample_output_protocol = self.get_sample_output_protocol()
        else:
            sample_output_protocol = None

        pretrained_module_file_names = {}
        if self.pretrained_module_file_name is not None:
            pretrained_module_file_names[KEY_MODULE] = self.pretrained_module_file_name

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
            pretrained_module_file_names=pretrained_module_file_names,
            example_per_snapshot=self.num_training_examples_per_snapshot,
            num_data_loader_workers=max(1, self.total_worker // world_size),
            distrib_backend=distrib_backend)
