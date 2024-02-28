from typing import List, Callable

from torch import Tensor
from torch.utils.data import Dataset


class ImagePosesAndOtherImagesDataset(Dataset):
    def __init__(self,
                 main_image_func: Callable[[], Tensor],
                 pose_dataset: Dataset,
                 other_image_funcs: List[Callable[[], Tensor]]):
        self.main_image_func = main_image_func
        self.other_image_funcs = other_image_funcs
        self.pose_dataset = pose_dataset
        self.main_image = None
        self.other_images = [None for i in range(len(self.other_image_funcs))]

    def get_main_image(self):
        if self.main_image is None:
            self.main_image = self.main_image_func()
        return self.main_image

    def get_other_image(self, image_index: int):
        if self.other_images[image_index] is None:
            self.other_images[image_index] = self.other_image_funcs[image_index]()
        return self.other_images[image_index]

    def __len__(self):
        return len(self.pose_dataset)

    def __getitem__(self, index):
        main_image = self.get_main_image()
        pose = self.pose_dataset[index][0]
        other_images = [self.get_other_image(i) for i in range(len(self.other_image_funcs))]
        return [main_image, pose] + other_images
