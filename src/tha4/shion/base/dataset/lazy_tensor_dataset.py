import torch
from torch.utils.data import Dataset, TensorDataset

from tha4.shion.core.load_save import torch_load


class LazyTensorDataset(Dataset):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.dataset = None

    def get_dataset(self):
        if self.dataset is None:
            data = torch_load(self.file_name)
            if isinstance(data, torch.Tensor):
                self.dataset = TensorDataset(data)
            elif isinstance(data, tuple):
                self.dataset = TensorDataset(*data)
            elif isinstance(data, list):
                self.dataset = TensorDataset(*data)
            else:
                raise RuntimeError("Unsupported data type: " + type(data))
        return self.dataset

    def __len__(self):
        dataset = self.get_dataset()
        return len(dataset)

    def __getitem__(self, item):
        dataset = self.get_dataset()
        return dataset.__getitem__(item)


