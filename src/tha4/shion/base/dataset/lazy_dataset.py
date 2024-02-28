from typing import Callable

from torch.utils.data import Dataset


class LazyDataset(Dataset):
    def __init__(self, source_func: Callable[[], Dataset]):
        self.source_func = source_func
        self.source = None

    def get_source(self):
        if self.source is None:
            self.source = self.source_func()
        return self.source

    def __len__(self):
        return len(self.get_source())

    def __getitem__(self, item):
        return self.get_source()[item]
