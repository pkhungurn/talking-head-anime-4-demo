from typing import Any, Callable

from torch.utils.data import Dataset


class XformedDataset(Dataset):
    def __init__(self, source: Dataset, xform_func: Callable[[Any], Any]):
        self.xform_func = xform_func
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        return self.xform_func(self.source[item])
