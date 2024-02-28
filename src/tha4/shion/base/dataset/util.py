from typing import List

import torch
from torch.utils.data import Dataset


def get_indexed_batch(dataset: Dataset, example_indices: List[int], device: torch.device):
    if len(example_indices) == 0:
        return []
    examples = []
    for index in range(len(example_indices)):
        example_index = example_indices[index]
        raw_example = dataset[example_index]
        example = []
        for x in raw_example:
            if isinstance(x, torch.Tensor):
                y = x.to(device).unsqueeze(0)
            elif isinstance(x, float) or isinstance(x, int):
                y = torch.tensor([[x]], device=device)
            else:
                raise RuntimeError(f"get_indexed_batch: Data of type {type(x)} is not supported.")
            example.append(y)
        examples.append(example)
    k = len(examples[0])
    transposed = [[] for i in range(k)]
    for example in examples:
        for i in range(k):
            transposed[i].append(example[i])
    return [torch.cat(x, dim=0) for x in transposed]
