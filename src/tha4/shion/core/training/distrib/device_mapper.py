from typing import Dict

import torch


class SimpleCudaDeviceMapper:
    def __call__(self, rank, local_rank):
        return torch.device("cuda", local_rank)


class UserSpecifiedLocalRankToDeviceMapper:
    def __init__(self, device_map: Dict[int, torch.device]):
        self.device_map = device_map

    def __call__(self, rank, local_rank):
        assert local_rank in self.device_map
        return self.device_map[local_rank]
