import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Iterator, Tuple


class FlowMatchingBatch:
    def __init__(self, states: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor):
        self.states = states
        self.obs = obs
        self.obs_mask = obs_mask
        self.batch_size, self.T, self.dim = states.shape

    def to(self, device: torch.device):
        self.states = self.states.to(device)
        self.obs = self.obs.to(device)
        self.obs_mask = self.obs_mask.to(device)
        return self


class FlowMatchingDataset(Dataset):
    def __init__(self, lorenz_dataset, T_max: float = 5.0):
        self.source = lorenz_dataset
        self.T_max = T_max

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        w = self.source[idx]
        return w["true_state"], w["obs"], w["obs_mask"]


def collate_fm(batch):
    states = torch.stack([b[0] for b in batch])
    obs = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    return FlowMatchingBatch(states, obs, masks)


def make_dataloaders(datasets: Dict[str, Dataset], batch_size: int = 32):
    return {
        "train": DataLoader(
            FlowMatchingDataset(datasets["train"]),
            batch_size=batch_size, shuffle=True, collate_fn=collate_fm,
        ),
        "val": DataLoader(
            FlowMatchingDataset(datasets["val"]),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
        "test_cs1": DataLoader(
            FlowMatchingDataset(datasets["test_cs1"]),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
        "test_cs2": DataLoader(
            FlowMatchingDataset(datasets["test_cs2"]),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
    }
