import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Iterator, Tuple


class FlowMatchingBatch:
    def __init__(self, states, obs, obs_mask, forcing, params=None, true_params=None):
        self.states = states
        self.obs = obs
        self.obs_mask = obs_mask
        self.forcing = forcing
        self.params = params
        self.true_params = true_params
        self.batch_size, self.T, self.dim = states.shape

    def to(self, device):
        self.states = self.states.to(device)
        self.obs = self.obs.to(device)
        self.obs_mask = self.obs_mask.to(device)
        self.forcing = self.forcing.to(device)
        if self.params is not None:
            self.params = self.params.to(device)
        if self.true_params is not None:
            self.true_params = self.true_params.to(device)
        return self


class FlowMatchingDataset(Dataset):
    def __init__(self, lorenz_dataset, T_max: float = 5.0, with_params: bool = False,
                 obs_interval: int = 20, R_var: float = 0.5):
        self.source = lorenz_dataset
        self.T_max = T_max
        self.with_params = with_params
        self.obs_interval = obs_interval
        self.R_var = R_var

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        from data.lorenz63 import generate_observations
        w = self.source[idx]
        if "obs" not in w or "obs_mask" not in w:
            obs_seed = w.get("obs_seed", self.obs_interval + idx)
            obs, obs_mask = generate_observations(
                w["true_state"], self.obs_interval, self.R_var, obs_seed)
            w["obs"] = obs
            w["obs_mask"] = obs_mask
        result = (w["true_state"], w["obs"], w["obs_mask"], w["forcing_corrupted"])
        if self.with_params and "sigma" in w:
            result = result + (
                w.get("sigma"), w.get("rho"), w.get("beta"), w.get("c1", 1.0),
            )
            result = result + (
                w.get("true_sigma", w["sigma"]),
                w.get("true_rho", w["rho"]),
                w.get("true_beta", w["beta"]),
                w.get("true_c1", w.get("c1", 1.0)),
            )
        return result


class ConcatFMDataset(Dataset):
    def __init__(self, datasets, with_params: bool = False,
                 obs_interval: int = 20, R_var: float = 0.5):
        self.datasets = datasets
        self.with_params = with_params
        self.obs_interval = obs_interval
        self.R_var = R_var
        self.cumlen = [0]
        for d in datasets:
            self.cumlen.append(self.cumlen[-1] + len(d))

    def __len__(self):
        return self.cumlen[-1]

    def __getitem__(self, idx):
        from data.lorenz63 import generate_observations
        for i in range(len(self.datasets)):
            if idx < self.cumlen[i + 1]:
                w = self.datasets[i][idx - self.cumlen[i]]
                if "obs" not in w or "obs_mask" not in w:
                    obs_seed = w.get("obs_seed", self.obs_interval + idx)
                    obs, obs_mask = generate_observations(
                        w["true_state"], self.obs_interval, self.R_var, obs_seed)
                    w["obs"] = obs
                    w["obs_mask"] = obs_mask
                result = (w["true_state"], w["obs"], w["obs_mask"], w["forcing_corrupted"])
                if self.with_params and "sigma" in w:
                    result = result + (
                        w.get("sigma"), w.get("rho"), w.get("beta"), w.get("c1", 1.0),
                    )
                    result = result + (
                        w.get("true_sigma", w["sigma"]),
                        w.get("true_rho", w["rho"]),
                        w.get("true_beta", w["beta"]),
                        w.get("true_c1", w.get("c1", 1.0)),
                    )
                return result
        raise IndexError


def collate_fm(batch):
    states = torch.stack([b[0] for b in batch])
    obs = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    forcing = torch.stack([b[3] for b in batch])
    params = None
    true_params = None
    if len(batch[0]) == 12:
        params = torch.stack([torch.tensor([b[4], b[5], b[6], b[7]], dtype=torch.float32) for b in batch])
        true_params = torch.stack([torch.tensor([b[8], b[9], b[10], b[11]], dtype=torch.float32) for b in batch])
    return FlowMatchingBatch(states, obs, masks, forcing, params=params, true_params=true_params)


def make_dataloaders(datasets: Dict[str, Dataset], batch_size: int = 32,
                     obs_interval: int = 20, R_var: float = 0.5):
    return {
        "train": DataLoader(
            ConcatFMDataset([datasets["train_cs1"], datasets["train_cs2"]],
                            obs_interval=obs_interval, R_var=R_var),
            batch_size=batch_size, shuffle=True, collate_fn=collate_fm,
        ),
        "val": DataLoader(
            ConcatFMDataset([datasets["val_cs1"], datasets["val_cs2"]],
                            obs_interval=obs_interval, R_var=R_var),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
        "test_cs1": DataLoader(
            FlowMatchingDataset(datasets["test_cs1"],
                                obs_interval=obs_interval, R_var=R_var),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
        "test_cs2": DataLoader(
            FlowMatchingDataset(datasets["test_cs2"],
                                obs_interval=obs_interval, R_var=R_var),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
    }
