import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict


def _l96_biased_param_vector(w):
    """Extract the biased (``*_da``) 8-param vector from an L96 window.

    The randomize/s0_s1 layout stores scalar biased scalar params as
    ``F_da, c1_da, hx_da, eps_da`` and the biased fast weights as the list
    ``fast_weights_da`` (there is no per-index ``w{j}_da``). Falls back to the
    un-biased key when a ``*_da`` entry is absent (e.g. test_s0 windows which
    carry no bias). Mirrors ``_window_param_vector`` in
    ``evaluation/neural_inference.py`` for the true/plain layout.
    """
    vec = [float(w.get(f"{n}_da", w.get(n, 1.0 if n == "c1" else 0.0)))
           for n in ("F", "c1", "hx", "eps")]
    fw = w.get("fast_weights_da", w.get("fast_weights"))
    if fw is None:
        fw = w.get("true_fast_weights", [1.0, 1.0, 0.1, 0.1])
    fw = list(fw)
    if len(fw) < 4:
        fw = fw + [0.0] * (4 - len(fw))
    return tuple(vec + [float(x) for x in fw])


def _l96_true_param_vector(w):
    """Extract the true 8-param vector from an L96 window, list-format aware.

    Mirrors ``_window_param_vector(bd, prefix="true_")`` in
    ``evaluation/neural_inference.py``: read scalar ``true_F..true_eps`` keys
    and the fast weights as scalar ``true_w1..true_w4`` when flattened, falling
    back to splitting the ``true_fast_weights`` list for older cached windows
    (which store only the list form). Used by the eval path so fast-weight RMSE
    is measured against the correct per-window truth instead of a silent 0.0.
    """
    vec = [float(w.get(f"true_{n}", w.get(n, 1.0 if n == "c1" else 0.0)))
           for n in ("F", "c1", "hx", "eps")]
    scalar_keys = [f"true_w{j}" for j in range(1, 5)]
    if all(k in w for k in scalar_keys):
        vec += [float(w[k]) for k in scalar_keys]
    else:
        fw = w.get("true_fast_weights")
        if fw is None:
            fw = w.get("fast_weights", [1.0, 1.0, 0.1, 0.1])
        fw = list(fw)
        if len(fw) < 4:
            fw = fw + [0.0] * (4 - len(fw))
        vec += [float(x) for x in fw]
    return tuple(vec)


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
                 obs_interval: int = 20, R_var: float = 0.5, param_names=None,
                 obs_var_indices=None, use_biased_params: bool = False,
                 resample_bias_draws: bool = False, bias_max: float = 0.2,
                 noisy_da_bias: bool = False, noisy_da_max: float = 1.5):
        self.source = lorenz_dataset
        self.T_max = T_max
        self.with_params = with_params
        self.obs_interval = obs_interval
        self.R_var = R_var
        self.param_names = param_names or ["sigma", "rho", "beta", "c1"]
        self.param_dim = len(self.param_names)
        self.obs_var_indices = obs_var_indices
        self.use_biased_params = use_biased_params
        self.resample_bias_draws = resample_bias_draws
        self.bias_max = bias_max
        self.noisy_da_bias = noisy_da_bias
        self.noisy_da_max = noisy_da_max

    def __len__(self):
        return len(self.source)

    def _extract_params(self, w):
        if self.noisy_da_bias:
            # Fresh per-access sample of a random fraction (0 to noisy_da_max,
            # e.g. up to 1.5x) of the window's own true->DA bias, rather than
            # always the full (fraction=1.0) DA bias or a bias-agnostic jitter.
            # S0 windows carry no *_da entries, so _l96_biased_param_vector
            # falls back to the true value there (bias vector = 0, so this is
            # a no-op on S0 regardless of the sampled fraction).
            true_vec = _l96_true_param_vector(w)
            da_vec = _l96_biased_param_vector(w)
            frac = torch.empty(len(true_vec)).uniform_(0.0, self.noisy_da_max)
            return tuple(t + float(f) * (d - t) for t, d, f in zip(true_vec, da_vec, frac.tolist()))
        if self.resample_bias_draws:
            true = tuple(w.get(f"true_{n}", w.get(n, 1.0 if n == "c1" else 0.0))
                         for n in self.param_names)
            draw = 1.0 + torch.empty(len(true)).uniform_(0.0, self.bias_max)
            return tuple(t * float(d) for t, d in zip(true, draw.tolist()))
        if not self.use_biased_params:
            return tuple(w.get(n, 1.0 if n == "c1" else 0.0) for n in self.param_names)
        return _l96_biased_param_vector(w)

    def _extract_true_params(self, w):
        if self.param_names == ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]:
            return _l96_true_param_vector(w)
        return tuple(w.get(f"true_{n}", w.get(n, 1.0 if n == "c1" else 0.0)) for n in self.param_names)

    def __getitem__(self, idx):
        from data.lorenz63 import generate_observations
        w = self.source[idx]
        if "obs" not in w or "obs_mask" not in w:
            obs_seed = w.get("obs_seed", self.obs_interval + idx)
            obs, obs_mask = generate_observations(
                w["true_state"], self.obs_interval, self.R_var, obs_seed)
            w["obs"] = obs
            w["obs_mask"] = obs_mask
        true_state = w["true_state"]
        if self.obs_var_indices is not None:
            true_state = true_state[:, self.obs_var_indices]
        result = (true_state, w["obs"], w["obs_mask"], w["forcing_corrupted"])
        if self.with_params and self.param_names[0] in w:
            result = result + self._extract_params(w)
            result = result + self._extract_true_params(w)
        return result


class ConcatFMDataset(Dataset):
    def __init__(self, datasets, with_params: bool = False,
                 obs_interval: int = 20, R_var: float = 0.5, param_names=None,
                 obs_var_indices=None):
        self.datasets = datasets
        self.with_params = with_params
        self.obs_interval = obs_interval
        self.R_var = R_var
        self.param_names = param_names or ["sigma", "rho", "beta", "c1"]
        self.param_dim = len(self.param_names)
        self.obs_var_indices = obs_var_indices
        self.cumlen = [0]
        for d in datasets:
            self.cumlen.append(self.cumlen[-1] + len(d))

    def __len__(self):
        return self.cumlen[-1]

    def _extract_params(self, w):
        return tuple(w.get(n, 1.0 if n == "c1" else 0.0) for n in self.param_names)

    def _extract_true_params(self, w):
        if self.param_names == ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]:
            return _l96_true_param_vector(w)
        return tuple(w.get(f"true_{n}", w.get(n, 1.0 if n == "c1" else 0.0)) for n in self.param_names)

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
                true_state = w["true_state"]
                if self.obs_var_indices is not None:
                    true_state = true_state[:, self.obs_var_indices]
                result = (true_state, w["obs"], w["obs_mask"], w["forcing_corrupted"])
                if self.with_params and self.param_names[0] in w:
                    result = result + self._extract_params(w)
                    result = result + self._extract_true_params(w)
                return result
        raise IndexError


def collate_fm(batch):
    states = torch.stack([b[0] for b in batch])
    obs = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    forcing = torch.stack([b[3] for b in batch])
    params = None
    true_params = None
    if len(batch[0]) > 4:
        n_params = (len(batch[0]) - 4) // 2
        params = torch.stack([torch.tensor(b[4:4 + n_params], dtype=torch.float32) for b in batch])
        true_params = torch.stack([torch.tensor(b[4 + n_params:4 + 2 * n_params], dtype=torch.float32) for b in batch])
    return FlowMatchingBatch(states, obs, masks, forcing, params=params, true_params=true_params)


def make_collate_fm(norm_stats: dict | None = None, obs_density_cfg: dict | None = None):
    """Return a ``collate_fm``-compatible collate fn that additionally
    z-score normalizes ``states``/``obs`` when ``norm_stats`` is given, and/or
    applies TRAINING-time fast-Y observation-density augmentation when
    ``obs_density_cfg`` is given.

    ``obs_density_cfg`` (optional dict, keys ``full_prob`` default 0.4 and
    ``min_keep`` default 0) draws a fresh
    ``data.obs_density.sample_training_density_mask`` every batch and NaNs
    out the dropped fast-Y channels of ``obs`` -- so the model sees
    partial-channel-dropout obs patterns during training instead of only at
    eval time (see ``eval_obs_density_l96.py`` / ``data/obs_density.py``'s
    module docstring for why that gap matters: DirectUNet/CFM read obs only
    via ``nan_to_num``, with no separate mask channel, so an untrained-for
    dropped channel is indistinguishable from a real near-zero observation).
    Requires the canonical 24D (8 slow-X + 16 fast-Y) obsj2 observed
    subspace -- raises if ``obs``'s last dim doesn't match.

    ``norm_stats is None and obs_density_cfg is None`` reproduces plain
    ``collate_fm`` exactly.
    """
    if norm_stats is None and obs_density_cfg is None:
        return collate_fm

    from data.normalization import normalize

    def _collate(batch):
        fm_batch = collate_fm(batch)
        if obs_density_cfg is not None:
            from data.obs_density import (
                NUM_FAST,
                NUM_SLOW,
                apply_density_mask_to_obs,
                sample_training_density_mask,
            )
            B, T, D = fm_batch.obs.shape
            if D != NUM_SLOW + NUM_FAST:
                raise ValueError(
                    f"obs_density_cfg requires the canonical {NUM_SLOW}+{NUM_FAST}D "
                    f"obsj2 observed subspace, got obs dim {D}"
                )
            keep_mask = sample_training_density_mask(
                B, T,
                full_prob=obs_density_cfg.get("full_prob", 0.4),
                min_keep=obs_density_cfg.get("min_keep", 0),
                device=fm_batch.obs.device,
            )
            fm_batch.obs = apply_density_mask_to_obs(fm_batch.obs, keep_mask)
        if norm_stats is not None:
            fm_batch.states = normalize(fm_batch.states, norm_stats)
            fm_batch.obs = normalize(fm_batch.obs, norm_stats)
        return fm_batch

    return _collate


def make_dataloaders(datasets: Dict[str, Dataset], batch_size: int = 32,
                     obs_interval: int = 20, R_var: float = 0.5,
                     obs_var_indices=None):
    return {
        "train": DataLoader(
            ConcatFMDataset([datasets["train_cs1"], datasets["train_cs2"]],
                            obs_interval=obs_interval, R_var=R_var,
                            obs_var_indices=obs_var_indices),
            batch_size=batch_size, shuffle=True, collate_fn=collate_fm,
        ),
        "val": DataLoader(
            ConcatFMDataset([datasets["val_cs1"], datasets["val_cs2"]],
                            obs_interval=obs_interval, R_var=R_var,
                            obs_var_indices=obs_var_indices),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
        "test_cs1": DataLoader(
            FlowMatchingDataset(datasets["test_cs1"],
                                obs_interval=obs_interval, R_var=R_var,
                                obs_var_indices=obs_var_indices),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
        "test_cs2": DataLoader(
            FlowMatchingDataset(datasets["test_cs2"],
                                obs_interval=obs_interval, R_var=R_var,
                                obs_var_indices=obs_var_indices),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fm,
        ),
    }
