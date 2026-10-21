"""normalize -> model -> denormalize round trip in train._make_eval_batch / evaluate_model."""
import numpy as np
import torch

from data.normalization import normalize
from train import _make_eval_batch, evaluate_model

STATS = {"mean": torch.tensor([1.0, -2.0, 25.0]), "std": torch.tensor([8.0, 9.0, 8.0])}


class _Window(dict):
    pass


class _Dataset:
    def __init__(self, n=4, T=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.windows = []
        for _ in range(n):
            state = torch.randn(T, 3, generator=g) * STATS["std"] + STATS["mean"]
            self.windows.append(_Window(true_state=state, obs=state.clone(),
                                        obs_mask=torch.ones(T, 3),
                                        forcing_corrupted=torch.zeros(T, 3)))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i]


class _EchoObs(torch.nn.Module):
    def forward(self, batch):
        return batch.obs


def test_make_eval_batch_normalizes_states_and_obs():
    w = _Dataset().windows[0]
    batch = _make_eval_batch(w, "cpu", param_dim=0, norm_stats=STATS)
    expected = normalize(w["true_state"].unsqueeze(0), STATS)
    torch.testing.assert_close(batch.states, expected)
    torch.testing.assert_close(batch.obs, expected)
    assert abs(batch.states.mean().item()) < 3.0 and batch.states.std().item() < 3.0


def test_make_eval_batch_without_stats_is_identity():
    w = _Dataset().windows[0]
    batch = _make_eval_batch(w, "cpu", param_dim=0)
    torch.testing.assert_close(batch.obs, w["obs"].unsqueeze(0))


def test_evaluate_model_round_trip_recovers_physical_units():
    mean, std = evaluate_model(_EchoObs(), _Dataset(), "cpu", "direct_unet", param_dim=0,
                               norm_stats=STATS)
    np.testing.assert_allclose(mean, 0.0, atol=1e-5)
    np.testing.assert_allclose(std, 0.0, atol=1e-5)

