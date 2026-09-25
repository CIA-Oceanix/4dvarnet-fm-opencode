"""Analysis-ensemble CRPS in the batched ETKF/EnKF."""
import numpy as np
import torch

from evaluation.baselines import ETKF, EnKF, ObsOperator, _analysis_crps_step
from evaluation.estimate_metrics import ensemble_window_scores
from evaluation.run_l96 import make_obs_j_indices
from models.lorenz96_dynamics import Lorenz96Dynamics

IDX = make_obs_j_indices(8, 4, 2)


def test_crps_step_matches_flow_formula():
    g = torch.Generator().manual_seed(0)
    ens = torch.randn(3, 9, 5, generator=g, dtype=torch.float64)
    ref = torch.randn(3, 5, generator=g, dtype=torch.float64)
    got = _analysis_crps_step(ens, ref).numpy()
    members = ens.permute(0, 2, 1).numpy()[:, None]
    want = ensemble_window_scores(members, ref.numpy()[:, None])["crps"]
    np.testing.assert_allclose(got, want, rtol=1e-10)


def _run(cls):
    torch.manual_seed(0)
    T, B = 30, 2
    da = cls(N_ensemble=8, dt=0.001, inflation=1.5, dynamics=Lorenz96Dynamics(dt=0.001),
             obs_operator=ObsOperator(40, IDX), NO=8, J=4)
    truth = torch.randn(B, T, 40)
    obs = torch.full((B, T, 24), float("nan"))
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, [0, 10, 20]] = True
    obs[:, [0, 10, 20]] = truth[:, [0, 10, 20]][..., list(IDX)]
    kw = {k: torch.full((B,), v) for k, v in dict(F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1).items()}
    return da.assimilate_batch(obs, mask, torch.zeros(B, T), truth, **kw), truth


def test_batched_filters_return_analysis_crps():
    for cls in (ETKF, EnKF):
        res, _ = _run(cls)
        for r in res:
            assert r.crps is not None and r.crps.shape == (40,)
            assert np.all(np.isfinite(r.crps)) and np.all(r.crps >= -1e-6)
