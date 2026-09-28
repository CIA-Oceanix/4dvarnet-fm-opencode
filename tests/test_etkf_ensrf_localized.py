import torch

from evaluation.baselines import ETKF, ObsOperator, _ensrf_localized_analysis


def _problem(N: int = 60, D: int = 12, od: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ens = torch.randn(N, D, generator=g, dtype=torch.float64) @ torch.randn(D, D, generator=g, dtype=torch.float64)
    idx = torch.arange(od) * 2
    mu = ens.mean(0)
    A = ens - mu
    HA = A[:, idx]
    y = mu[idx] + torch.randn(od, generator=g, dtype=torch.float64)
    return mu, A, HA, y - mu[idx], idx


def test_unlocalized_ensrf_is_the_exact_kalman_update():
    mu, A, HA, dy, idx = _problem()
    N1 = A.shape[0] - 1
    r = 0.7
    R = torch.eye(len(idx), dtype=torch.float64) * r
    ones_x = torch.ones(A.shape[1], len(idx), dtype=torch.float64)
    ones_y = torch.ones(len(idx), len(idx), dtype=torch.float64)
    ens_a = _ensrf_localized_analysis(mu, A, HA, dy, ones_x, ones_y, R, 0.0, N1)
    P = A.T @ A / N1
    H = torch.zeros(len(idx), A.shape[1], dtype=torch.float64)
    H[torch.arange(len(idx)), idx] = 1.0
    K = P @ H.T @ torch.linalg.inv(H @ P @ H.T + R)
    torch.testing.assert_close(ens_a.mean(0), mu + K @ dy, rtol=1e-9, atol=1e-9)
    Aa = ens_a - ens_a.mean(0)
    torch.testing.assert_close(Aa.T @ Aa / N1, (torch.eye(A.shape[1], dtype=torch.float64) - K @ H) @ P,
                               rtol=1e-8, atol=1e-8)


def test_legacy_square_root_mode_over_contracts_the_spread():
    mu, A, HA, dy, idx = _problem(seed=1)
    N1 = A.shape[0] - 1
    R = torch.eye(len(idx), dtype=torch.float64) * 0.7
    ones_x = torch.ones(A.shape[1], len(idx), dtype=torch.float64)
    ones_y = torch.ones(len(idx), len(idx), dtype=torch.float64)
    good = _ensrf_localized_analysis(mu, A, HA, dy, ones_x, ones_y, R, 0.0, N1)
    K_legacy = torch.linalg.solve(HA.T @ HA + R, (A.T @ HA).T).T
    legacy = (mu + K_legacy @ dy).unsqueeze(0) + A - HA @ K_legacy.T
    assert float(legacy.var(0)[idx].sum()) < 0.01 * float(good.var(0)[idx].sum())


def test_etkf_ensrf_mode_runs_and_keeps_more_spread_than_square_root():
    torch.manual_seed(0)
    D, N, T = 16, 40, 30

    class Linear:
        state_dim = D
        param_dim = 0
        param_names = []
        forcing_dim = 0

        def step(self, x, W=None, **kw):
            return 0.95 * torch.roll(x, 1, dims=-1) + 0.05 * x

    idx = list(range(0, D, 2))
    op = ObsOperator(D, obs_indices=idx)
    truth = torch.randn(T, D)
    obs = truth[:, idx] + 0.3 * torch.randn(T, len(idx))
    mask = torch.ones(T, dtype=torch.bool)
    Lx = torch.ones(D, len(idx))
    Ly = torch.ones(len(idx), len(idx))
    out = {}
    for mode in ("square_root", "ensrf"):
        f = ETKF(N_ensemble=N, R_var=0.09, inflation=1.0, dynamics=Linear(), obs_operator=op,
                 loc_mode=mode, loc_Lx_t=[Lx] * T, loc_Ly_t=[Ly] * T, noise_init_std=1.0)
        f.init_ensemble = torch.randn(N, D)
        res = f.assimilate(obs, mask, torch.zeros(T, 0), true_state=truth)
        out[mode] = float(res.ensemble_variance[1:].mean())
    assert out["ensrf"] > 1.5 * out["square_root"]
