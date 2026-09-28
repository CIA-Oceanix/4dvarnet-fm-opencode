import dataclasses
import os
import shutil

import pytest
import torch

from data.qg_datasets import QGDatasetSpec, assemble_split, split_dir, write_shard
from evaluation.qg_specwind_s1 import (
    MODE_RMS,
    REALISTIC,
    REFERENCE,
    S1Levels,
    corrupt_amplitudes,
    corrupt_obs,
    obs_error_frac,
)
from evaluation.qg_specwind_s1_sweep import realistic_tasks, shapley, task_name, tasks
from evaluation.run_qg_specwind_da import apply_s1, build_cfg, da_cfg, evaluate, s0_windows
from models.qg_wind_modes import FourierWindBasis

TINY = QGDatasetSpec(name="tiny_s1", n_train=3, n_val=3, n_test=3, nx=16, spinup_days=1.0,
                     lead_days=0.5, window_days=1.0, burnin_units=2.0, keep_every_trainval=6)
WAVEVECTORS = FourierWindBasis(nx=16, kmax=2).wavevectors


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("s1"))
    write_shard(TINY, "test", root, 0, 1, batch_size=3)
    assemble_split(TINY, "test", root)
    yield root
    d = split_dir(root, TINY, "test")
    for f in os.listdir(d):
        os.chmod(os.path.join(d, f), 0o644)
    shutil.rmtree(root, ignore_errors=True)


def _amps(n: int = 50) -> torch.Tensor:
    return torch.randn(n, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 2e-12


def test_zero_levels_leave_the_wind_unchanged():
    a = _amps()
    assert torch.equal(corrupt_amplitudes(a, WAVEVECTORS, S1Levels(), 7200.0, key=3), a)
    assert REFERENCE.scaled(0.0) == S1Levels(tau_days=REFERENCE.tau_days)


def test_components_use_independent_draws():
    a = _amps()
    lv = REFERENCE.scaled(2.0)
    noise = corrupt_amplitudes(a, WAVEVECTORS, lv.only(["noise"]), 7200.0, key=5) - a
    both = corrupt_amplitudes(a, WAVEVECTORS, lv.only(["amp", "noise"]), 7200.0, key=5)
    torch.testing.assert_close(both - (1 + lv.amp_bias) * a, noise)
    other = corrupt_amplitudes(a, WAVEVECTORS, lv.only(["noise"]), 7200.0, key=6) - a
    assert not torch.allclose(noise, other, rtol=1e-3, atol=0.0)
    rel = noise.std(0) / torch.tensor(MODE_RMS, dtype=torch.float64)
    assert float(rel.mean()) == pytest.approx(lv.noise_frac, rel=0.8)


def test_shift_is_a_translation_of_the_pattern():
    a = _amps(20)
    shifted = corrupt_amplitudes(a, WAVEVECTORS, S1Levels(shift_frac=0.1), 7200.0, key=1)
    mag = lambda x: x[:, 0::2] ** 2 + x[:, 1::2] ** 2  # noqa: E731
    torch.testing.assert_close(mag(shifted), mag(a))
    assert not torch.allclose(shifted, a, rtol=1e-3, atol=0.0)
    with pytest.raises(ValueError):
        S1Levels().only(["storm"])


def test_s1_windows_corrupt_only_the_da_side_and_run_end_to_end(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    lv = REFERENCE.scaled(1.0)
    s1 = apply_s1(ws, TINY, lv)
    w0, w1 = ws[0], s1[0]
    assert torch.equal(w1["true_state"], w0["true_state"])
    torch.testing.assert_close(w1["obs"], w0["obs"], equal_nan=True)
    assert w1["da_params"]["rd"] == pytest.approx(w0["true_params"]["rd"] * (1 - lv.param_bias))
    assert w1["da_params"]["U1"] == w0["true_params"]["U1"]
    assert w1["wind_lead_da"].shape == w0["wind_lead"].shape
    assert not torch.allclose(w1["wind_state_corrupted"], w0["wind_state_true"], rtol=1e-3, atol=0.0)
    payload, per_window = evaluate(s1, cfg, "etkf", torch.device("cpu"), N=6, loc_radius=4.0,
                                   init_lag_days=0.2, breed_days=2 / 12)
    assert len(per_window) == 2 and all(m["score"] == m["score"] for m in per_window)
    assert {"ev_da_psi1", "ev_free_q1", "rmse_da_q2"} <= set(per_window[0])


def test_shapley_is_exact_on_an_additive_toy():
    loss = {"amp": 0.1, "noise": 0.2, "shift": 0.05, "param": 0.3}
    recs = {}
    for kap, comps in tasks("shapley", 2.0):
        v = 1.0 - sum(loss[c] for c in comps)
        pw = [{"score": v, **{f"ev_da_{f}": v for f in ("psi1", "psi2", "q1", "q2")}}] * 3
        recs[task_name(kap, comps, 1.0)] = {"per_window": pw}
    out = shapley(recs, 2.0)
    for c, x in loss.items():
        assert out["score"]["components"][c]["shapley"] == pytest.approx(x)
    assert out["score"]["interaction"] == pytest.approx(0.0, abs=1e-12)


def test_obs_error_adds_white_and_per_pass_correlated_error():
    obs = torch.full((200, 32), float("nan"), dtype=torch.float64)
    obs[::2] = 0.0
    lv = S1Levels(obs_white_frac=0.15, obs_corr_frac=0.15)
    out = corrupt_obs(obs, 2.0, lv, key=4)
    assert torch.isnan(out[1::2]).all()
    err = out[::2]
    assert float(err.std()) == pytest.approx(2.0 * (0.15 ** 2 - 0.05 ** 2 + 0.15 ** 2) ** 0.5, rel=0.15)
    assert float(err.mean(dim=1).std()) > 0.5 * 2.0 * 0.15 * 0.5 ** 0.5
    assert obs_error_frac(lv) == pytest.approx((0.15 ** 2 + 0.15 ** 2) ** 0.5)
    assert corrupt_obs(obs, 2.0, S1Levels(), key=4) is obs


def test_realistic_groups_split_parameters_and_run_on_a_coarser_da_grid(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    lv = dataclasses.replace(REALISTIC, da_nx=8)
    only_rd = apply_s1(ws, TINY, lv.only(["rd"]))[0]
    assert only_rd["da_params"]["rd"] == pytest.approx(ws[0]["true_params"]["rd"] * 0.9)
    assert only_rd["da_params"]["rek"] == ws[0]["true_params"]["rek"]
    assert only_rd["da_nx"] == 16
    assert torch.equal(only_rd["wind_state_corrupted"], ws[0]["wind_state_true"])
    s1 = apply_s1(ws, TINY, lv)
    assert s1[0]["da_nx"] == 8 and s1[0]["da_params"]["rek"] == pytest.approx(ws[0]["true_params"]["rek"] * 0.5)
    _, per_window = evaluate(s1, da_cfg(cfg, lv), "etkf", torch.device("cpu"), N=6, loc_radius=2.0,
                             init_lag_days=0.2, breed_days=2 / 12)
    assert len(per_window) == 2 and all(m["score"] == m["score"] for m in per_window)
    assert da_cfg(cfg, lv).obs_noise_std_frac == pytest.approx(obs_error_frac(lv))
    names = [t["name"] for t in realistic_tasks("realistic_shapley", "base")]
    assert len(names) == 32 and names[0] == "s0" and names[-1] == "s1r_base"
