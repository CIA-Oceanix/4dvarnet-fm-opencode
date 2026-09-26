import os
import shutil

import pytest
import torch

from data.qg_datasets import QGDatasetSpec, assemble_split, split_dir, write_shard
from evaluation.run_qg_baselines import _build_dyn, run
from evaluation.run_qg_specwind_da import build_cfg, s0_windows
from models.qg_batched import BatchedQGDynamics, SpectralWindForcing
from models.qg_dynamics import QGDynamics
from models.qg_wind_modes import FourierWindBasis

TINY = QGDatasetSpec(name="tiny_da", n_train=3, n_val=3, n_test=3, nx=16, spinup_days=1.0,
                     lead_days=0.5, window_days=1.0, burnin_units=2.0,
                     keep_every_trainval=6)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("da1"))
    for split in ("val", "test"):
        write_shard(TINY, split, root, 0, 1, batch_size=3)
        assemble_split(TINY, split, root)
    yield root
    d = split_dir(root, TINY, "test")
    for f in os.listdir(d):
        os.chmod(os.path.join(d, f), 0o644)
    shutil.rmtree(root, ignore_errors=True)


def test_spectral_qgdynamics_matches_batched_dynamics_with_eddy_drag():
    basis = FourierWindBasis(nx=16, kmax=2)
    a = torch.randn(10, basis.n_amp, dtype=torch.float64) * 2e-12
    ref = BatchedQGDynamics(1, nx=16, rd=14000.0, U1=0.05, rek=6e-7, r_cf=2.8e-8, dtype=torch.float64)
    q0 = ref.rollout(ref.initial_q([3]), 100)[1]
    batched, _ = ref.rollout(q0, 10, forcing=SpectralWindForcing(basis, a.unsqueeze(0)))
    dyn = QGDynamics(nx=16, rd=14000.0, U1=0.05, rek=6e-7, r_cf=2.8e-8, wind_driver="gyrostat",
                     dtype=torch.float64)
    single = dyn.rollout_trajectory(dyn._flatten(q0[0]), 10, wind_state=a)
    torch.testing.assert_close(single, ref._flatten(batched[0]), rtol=1e-10, atol=1e-20)
    assert dyn.wind_state_dim == 12


def test_default_qgdynamics_is_unchanged_by_the_hook():
    dyn = QGDynamics(nx=16, wind_amp=1e-11)
    assert dyn.wind_state_dim == 3 and dyn.r_cf == 0.0
    with pytest.raises(NotImplementedError):
        QGDynamics(nx=16, wind_driver="gyrostat").generate_wind_state(3)
    with pytest.raises(ValueError):
        QGDynamics(nx=16, wind_driver="storm")


def test_s0_windows_have_the_scenario_fields_and_index_seeded_obs(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.25)
    many, _ = s0_windows(TINY, "test", built, [0, 1, 2], cfg)
    one, _ = s0_windows(TINY, "test", built, [2], cfg)
    torch.testing.assert_close(many[2]["obs"], one[0]["obs"], equal_nan=True)
    w = many[0]
    assert w["da_model"] == "qg2l" and w["da_params"] == w["true_params"]
    assert w["wind_state_corrupted"].shape == (TINY.n_window, 12)
    assert w["target_state_psi"].shape == (TINY.n_window, 16 * 16)
    dyn = _build_dyn(cfg, w, torch.device("cpu"))
    assert dyn.inner.wind_state_dim == 12 and dyn.inner.r_cf == TINY.r_cf


def test_val_windows_are_regenerated_and_checked(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.25)
    ws, rep = s0_windows(TINY, "val", built, [1], cfg)
    assert rep["regenerated"] and rep["max_rel_diff"] <= 1e-4
    assert ws[0]["true_state"].shape[0] == TINY.n_window


def test_etkf_runs_end_to_end_on_s0_windows(built, tmp_path):
    # lag band [0.1, 0.42] d keeps the sampled lag above one model step: _sample_init_state
    # indexes out of range for a lag below one step (kk = 0), unreachable at the 5-day lag.
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.35)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    payload = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=6, inflation=1.0,
                  loc_radius=2.0, scenarios=("test_s0",), out_path=str(tmp_path / "s.json"),
                  init="lagged", geometry="random_columns", obs_var="psi", init_lag_days=0.35,
                  ds={"test_s0": ws}, etkf_ridge=0.1)
    s = payload["scenarios"]["test_s0"]
    assert s["expvar_full"] == s["expvar_full"]
    assert (tmp_path / "s.json").exists()
