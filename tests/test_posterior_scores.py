import json
import os

import numpy as np
import pytest
import torch

from evaluation.metrics import crps as metrics_crps
from evaluation.posterior_scores import (aggregate, inflate, metric_stat, paired_bootstrap,
                                         reliability_index, window_scores)


def _calibrated_windows(n_windows: int = 40, M: int = 30, T: int = 50, D: int = 6, seed: int = 0) -> list:
    """Truth and members drawn from the same Gaussian predictive N(mu, s^2)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_windows):
        mu = torch.randn(T, D, generator=g)
        s = 0.3 + torch.rand(T, D, generator=g)
        truth = mu + s * torch.randn(T, D, generator=g)
        members = mu + s * torch.randn(M, T, D, generator=g)
        out.append(window_scores(members, truth))
    return out


def test_calibrated_gaussian_ensemble_has_unit_ssr_flat_ranks_and_nominal_coverage():
    agg = aggregate(_calibrated_windows())
    assert agg["ssr"] == pytest.approx(1.0, abs=0.03)
    counts = np.asarray(agg["rank_counts"])
    assert counts.size == 31
    assert agg["reliability"] < 0.12
    assert agg["cov50"] == pytest.approx(0.5, abs=0.015)
    assert agg["cov90"] == pytest.approx(0.9, abs=0.01)


def test_under_dispersed_ensemble_is_flagged():
    g = torch.Generator().manual_seed(1)
    truth = torch.randn(200, 4, generator=g)
    members = 0.3 * torch.randn(30, 200, 4, generator=g)
    s = window_scores(members, truth)
    assert s["ssr_window"] < 0.5
    assert s["reliability"] > 0.8
    assert s["cov90"] < 0.5


def test_crps_matches_evaluation_metrics_crps():
    g = torch.Generator().manual_seed(2)
    members = torch.randn(17, 30, 5, generator=g, dtype=torch.float64)
    truth = torch.randn(30, 5, generator=g, dtype=torch.float64)
    s = window_scores(members, truth)
    ref = metrics_crps(members.numpy(), truth.numpy())
    assert s["crps"] == pytest.approx(float(ref.mean()), rel=1e-10)
    M = 17
    assert s["crps_fair"] < s["crps"]
    pair = 2 * (np.abs(members.numpy()[:, None] - members.numpy()[None]).mean(axis=(0, 1)))
    fair = np.abs(members.numpy() - truth.numpy()).mean(0) - 0.25 * pair * M / (M - 1)
    assert s["crps_fair"] == pytest.approx(float(fair.mean()), rel=1e-10)


def test_single_member_crps_is_mae_and_rmse_is_per_channel_time_rmse():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 40, 3, generator=g)
    y = torch.randn(40, 3, generator=g)
    s = window_scores(x, y)
    assert s["crps"] == pytest.approx(float((x[0] - y).abs().double().mean()), rel=1e-6)
    assert s["rmse"] == pytest.approx(float(((x[0] - y) ** 2).double().mean(0).sqrt().mean()), rel=1e-6)
    assert s["var"] == 0.0


def test_energy_score_reduces_to_scaled_norm_for_one_member():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 10, 2, generator=g)
    y = torch.randn(10, 2, generator=g)
    s = window_scores(x, y)
    assert s["energy_score"] == pytest.approx(float((x[0] - y).double().norm()) / np.sqrt(20), rel=1e-6)


def test_reliability_index_bounds():
    assert reliability_index(np.ones(31)) == pytest.approx(0.0)
    extreme = np.zeros(31)
    extreme[0] = extreme[-1] = 5
    assert reliability_index(extreme) == pytest.approx(2 - 4 / 31)
    one_bin = np.zeros(31)
    one_bin[0] = 7
    assert reliability_index(one_bin) == pytest.approx(2 * 30 / 31)


def test_inflation_leaves_the_mean_unchanged_and_scales_spread():
    g = torch.Generator().manual_seed(5)
    members = torch.randn(30, 20, 4, generator=g, dtype=torch.float64) + 3.0
    infl = inflate(members, 1.7)
    torch.testing.assert_close(infl.mean(0), members.mean(0))
    torch.testing.assert_close(infl.std(0), 1.7 * members.std(0))
    truth = torch.randn(20, 4, generator=g, dtype=torch.float64)
    a, b = window_scores(members, truth), window_scores(infl, truth)
    assert a["rmse"] == pytest.approx(b["rmse"])
    assert b["ssr_window"] == pytest.approx(1.7 * a["ssr_window"])


def test_paired_bootstrap_of_identical_rows_is_zero_and_covers_zero():
    rows = _calibrated_windows(n_windows=20)
    for metric in ("rmse", "crps", "ssr", "reliability", "cov90", "energy_score"):
        r = paired_bootstrap(rows, rows, metric)
        assert r["diff"] == 0.0 and r["lo"] <= 0.0 <= r["hi"]
        assert r["n_boot"] == 2000


def test_paired_bootstrap_detects_a_shift_and_is_reproducible():
    rows = _calibrated_windows(n_windows=30)
    worse = [dict(w, rmse=w["rmse"] + 0.05) for w in rows]
    r1 = paired_bootstrap(worse, rows, "rmse")
    r2 = paired_bootstrap(worse, rows, "rmse")
    assert r1 == r2
    assert r1["lo"] > 0.0
    assert r1["diff"] == pytest.approx(0.05)


def test_metric_stat_ssr_matches_aggregate():
    rows = _calibrated_windows(n_windows=10)
    assert metric_stat(rows, "ssr")(np.arange(10)) == pytest.approx(aggregate(rows)["ssr"])


def test_harness_registry_and_argument_parsing():
    harness = pytest.importorskip("eval_fdv_fm_posterior_l96")
    assert {"B1", "B2", "B3", "B4", "B5", "PUB", "M1", "M1b", "M1bI", "M1c"} <= set(harness.METHODS)
    a = harness.parse_args_kv("tau0=0.5,gw=2", harness.METHODS["B3"].defaults)
    assert a["tau0"] == 0.5 and a["gw"] == 2 and a["n_steps"] == 10
    with pytest.raises(ValueError):
        harness.parse_args_kv("nope=1", harness.METHODS["B3"].defaults)
    assert harness.parse_windows("0:3,7", 10) == [0, 1, 2, 7]
    assert harness.window_seed(0, 3, "mean") == harness.window_seed(0, 3, "mean")
    assert harness.window_seed(0, 3, "mean") != harness.window_seed(0, 4, "mean")
    test_opts = harness.build_parser().parse_args(["--method", "B5", "--split", "test", "--out", "x.json"])
    with pytest.raises(ValueError):
        harness.resolve_inflation({"inflation": None, "inflation_from": None}, test_opts)
    assert harness.resolve_inflation({"inflation": 1.5, "inflation_from": None}, test_opts) == 1.5


def _write_val_run(path, harness, method: str, ssr: float, inflation=None, **overrides) -> str:
    margs = dict(harness.METHODS[method].defaults)
    config = {"method": method, "method_args": margs, "split": "val", "layout": "regular", "case": "s0",
              "fdv_seed": 2, "fdv_run": harness.FDV_RUNS[2], "sda_run": harness.SDA_RUN, "n_members": 30}
    if inflation is not None:
        config["inflation"] = inflation
    config.update(overrides)
    with open(path, "w") as f:
        json.dump({"config": config, "aggregate": {"ssr": ssr}}, f)
    return str(path)


def test_resolve_inflation_from_a_tuned_b5_val_run_uses_its_stored_factor(tmp_path):
    harness = pytest.importorskip("eval_fdv_fm_posterior_l96")
    src = _write_val_run(tmp_path / "b5.json", harness, "B5", ssr=1.0, inflation=1.7)
    opts = harness.build_parser().parse_args(["--method", "B5", "--split", "test", "--out", "x.json"])
    margs = harness.parse_args_kv(f"inflation_from={src}", harness.METHODS["B5"].defaults)
    assert harness.resolve_inflation(margs, opts) == pytest.approx(1.7)


def test_resolve_inflation_from_b3_inverts_ssr_and_rejects_mismatched_runs(tmp_path):
    harness = pytest.importorskip("eval_fdv_fm_posterior_l96")
    opts = harness.build_parser().parse_args(["--method", "B5", "--split", "test", "--out", "x.json"])
    b3 = _write_val_run(tmp_path / "b3.json", harness, "B3", ssr=0.5)
    margs = harness.parse_args_kv(f"inflation_from={b3}", harness.METHODS["B5"].defaults)
    assert harness.resolve_inflation(margs, opts) == pytest.approx(2.0)
    for name, method, overrides in [("b1", "B1", {}), ("tau", "B3", {"method_args": {"tau0": 0.5}}),
                                    ("lay", "B3", {"layout": "random"}), ("seed", "B3", {"fdv_seed": 1}),
                                    ("m", "B3", {"n_members": 10}), ("case", "B3", {"case": "s1"})]:
        if "method_args" in overrides:
            overrides = {"method_args": {**harness.METHODS[method].defaults, **overrides["method_args"]}}
        bad = _write_val_run(tmp_path / f"{name}.json", harness, method, ssr=0.5, **overrides)
        with pytest.raises(ValueError):
            harness.resolve_inflation(harness.parse_args_kv(f"inflation_from={bad}",
                                                            harness.METHODS["B5"].defaults), opts)


class _ToyPrior(torch.nn.Module):
    state_dim, sigma_prior = 4, 0.5

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, batch, tau: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x)) * (1.0 + tau.view(-1, 1, 1)) - x


class _ToyFDV(torch.nn.Module):
    prior_residual = False

    def __init__(self) -> None:
        super().__init__()
        self.prior_unet = lambda x, tau=None: 0.5 * x

    def sample(self, batch, N_outer: int = 10) -> torch.Tensor:
        return torch.nan_to_num(batch.obs, nan=0.0) * 0.9 + 0.01 * torch.randn_like(batch.obs)


def _toy_context(harness, n_members: int = 6, member_chunk: int = 0):
    torch.manual_seed(0)
    ns = {"mean": torch.zeros(4), "std": torch.tensor([1.9, 1.9, 1.7, 1.7])}
    ctx = harness.Context(torch.device("cpu"), ns, "fdv", "sda", n_members, member_chunk, 0)
    ctx._models.update(fdv=_ToyFDV(), sda=_ToyPrior())
    obs = torch.randn(1, 8, 4)
    obs[:, 1::2] = float("nan")
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, ::2] = True
    return ctx, {"obs": obs, "obs_mask": mask, "true_state": torch.randn(1, 8, 4)}


@pytest.mark.parametrize("member_chunk", [0, 4])
def test_m1_deterministic_normalized_raw_r_equals_b3(member_chunk):
    harness = pytest.importorskip("eval_fdv_fm_posterior_l96")
    ctx, batch = _toy_context(harness, member_chunk=member_chunk)
    b3, nfe_b3 = harness.METHODS["B3"].fn(ctx, batch, 3, harness.parse_args_kv(
        "tau0=0.5,gw=2.0,n_steps=10", harness.METHODS["B3"].defaults))
    m1, nfe_m1 = harness.METHODS["M1"].fn(ctx, batch, 3, harness.parse_args_kv(
        "tau0=0.5,gamma=0,guidance=normalized,gw=2.0,n_steps=5,r_units=raw", harness.METHODS["M1"].defaults))
    assert torch.allclose(b3, m1, atol=1e-5) and nfe_b3 == nfe_m1
    m1n, _ = harness.METHODS["M1"].fn(ctx, batch, 3, harness.parse_args_kv(
        "tau0=0.5,gamma=0,guidance=normalized,gw=2.0,n_steps=5", harness.METHODS["M1"].defaults))
    assert not torch.allclose(b3, m1n, atol=1e-5)


def test_m1_stochastic_pigdm_adds_spread_and_m1b_m1c_run():
    harness = pytest.importorskip("eval_fdv_fm_posterior_l96")
    ctx, batch = _toy_context(harness, n_members=8)
    d = harness.METHODS["M1"].defaults
    det, _ = harness.METHODS["M1"].fn(ctx, batch, 0, harness.parse_args_kv("gamma=0,guidance=pigdm", d))
    sto, nfe = harness.METHODS["M1"].fn(ctx, batch, 0, harness.parse_args_kv(
        "gamma=0.5,guidance=pigdm,n_correctors=1", d))
    assert sto.shape == det.shape == (8, 8, 4) and nfe == 10 * 2 + 10
    assert sto.std(0).mean() > det.std(0).mean()
    m1b, _ = harness.METHODS["M1b"].fn(ctx, batch, 0, harness.parse_args_kv("gamma=0.25", d))
    assert m1b.shape == (8, 8, 4) and torch.isfinite(m1b).all()
    spec = harness.METHODS["M1c"]
    margs = harness.parse_args_kv("a=0.7,b=0.6", spec.defaults)
    items = [(w, {**batch, "obs": batch["obs"] * (1 + w)}) for w in range(3)]
    spec.prepare(ctx, items, margs)
    tau0 = ctx.state["tau0_w"]
    assert set(tau0) == {0, 1, 2} and tau0[0] > tau0[2] and all(0.05 <= t <= 0.9 for t in tau0.values())
    x, _ = spec.fn(ctx, batch, 1, margs)
    assert x.shape == (8, 8, 4)
    xp, _ = spec.fn(ctx, batch, 1, harness.parse_args_kv("a=0.7,b=0.6,starts=perturbed", spec.defaults))
    assert xp.shape == (8, 8, 4) and not torch.allclose(xp, x)


def test_pub_runs_on_nan_obs_and_member_chunks_reproduce_the_seeded_members():
    harness = pytest.importorskip("eval_fdv_fm_posterior_l96")
    d = harness.METHODS["PUB"].defaults
    margs = harness.parse_args_kv("n_steps=5", d)
    out = {}
    for chunk in (0, 3):
        ctx, batch = _toy_context(harness, n_members=6, member_chunk=chunk)
        batch["obs"][:, 0, 1] = float("nan")
        out[chunk], nfe = harness.METHODS["PUB"].fn(ctx, batch, 2, margs)
        assert out[chunk].shape == (6, 8, 4) and torch.isfinite(out[chunk]).all() and nfe == 9
    assert d["gamma"] == 3e-4 and d["n_steps"] == 100 and d["n_correctors"] == 1
    assert out[0].std(0).mean() > 0


def test_obs_pert_scales_the_perturbed_obs_spread():
    harness = pytest.importorskip("eval_fdv_fm_posterior_l96")
    ctx, batch = _toy_context(harness, n_members=8)
    d = harness.METHODS["M1b"].defaults
    base, _ = harness.b2_perturbed_obs(ctx, batch, 0, harness.parse_args_kv("", d))
    same, _ = harness.b2_perturbed_obs(ctx, batch, 0, harness.parse_args_kv("obs_pert=1.0", d))
    wide, _ = harness.b2_perturbed_obs(ctx, batch, 0, harness.parse_args_kv("obs_pert=2.0", d))
    assert torch.equal(base, same) and wide.std(0).mean() > base.std(0).mean()
    assert harness.METHODS["M1bI"].tune_inflation and "obs_pert" in harness.METHODS["M1bI"].defaults


def test_fdv_mean_cache_is_keyed_by_data_and_reused_across_rows():
    harness = pytest.importorskip("eval_fdv_fm_posterior_l96")
    ctx, batch = _toy_context(harness)
    first = harness.fdv_mean(ctx, batch, 5, 10)
    assert torch.equal(first, harness.fdv_mean(ctx, batch, 5, 10)) and not harness.MEAN_CACHE
    ctx.data_key = "toy|s0|cpu"
    cached = harness.fdv_mean(ctx, batch, 5, 10)
    assert harness.fdv_mean(ctx, batch, 5, 10) is cached and torch.equal(cached, first)
    assert harness.fdv_mean(ctx, batch, 6, 10) is not cached and len(harness.MEAN_CACHE) == 2
    harness.MEAN_CACHE.clear()


PROBE_JSON = ("/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-fdv-mask/experiments/"
              "diag_resid_collapse/sda_warm.json")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(PROBE_JSON),
                    reason="needs CUDA and the 2026-10-12 sda_warm probe output")
def test_b3_reproduces_the_sda_warm_probe():
    import eval_fdv_fm_posterior_l96 as h
    from data.normalization import denormalize, load_norm_stats
    from evaluation.neural_inference import BatchDict, load_model, prepare_dataset
    from evaluation.sda_sampler import sda_guided_sample

    probe = json.load(open(PROBE_JSON))["fdv2_ds_coupled_s2/regular/s0/tau0=0.5/gw=2.0"]
    dev = torch.device("cuda")
    ns = load_norm_stats(h.NORM_STATS)
    fdv, cfg = load_model(f"{h.FDV_RUNS[2]}/checkpoints/stage1_best.ckpt", None, device=dev)
    sda, _ = load_model(f"{h.SDA_RUN}/checkpoints/stage1_best.ckpt", f"{h.SDA_RUN}/resolved_config.yaml",
                        device=dev)
    _, dls, ovi = prepare_dataset(cfg, h.DATASETS[("test", "regular")], 200, 100, obs_j=2, norm_stats=ns,
                                  batch_size=10)
    batch = {k: v.to(dev) if v is not None else v for k, v in next(iter(dls["s0"])).items()}
    torch.manual_seed(0)
    b = BatchDict(batch)
    with torch.no_grad():
        m_hat = fdv.sample(b, N_outer=10)
    mem, _ = sda_guided_sample(sda, b, R_var=0.5, N_outer=10, guidance_weight=2.0, n_members=30,
                               mean_estimate=m_hat, tau0=0.5)
    mem = denormalize(mem.detach().permute(0, 3, 1, 2), ns)
    truth = batch["true_state"][..., list(ovi)]
    for w in range(4):
        s = window_scores(mem[w], truth[w])
        assert s["rmse"] == pytest.approx(probe["rmse"][w], rel=2e-3)
        assert s["crps"] == pytest.approx(probe["crps"][w], rel=2e-3)
        assert s["spread_std"] == pytest.approx(probe["spread"][w], rel=2e-3)

    opts = h.build_parser().parse_args(["--method", "B3", "--args", "tau0=0.5,gw=2", "--split", "test",
                                        "--windows", "0:4", "--out", "unused.json"])
    res = h.run(opts)
    for w in range(4):
        s = res["per_window"][w]
        assert s["rmse"] == pytest.approx(probe["rmse"][w], rel=0.08)
        assert s["crps"] == pytest.approx(probe["crps"][w], rel=0.08)
        assert s["spread_std"] == pytest.approx(probe["spread"][w], rel=0.15)
