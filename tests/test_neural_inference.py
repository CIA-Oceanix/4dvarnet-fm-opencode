#!/usr/bin/env python3
"""Tests for neural inference and evaluation."""
import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from unittest.mock import Mock, patch

from evaluation.neural_inference import (
    load_checkpoint,
    resolve_model_class,
    create_model,
    load_model,
    run_inference,
    _run_case_inference,
)
from evaluation.estimate_metrics import evaluate_estimates, evaluate_npz
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM


class _DictBatch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _build_case_dataloader(truth, preds):
    """Build a single-case dataloader returning [truth] and [preds]."""
    from torch.utils.data import DataLoader, TensorDataset

    class _Collate:
        def __call__(self, batch):
            t = torch.stack([b[0] for b in batch])
            p = torch.stack([b[1] for b in batch])
            return {"true_state": t, "obs": p, "obs_mask": torch.ones(t.shape[0], dtype=torch.bool),
                    "forcing": torch.zeros(t.shape[0], t.shape[1])}


    ds = TensorDataset(truth, preds)
    return DataLoader(ds, batch_size=len(truth), collate_fn=_Collate())


class _IdentityModel:
    """Deterministic model returning the input obs as prediction."""
    def __call__(self, batch):
        return batch.obs


class TestNeuralInference:
    """Test neural inference utilities."""

    def test_resolve_model_class_direct_unet(self):
        """Test model class resolution for DirectUNet."""
        cfg = Mock()
        cfg.model = {"type": "DirectUNet"}
        model_class, cfg_model = resolve_model_class(cfg)
        assert model_class == DirectUNet

    def test_resolve_model_class_vanilla_cfm(self):
        """Test model class resolution for VanillaCFM."""
        cfg = Mock()
        cfg.model = {"type": "VanillaCFM"}
        model_class, cfg_model = resolve_model_class(cfg)
        assert model_class == VanillaCFM

    def test_resolve_model_class_unknown_type(self):
        """Test unknown model type raises error."""
        cfg = Mock()
        cfg.model = {"type": "UnknownModel"}
        with pytest.raises(ValueError, match="Unknown model type"):
            resolve_model_class(cfg)

    def test_create_model_direct_unet(self):
        """Test DirectUNet model creation."""
        cfg_dict = {
            "model": {
                "state_dim": 24,
                "hidden_channels": [64, 128, 256],
            }
        }
        cfg = OmegaConf.create(cfg_dict)
        model = create_model(DirectUNet, cfg)
        assert isinstance(model, DirectUNet)
        assert model.state_dim == 24

    def test_create_model_vanilla_cfm_tau0(self):
        """Test VanillaCFM model creation with tau=0."""
        cfg_dict = {
            "model": {
                "state_dim": 24,
                "hidden_channels": [64, 128, 256],
                "time_emb_dim": 64,
                "train_tau_0_only": True,
            }
        }
        cfg = OmegaConf.create(cfg_dict)
        model = create_model(VanillaCFM, cfg)
        assert isinstance(model, VanillaCFM)
        assert model.train_tau_0_only

    def test_state_mse_loss(self):
        """Test MSE computation."""
        preds = torch.randn(10, 24)
        true = torch.randn(10, 24)

        mse = torch.mean((preds - true) ** 2)
        assert isinstance(mse, torch.Tensor)
        assert mse.shape == ()
        assert mse > 0

    def test_evaluate_estimates_schema_and_values(self):
        """Generic evaluator: pooled RMSE/EV/ES grouped by component."""
        B, T, D = 4, 5, 24
        truth = np.zeros((B, T, D))
        traj = truth + 0.1  # constant 0.1 error
        m = evaluate_estimates(traj, truth)

        assert m["rmse"] == pytest.approx(0.1)
        for grp in ("slow", "obs_fast", "all_obs"):
            assert grp in m["groups"]
            assert grp in m["ev"]["groups"]
            assert grp in m["es"]["groups"]
        # EV = 1 - 0.01/0 = 1 with zero-variance truth is degenerate; check ok
        assert set(m.keys()) >= {"rmse", "groups", "ev", "es", "num_samples"}

    def test_run_inference_returns_estimates_and_truth(self):
        """run_inference returns per-case trajectories/truth arrays (no metrics)."""
        B, T, D = 4, 5, 24

        class StubDirectUNet(DirectUNet):
            """Minimal DirectUNet returning obs + offset (skips UNet init)."""
            def __init__(self):
                super().__init__(state_dim=D, hidden_channels=[4, 8])
                self.offset = 0.0

            def forward(self, batch):
                return batch.obs + self.offset

        s0_truth = torch.zeros(B, T, D)
        s1_truth = torch.zeros(B, T, D)

        model = StubDirectUNet()
        model.offset = 0.0
        dataloaders = {
            "s0": _build_case_dataloader(s0_truth, s0_truth),
            "s1": _build_case_dataloader(s1_truth, s1_truth),
        }

        est = run_inference(model, dataloaders, torch.device("cpu"))
        assert set(est.keys()) == {"s0", "s1"}
        for case in ("s0", "s1"):
            assert set(est[case].keys()) >= {"trajectories", "truth"}
            assert est[case]["trajectories"].shape == (B, T, D)
            assert est[case]["truth"].shape == (B, T, D)
            # metrics come from the generic evaluator
            m = evaluate_estimates(est[case]["trajectories"], est[case]["truth"])
            assert m["rmse"] == pytest.approx(0.0)

    def test_run_case_inference_uses_obs_var_indices_not_first_cols(self):
        """_run_case_inference must subsample the full truth by the non-contiguous
        observed subspace, not by the first `state_dim` columns.

        The bug fixed here: with a 40D truth and an identity observation predictor,
        ``all_true[..., :d_pred]`` grabbed the first 24 columns (mixing in 8
        unobserved fast vars Y3,Y4 of the first nodes), inflating RMSE ~2.6x.
        No dataset generation and no model are needed — we only need the model to
        return the input obs as its prediction so trajectories == obs, and truth to
        be subsampled by obs_var_indices.
        """
        NO, J, obs_j = 8, 4, 2
        obs_var_indices = tuple(list(range(NO)) +
                                [NO + k * J + j for k in range(NO) for j in range(obs_j)])
        assert len(obs_var_indices) == 24
        # non-contiguous: index 10 (Y3 of node 0) and 40-8-1 (last Y3,Y4) excluded
        assert 10 not in obs_var_indices
        assert 40 - 1 not in obs_var_indices

        B, T = 3, 5
        D = 24
        full_state = 40

        # truth: each of the 40 columns has a distinct constant value equal to its
        # column index, so we can check exactly which columns are selected.
        truth = torch.zeros(B, T, full_state)
        for c in range(full_state):
            truth[..., c] = c
        obs = truth[..., list(obs_var_indices)]  # already subsampled obs (24D)

        class _Identity(DirectUNet):
            def __init__(self):
                super().__init__(state_dim=D, hidden_channels=[4, 8])
            def forward(self, batch):
                return batch.obs

        model = _Identity()
        dataloader = _build_case_dataloader(truth, obs)
        out = _run_case_inference(model, dataloader, torch.device("cpu"), obs_var_indices)

        assert out["trajectories"].shape == (B, T, D)
        assert out["truth"].shape == (B, T, D)
        # trajectories == obs == truth[..., obs_var_indices]; with identity model and
        # the truth columns equal to their index, RMSE must be 0.
        m = evaluate_estimates(out["trajectories"], out["truth"])
        assert m["rmse"] == pytest.approx(0.0)

        # Direct check: the returned truth columns are exactly obs_var_indices.
        expected_truth = truth[..., list(obs_var_indices)].numpy()
        assert np.allclose(out["truth"], expected_truth)

    def _save_lightning_ckpt(self, tmp_path, model, model_type):
        state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
        path = tmp_path / f"stage1_{model_type}.ckpt"
        torch.save({"state_dict": state_dict, "hyper_parameters": {"model_type": model_type}}, str(path))
        return str(path)

    def test_infer_hidden_channels_reads_all_down_blocks(self, tmp_path):
        """hidden_channels must be recovered from downs.1 AND downs.2.

        The bug: the third channel was hardcoded to 256, so a [32,64,128]
        checkpoint built a [32,64,256] model and strict=False loading silently
        skipped every downs.2/ups weight (shape mismatch), producing garbage
        metrics with no error.
        """
        model = VanillaCFM(state_dim=24, hidden_channels=[32, 64, 128], param_dim=0)
        path = self._save_lightning_ckpt(tmp_path, model, "vanilla_cfm")
        _, cfg = load_checkpoint(path)
        assert list(cfg.model.hidden_channels) == [32, 64, 128]
        assert cfg.model.cond_extra_dim == 0

    def test_load_model_overrides_train_tau_0_only(self, tmp_path):
        """overrides must reach the instantiated model.

        The bug: Lightning hyper_parameters do not record train_tau_0_only, so
        tau=0-trained CFM checkpoints were loaded with the flag False and
        sampled via multi-step integration (residual-noise estimates) instead
        of the single Euler step used at training.
        """
        model = VanillaCFM(state_dim=24, hidden_channels=[32, 64, 128], param_dim=0)
        path = self._save_lightning_ckpt(tmp_path, model, "vanilla_cfm")
        m_default, _ = load_model(path)
        assert not m_default.train_tau_0_only
        m_tau0, cfg = load_model(path, overrides={"train_tau_0_only": True})
        assert m_tau0.train_tau_0_only
        assert cfg.model.train_tau_0_only

    def test_evaluate_npz_roundtrip(self, tmp_path):
        """evaluate_npz loads stored .npz and returns metrics."""
        from evaluation.estimate_metrics import save_estimates

        B, T, D = 3, 4, 24
        traj = np.random.randn(B, T, D)
        truth = np.random.randn(B, T, D)
        path = str(tmp_path / "est.npz")
        save_estimates(path, traj, truth)
        m = evaluate_npz(path)
        # RMSE = mean over dims of per-dim RMSE (same convention as the DA baselines)
        expected = float(np.mean(np.sqrt(np.mean((traj - truth) ** 2, axis=(0, 1)))))
        assert m["rmse"] == pytest.approx(expected)


class TestEnsembleInference:
    """Multi-member (N=30-style) CFM inference + ensemble ES evaluation."""

    def test_run_inference_multi_member_shapes_mean_and_dtype(self):
        from evaluation.estimate_metrics import evaluate_ensemble_estimates

        class StubCFM(VanillaCFM):
            def __init__(self):
                super().__init__(state_dim=24, hidden_channels=[4, 8], param_dim=0)

            def sample(self, batch, N_outer=1):
                return torch.randn_like(batch.obs) * N_outer

        B, T, D, M = 2, 3, 24, 3
        truth = torch.zeros(B, T, D)
        model = StubCFM()
        dl = {"s0": _build_case_dataloader(truth, truth)}

        est = run_inference(model, dl, torch.device("cpu"), n_members=M, n_outer=2)
        assert set(est["s0"].keys()) == {"trajectories", "truth", "members"}
        assert est["s0"]["members"].shape == (B, T, D, M)
        assert est["s0"]["members"].dtype == np.float32
        assert np.allclose(est["s0"]["trajectories"], est["s0"]["members"].mean(axis=-1))
        m = evaluate_ensemble_estimates(est["s0"]["members"], est["s0"]["truth"])
        assert m["ensemble"]["num_members"] == M

    def test_run_case_inference_multi_member_subsamples_truth_columns(self):
        class StubCFM(VanillaCFM):
            def __init__(self):
                super().__init__(state_dim=6, hidden_channels=[4, 8], param_dim=0)

            def sample(self, batch, N_outer=1):
                return batch.obs

        B, T, D_full, d_obs, M = 2, 3, 10, 6, 2
        full_truth = torch.arange(B * T * D_full, dtype=torch.float32).reshape(B, T, D_full)
        obs = torch.zeros(B, T, d_obs)
        idx = tuple(range(4)) + (8, 9)
        dl = {"s0": _build_case_dataloader(full_truth, obs)}
        out = _run_case_inference(StubCFM(), dl["s0"], torch.device("cpu"), idx, M, 1)
        expected_truth = full_truth.numpy()[..., list(idx)]
        assert np.array_equal(out["truth"], expected_truth)
        assert out["members"].shape == (B, T, d_obs, M)

    def test_pooled_ensemble_es_matches_per_window_energy_score_and_accumulator(self):
        from evaluation.estimate_metrics import ensemble_es_terms, pooled_ensemble_es
        from evaluation.metrics import energy_score

        rng = np.random.default_rng(7)
        W, T, D, M = 4, 5, 3, 4
        members = rng.normal(size=(W, T, D, M))
        truth = rng.normal(size=(W, T, D))

        es_tb = pooled_ensemble_es(members, truth, convention="textbook")
        es_per_window = np.mean(
            [energy_score(np.moveaxis(members[w], -1, 0), truth[w]) for w in range(W)], axis=0
        )
        assert np.allclose(es_tb, es_per_window)

        ensemble_es_terms(members, truth)
        acc_abs = np.zeros(D)
        acc_pw = np.zeros(D)
        for w in range(W):
            e = members[w]
            acc_abs += np.abs(e - truth[w][:, :, None]).mean(axis=(0, 2))
            for i in range(M):
                for j in range(M):
                    acc_pw += np.abs(e[:, :, i] - e[:, :, j]).mean(axis=0) / (M * M)
        es_cache_manual = acc_abs / W / M - 0.5 * acc_pw / W
        assert np.allclose(pooled_ensemble_es(members, truth, convention="cache"), es_cache_manual)

    def test_ensemble_es_degenerate_cases(self):
        from evaluation.estimate_metrics import pooled_ensemble_es

        W, T, D, M = 2, 4, 3, 5
        rng = np.random.default_rng(3)
        traj = rng.normal(size=(W, T, D))
        truth = rng.normal(size=(W, T, D))
        members = np.stack([traj] * M, axis=-1)

        # Identical members: spread term vanishes -> ES == MAE of the trajectory
        mae = np.mean(np.abs(traj - truth), axis=(0, 1))
        assert np.allclose(pooled_ensemble_es(members, truth, convention="textbook"), mae)
        assert np.allclose(
            pooled_ensemble_es(members, truth, convention="cache"), mae / M
        )
        # Single member: both conventions coincide with the MAE proxy
        single = traj[:, :, :, None]
        assert np.allclose(pooled_ensemble_es(single, truth, convention="cache"), mae)
        assert np.allclose(pooled_ensemble_es(single, truth, convention="textbook"), mae)

    def test_evaluate_ensemble_estimates_schema_and_member_mean_consistency(self):
        from evaluation.estimate_metrics import (
            evaluate_ensemble_estimates,
            evaluate_estimates,
        )

        W, T, D, M = 3, 4, 24, 6
        rng = np.random.default_rng(11)
        members = rng.normal(size=(W, T, D, M))
        truth = rng.normal(size=(W, T, D))

        m = evaluate_ensemble_estimates(members, truth)
        ref = evaluate_estimates(members.mean(axis=-1), truth)
        for key in ("rmse", "groups"):
            assert m[key] == pytest.approx(ref[key])
        assert set(m["ensemble"]) == {"num_members", "es_cache_convention", "es_textbook", "spread"}
        assert m["ensemble"]["num_members"] == M
        for blk in ("es_cache_convention", "es_textbook", "spread"):
            assert set(m["ensemble"][blk]["groups"]) == {"slow", "obs_fast", "all_obs"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
