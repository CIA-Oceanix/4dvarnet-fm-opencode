"""QG run artifacts: the config training writes, and what an evaluation reads.

Pure-logic tests against a synthetic tree (tmp_path), deliberately independent
of the real ``experiments/`` directory: that directory is gitignored
machine-local data, so a test that required it would pass here and fail in CI.

What these guard is the failure this work removed: ``eval_qg_neural_s0_s1.py``
re-declared every scheme's architecture as a literal beside the checkpoint path,
with nothing connecting the two, so a config change would have loaded weights
into a differently-shaped model.
"""
import json

import pytest
import torch
from omegaconf import OmegaConf

from evaluation import archive, qg_runs


@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """experiments/ with a canonical QG archive dir and a legacy flat dir."""
    exp = tmp_path / "experiments"
    (exp / "qg" / "runA").mkdir(parents=True)
    (exp / "runB").mkdir(parents=True)
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    monkeypatch.setattr(archive, "EXPERIMENTS", exp)
    monkeypatch.setattr(qg_runs, "EXPERIMENT_CONFIGS", tmp_path / "config" / "experiment")
    return exp


def _resolved(**model) -> str:
    return OmegaConf.to_yaml(OmegaConf.create({
        "experiment_id": "runA", "system": "qg",
        "model": {"model_type": "direct_unet_tchannels", "param_dim": 0,
                  "cond_extra_dim": 0, "ic_dim": 0, **model},
        "data": {}, "training": {}}))


class TestArchitectureSource:
    def test_prefers_resolved_config(self, fake_tree):
        (fake_tree / "qg" / "runA" / "resolved_config.yaml").write_text(
            _resolved(param_dim=3, cond_extra_dim=1, ic_dim=2))
        (fake_tree / "qg" / "runA" / "results.json").write_text(json.dumps(
            {"model_type": "direct_unet", "config": {"param_dim": 0}}))
        arch = qg_runs.architecture("runA")
        assert arch["source"] == "resolved_config.yaml"
        assert (arch["param_dim"], arch["cond_extra_dim"], arch["ic_dim"]) == (3, 1, 2)

    def test_falls_back_to_results_json(self, fake_tree):
        """The Q1-Q4 runs archived before resolved_config.yaml existed must stay
        evaluable without anyone reconstructing a config by hand."""
        (fake_tree / "qg" / "runA" / "results.json").write_text(json.dumps(
            {"model_type": "direct_unet_tchannels",
             "config": {"param_dim": 3, "cond_extra_dim": 1, "include_ic": True,
                        "ic_dim": 2}}))
        arch = qg_runs.architecture("runA")
        assert arch["source"] == "results.json"
        assert arch["model_type"] == "direct_unet_tchannels"
        assert (arch["param_dim"], arch["cond_extra_dim"], arch["ic_dim"]) == (3, 1, 2)

    def test_results_json_ic_dim_derived_from_include_ic(self, fake_tree):
        """Older runs record include_ic without ic_dim."""
        (fake_tree / "qg" / "runA" / "results.json").write_text(json.dumps(
            {"model_type": "direct_unet", "config": {"include_ic": True}}))
        assert qg_runs.architecture("runA")["ic_dim"] == 2

    def test_falls_back_to_experiment_yaml(self, fake_tree):
        cfgdir = qg_runs.EXPERIMENT_CONFIGS
        cfgdir.mkdir(parents=True)
        (cfgdir / "runB.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create({
            "model_type": "direct_unet_tchannels",
            "model": {"param_dim": 3, "cond_extra_dim": 1},
            "data": {"include_ic": False}})))
        arch = qg_runs.architecture("runB")
        assert arch["source"] == "config/experiment"
        assert (arch["param_dim"], arch["ic_dim"]) == (3, 0)

    def test_unknown_run_names_every_place_it_looked(self, fake_tree):
        with pytest.raises(FileNotFoundError) as e:
            qg_runs.architecture("nope")
        assert "qg/nope" in str(e.value) and "config/experiment" in str(e.value)

    def test_legacy_layout_resolves(self, fake_tree):
        (fake_tree / "runB" / "results.json").write_text(json.dumps(
            {"model_type": "direct_unet", "config": {}}))
        assert qg_runs.architecture("runB")["source"] == "results.json"


class TestCheckpointChoice:
    def test_finished_run_prefers_the_bare_state_dict(self, fake_tree):
        run = fake_tree / "qg" / "runA"
        (run / "checkpoints").mkdir()
        (run / "checkpoints" / "stage1_best.ckpt").write_bytes(b"x")
        (run / "stage1_best.pt").write_bytes(b"x")
        assert qg_runs.checkpoint("runA").name == "stage1_best.pt"

    def test_interrupted_run_still_has_one(self, fake_tree):
        """A run killed at its wall limit (the Q5 sweep) has only Lightning's
        own checkpoints -- it must look interrupted, not missing."""
        run = fake_tree / "qg" / "runA"
        (run / "checkpoints").mkdir()
        (run / "checkpoints" / "stage1_last.ckpt").write_bytes(b"x")
        assert qg_runs.checkpoint("runA").name == "stage1_last.ckpt"

    def test_absent_is_none_when_optional(self, fake_tree):
        assert qg_runs.checkpoint("runA", required=False) is None


class TestNormStats:
    def test_recorded_path_wins(self, fake_tree):
        stats = fake_tree / "custom.pt"
        stats.write_bytes(b"x")
        (fake_tree / "qg" / "runA" / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.create({
                "model": {}, "data": {"norm_stats_path": "experiments/custom.pt"}})))
        assert qg_runs.norm_stats_paths("runA")["psi"] == stats

    def test_archive_copy_found_without_any_config(self, fake_tree):
        """The fix for the actual breakage: stats kept with the archive are
        reachable from a worktree that never trained anything."""
        shared = fake_tree / "qg" / "qg_psi_norm_stats.pt"
        shared.write_bytes(b"x")
        assert qg_runs.norm_stats_paths("runA")["psi"] == shared

    def test_absent_stats_are_none_not_an_error(self, fake_tree):
        """Only conditioned schemes have param/forcing stats."""
        assert qg_runs.norm_stats_paths("runA")["param"] is None


class TestResolvedConfigContents:
    """What training writes must be enough to rebuild the model it trained."""

    @pytest.fixture
    def written(self, tmp_path):
        import train_qg_neural as t

        class Cfg:
            nx, ny, state_dim = 64, 64, 8192
            obs_geometry, cols_per_day = "random_columns", 4
            obs_noise_std_frac, init_lag_days = 0.05, 5.0
            s1_param_bias = s1_amp_bias = 0.1

        model = torch.nn.Linear(3, 4)
        path = t.write_resolved_config(
            str(tmp_path), "Q1_x", model, model_type="direct_unet_tchannels",
            cfg=Cfg(), epochs=200, lr=1e-3, batch_size=2, gradient_clip_val=1.0,
            q_loss_weight=2.5e9, use_cosine_scheduler=True, seed=None,
            num_train=1000, num_val=100, num_test=100, train_seed=42,
            val_seed=10042, test_seed=20042, on_the_fly_split_obs=True,
            cache_dir="reports/qg_cache", normalize=True,
            norm_stats_path="experiments/qg_psi_norm_stats.pt",
            param_norm_stats_path=None, forcing_norm_stats_path=None,
            cond_mode="none", noisy_max=1.5, param_dim=0, cond_extra_dim=0,
            include_ic=False, ic_dim=0, cols_per_day_range=None, fdv_kwargs=None)
        return OmegaConf.load(path)

    def test_carries_every_architecture_field(self, written):
        for field in qg_runs.FIELDS:
            assert field in written.model, f"{field} missing from resolved_config.yaml"

    def test_records_the_capacity_actually_built(self, written):
        """Not the experiment YAML's `model.hidden_channels`, which build_model
        ignores for these model types -- recording that would state a capacity
        the weights may not have."""
        import train_qg_neural as t
        assert list(written.model.hidden_channels) == t.DEFAULT_HIDDEN_CHANNELS
        assert written.model.param_count == 16

    def test_records_cli_overridable_training_values(self, written):
        """The gap a copy of the source YAML leaves: these are argparse-driven."""
        assert written.training.epochs == 200
        assert written.training.batch_size == 2
        assert written.training.gradient_clip_val == 1.0
        assert written.data.train_seed == 42
        assert written.data.cache_dir == "reports/qg_cache"

    def test_records_the_observation_config_trained_against(self, written):
        assert written.data.init_lag_days == 5.0
        assert written.data.obs_noise_std_frac == 0.05

    def test_round_trips_through_the_reader(self, written, tmp_path, monkeypatch):
        exp = tmp_path / "experiments"
        (exp / "qg" / "runA").mkdir(parents=True)
        OmegaConf.save(written, exp / "qg" / "runA" / "resolved_config.yaml")
        monkeypatch.setattr(archive, "EXPERIMENTS", exp)
        arch = qg_runs.architecture("runA")
        assert arch["model_type"] == "direct_unet_tchannels"
        assert arch["source"] == "resolved_config.yaml"
