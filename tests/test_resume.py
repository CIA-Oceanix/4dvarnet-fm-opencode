import os
import sys

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataloader import FlowMatchingBatch
from models.direct_unet import DirectUNet
from training.lightning_module import LitModel
from training.pipeline import create_trainer
from training.resume import config_fingerprint, resolve_experiment_dir, resume_ckpt_path


def _base_cfg(**training_overrides):
    cfg = OmegaConf.create({
        "model": {"model_type": "direct_unet", "hidden_channels": [4, 8], "param_dim": 0},
        "data": {"system": "lorenz96", "obs_interval": 20, "num_train_windows": 4},
        "training": {
            "stage1": {"epochs": 2, "gradient_clip_val": 10.0, "lr": 1e-3},
            "stage2": {"epochs": 0, "gradient_clip_val": 10.0, "lr": 1e-3},
            "accelerator": "cpu",
        },
        "paths": {"checkpoint_dir": "checkpoints", "outputs_dir": "outputs"},
    })
    for k, v in training_overrides.items():
        OmegaConf.update(cfg, k, v, merge=True)
    return cfg


class _RepeatBatchDataset(Dataset):
    """Yields the same tiny synthetic FlowMatchingBatch-shaped window every time."""

    def __init__(self, n=4, T=16, dim=3):
        self.n = n
        self.T = T
        self.dim = dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "states": torch.randn(self.T, self.dim),
            "obs": torch.randn(self.T, self.dim),
            "obs_mask": torch.ones(self.T, self.dim),
            "forcing": torch.zeros(self.T),
        }


def _collate(batch):
    states = torch.stack([b["states"] for b in batch])
    obs = torch.stack([b["obs"] for b in batch])
    obs_mask = torch.stack([b["obs_mask"] for b in batch])
    forcing = torch.stack([b["forcing"] for b in batch])
    return FlowMatchingBatch(states, obs, obs_mask, forcing)


def _tiny_loaders():
    ds = _RepeatBatchDataset()
    train = DataLoader(ds, batch_size=2, collate_fn=_collate)
    val = DataLoader(ds, batch_size=2, collate_fn=_collate)
    return {"train": train, "val": val}


def _tiny_lit(max_epochs):
    model = DirectUNet(state_dim=3, hidden_channels=[4, 8], param_dim=0, cond_extra_dim=0)
    return LitModel(model, model_type="direct_unet", stage=1, lr=1e-3, max_epochs=max_epochs)


class TestConfigFingerprint:
    def test_identical_configs_match(self):
        c1 = _base_cfg()
        c2 = _base_cfg()
        assert config_fingerprint(c1) == config_fingerprint(c2)

    def test_model_change_mismatches(self):
        c1 = _base_cfg()
        c2 = _base_cfg()
        c2.model.hidden_channels = [8, 16]
        assert config_fingerprint(c1) != config_fingerprint(c2)

    def test_training_change_mismatches(self):
        c1 = _base_cfg()
        c2 = _base_cfg()
        c2.training.stage1.epochs = 400
        assert config_fingerprint(c1) != config_fingerprint(c2)

    def test_paths_change_is_irrelevant(self):
        c1 = _base_cfg()
        c2 = _base_cfg()
        c2.paths.checkpoint_dir = "somewhere_else"
        assert config_fingerprint(c1) == config_fingerprint(c2)


class TestResolveExperimentDir:
    def test_missing_dir_is_a_noop(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        assert resolve_experiment_dir(exp_dir, _base_cfg()) is None
        assert not os.path.exists(exp_dir)

    def test_dir_without_checkpoints_is_a_noop(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        os.makedirs(exp_dir)
        assert resolve_experiment_dir(exp_dir, _base_cfg()) is None
        assert os.path.isdir(exp_dir)

    def test_matching_config_resumes_in_place(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        cfg = _base_cfg()
        os.makedirs(os.path.join(exp_dir, "checkpoints"))
        open(os.path.join(exp_dir, "checkpoints", "stage1_last.ckpt"), "w").close()
        OmegaConf.save(cfg, os.path.join(exp_dir, "resolved_config.yaml"))

        archived = resolve_experiment_dir(exp_dir, cfg)

        assert archived is None
        assert os.path.exists(os.path.join(exp_dir, "checkpoints", "stage1_last.ckpt"))

    def test_mismatched_config_archives_never_deletes(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        old_cfg = _base_cfg()
        new_cfg = _base_cfg()
        new_cfg.model.hidden_channels = [64, 128, 256]
        os.makedirs(os.path.join(exp_dir, "checkpoints"))
        ckpt_path = os.path.join(exp_dir, "checkpoints", "stage1_last.ckpt")
        with open(ckpt_path, "w") as f:
            f.write("not a real checkpoint, just needs to exist")
        OmegaConf.save(old_cfg, os.path.join(exp_dir, "resolved_config.yaml"))

        archived = resolve_experiment_dir(exp_dir, new_cfg)

        assert archived is not None
        assert not os.path.exists(exp_dir), "the stale dir must be moved away, not left in place"
        assert os.path.isdir(archived)
        assert os.path.exists(os.path.join(archived, "checkpoints", "stage1_last.ckpt"))
        with open(os.path.join(archived, "checkpoints", "stage1_last.ckpt")) as f:
            assert "not a real checkpoint" in f.read()

    def test_missing_resolved_config_archives_conservatively(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        os.makedirs(os.path.join(exp_dir, "checkpoints"))
        open(os.path.join(exp_dir, "checkpoints", "stage1_last.ckpt"), "w").close()

        archived = resolve_experiment_dir(exp_dir, _base_cfg())

        assert archived is not None
        assert not os.path.exists(exp_dir)

    def test_finished_plus_fresh_archives_even_if_config_matches(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        cfg = _base_cfg()
        os.makedirs(os.path.join(exp_dir, "checkpoints"))
        open(os.path.join(exp_dir, "checkpoints", "stage1_last.ckpt"), "w").close()
        OmegaConf.save(cfg, os.path.join(exp_dir, "resolved_config.yaml"))
        with open(os.path.join(exp_dir, "results.json"), "w") as f:
            f.write("{}")

        archived = resolve_experiment_dir(exp_dir, cfg, fresh=True)

        assert archived is not None
        assert not os.path.exists(exp_dir)

    def test_unfinished_plus_fresh_still_resumes(self, tmp_path):
        """--fresh only forces a new run of an *already-completed* experiment
        (e.g. an independent replicate) -- it must not turn into a way to
        accidentally nuke an in-progress/interrupted run's checkpoint."""
        exp_dir = str(tmp_path / "exp")
        cfg = _base_cfg()
        os.makedirs(os.path.join(exp_dir, "checkpoints"))
        open(os.path.join(exp_dir, "checkpoints", "stage1_last.ckpt"), "w").close()
        OmegaConf.save(cfg, os.path.join(exp_dir, "resolved_config.yaml"))

        archived = resolve_experiment_dir(exp_dir, cfg, fresh=True)

        assert archived is None
        assert os.path.isdir(exp_dir)


class TestResumeCkptPath:
    def test_no_checkpoint_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resume_ckpt_path(1) is None

    def test_existing_checkpoint_returns_its_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs("checkpoints")
        open(os.path.join("checkpoints", "stage1_last.ckpt"), "w").close()
        assert resume_ckpt_path(1) == os.path.join("checkpoints", "stage1_last.ckpt")
        assert resume_ckpt_path(2) is None


class TestCreateTrainerSavesLast:
    def test_last_checkpoint_named_per_stage(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = _base_cfg()
        trainer1 = create_trainer(cfg, 1)
        trainer2 = create_trainer(cfg, 2)
        cb1, cb2 = trainer1.checkpoint_callback, trainer2.checkpoint_callback
        assert cb1.save_last is True
        assert cb1.CHECKPOINT_NAME_LAST == "stage1_last"
        assert cb2.CHECKPOINT_NAME_LAST == "stage2_last"


@pytest.mark.slow
class TestEndToEndResume:
    def test_resume_continues_epoch_count_not_from_scratch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = _base_cfg()

        lit = _tiny_lit(max_epochs=2)
        trainer = create_trainer(cfg, 1)
        trainer.fit(lit, **_tiny_loaders_kwargs(), ckpt_path=resume_ckpt_path(1))
        assert trainer.current_epoch == 2
        ckpt_path = os.path.join("checkpoints", "stage1_last.ckpt")
        assert os.path.exists(ckpt_path)
        first_optimizer_state = trainer.optimizers[0].state_dict()

        cfg.training.stage1.epochs = 4
        lit2 = _tiny_lit(max_epochs=4)
        trainer2 = create_trainer(cfg, 1)
        resumed = resume_ckpt_path(1)
        assert resumed == ckpt_path
        trainer2.fit(lit2, **_tiny_loaders_kwargs(), ckpt_path=resumed)

        assert trainer2.current_epoch == 4
        resumed_optimizer_state = trainer2.optimizers[0].state_dict()
        assert resumed_optimizer_state["state"], "optimizer state must not be reset by resume"
        assert set(first_optimizer_state["state"].keys()) <= set(resumed_optimizer_state["state"].keys())

    def test_fingerprint_mismatch_archives_instead_of_bad_resume(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        exp_dir = "exp"
        cfg1 = _base_cfg()

        os.makedirs(exp_dir)
        assert resolve_experiment_dir(exp_dir, cfg1) is None
        os.chdir(exp_dir)
        lit = _tiny_lit(max_epochs=2)
        trainer = create_trainer(cfg1, 1)
        trainer.fit(lit, **_tiny_loaders_kwargs(), ckpt_path=resume_ckpt_path(1))
        OmegaConf.save(cfg1, "resolved_config.yaml")
        os.chdir("..")

        cfg2 = _base_cfg()
        cfg2.model.hidden_channels = [64, 128, 256]
        archived = resolve_experiment_dir(exp_dir, cfg2)
        assert archived is not None
        assert os.path.exists(os.path.join(archived, "checkpoints", "stage1_last.ckpt"))
        assert not os.path.exists(os.path.join(exp_dir, "checkpoints"))


def _tiny_loaders_kwargs():
    loaders = _tiny_loaders()
    return {"train_dataloaders": loaders["train"], "val_dataloaders": loaders["val"]}
