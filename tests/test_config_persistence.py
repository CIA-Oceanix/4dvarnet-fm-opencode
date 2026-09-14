"""Tests for the checkpoint/resolved-config persistence infra.

train.py now saves the fully-resolved (defaults-composed) Hydra config next
to each experiment's checkpoints (``<exp_dir>/resolved_config.yaml``), and
evaluation.neural_inference auto-discovers + prefers it over state-dict
shape-inference. See PLAN.md / memory: l96-ckpt-config-persistence-plan.
"""
import hydra
import torch
from omegaconf import OmegaConf

from evaluation.neural_inference import (
    RESOLVED_CONFIG_FILENAME,
    _find_resolved_config,
    load_checkpoint,
    load_model,
)
from models.direct_unet import DirectUNet


def _compose(experiment_name):
    with hydra.initialize(config_path="../config", version_base="1.3"):
        return hydra.compose(config_name=f"experiment/{experiment_name}")


class TestResolvedConfigCompose:
    def test_l1_direct_unet_composes_and_saves_resolve_true(self, tmp_path):
        """The exact save call train.py now makes (OmegaConf.save(cfg, ...,
        resolve=True)) must succeed on a real composed experiment config --
        the main risk of this change is an unresolved interpolation/MISSING
        value blowing up resolve=True at training time.
        """
        cfg = _compose("L1_direct_unet_s0s1")
        out_path = tmp_path / RESOLVED_CONFIG_FILENAME
        OmegaConf.save(cfg, str(out_path), resolve=True)

        reloaded = OmegaConf.load(str(out_path))
        assert reloaded.model.model_type == "direct_unet"
        assert reloaded.model.state_dim == 24
        assert list(reloaded.model.direct_unet.hidden_channels) == [64, 128, 256]


class TestFindResolvedConfig:
    def test_discovers_config_next_to_checkpoints_subdir(self, tmp_path):
        exp_dir = tmp_path / "L1_direct_unet_s0s1"
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (exp_dir / RESOLVED_CONFIG_FILENAME).write_text("model: {model_type: direct_unet}\n")
        ckpt_path = ckpt_dir / "stage1.ckpt"
        ckpt_path.write_text("")

        found = _find_resolved_config(str(ckpt_path))
        assert found == str(exp_dir / RESOLVED_CONFIG_FILENAME)

    def test_discovers_config_in_same_dir_as_checkpoint(self, tmp_path):
        (tmp_path / RESOLVED_CONFIG_FILENAME).write_text("model: {model_type: direct_unet}\n")
        ckpt_path = tmp_path / "stage1.ckpt"
        ckpt_path.write_text("")

        found = _find_resolved_config(str(ckpt_path))
        assert found == str(tmp_path / RESOLVED_CONFIG_FILENAME)

    def test_returns_none_when_absent(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "stage1.ckpt"
        ckpt_path.write_text("")

        assert _find_resolved_config(str(ckpt_path)) is None


class TestLoadCheckpointPrefersResolvedConfig:
    def _save_lightning_ckpt(self, path, model, model_type):
        state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
        torch.save({"state_dict": state_dict, "hyper_parameters": {"model_type": model_type}}, str(path))

    def test_auto_discovered_config_skips_shape_inference(self, tmp_path):
        """With a real composed resolved_config.yaml sitting next to the
        checkpoint (no --config passed), load_model must build the model via
        train.model_factory from the declared architecture, not by reverse-
        engineering shapes from the state dict.
        """
        cfg = _compose("L1_direct_unet_s0s1")
        exp_dir = tmp_path / "L1_direct_unet_s0s1"
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        OmegaConf.save(cfg, str(exp_dir / RESOLVED_CONFIG_FILENAME), resolve=True)

        model = DirectUNet(state_dim=24, hidden_channels=[64, 128, 256],
                           dropout=0.1, param_dim=0, cond_extra_dim=0)
        ckpt_path = ckpt_dir / "stage1.ckpt"
        self._save_lightning_ckpt(ckpt_path, model, "direct_unet")

        loaded, loaded_cfg = load_model(str(ckpt_path))

        assert isinstance(loaded, DirectUNet)
        # Nested training-config schema, not the flat shape-inferred one.
        assert loaded_cfg.model.model_type == "direct_unet"
        assert "type" not in loaded_cfg.model or loaded_cfg.model.get("type") is None

        src, dst = model.state_dict(), loaded.state_dict()
        assert set(src) == set(dst)
        for k in src:
            assert torch.allclose(src[k], dst[k]), k

    def test_explicit_resolved_config_path_is_also_trusted(self, tmp_path):
        """A resolved_config.yaml explicitly passed via --config (e.g. an
        sbatch script pointing straight at experiments/<exp>/resolved_config.yaml
        instead of relying on auto-discovery) must be trusted exactly like an
        auto-discovered one, skipping shape-inference. Regression test for a
        real bug: an explicitly-passed resolved_config.yaml for a model type
        with no shape-inference support (e.g. a Monai-backed CFM variant)
        fell through to shape-inference and crashed with UnboundLocalError,
        because the old check gated on *how* config_path was obtained
        (auto-discovered vs explicit) rather than on the file itself.
        """
        cfg = _compose("L1_direct_unet_s0s1")
        exp_dir = tmp_path / "L1_direct_unet_s0s1"
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        resolved_path = exp_dir / RESOLVED_CONFIG_FILENAME
        OmegaConf.save(cfg, str(resolved_path), resolve=True)

        model = DirectUNet(state_dim=24, hidden_channels=[64, 128, 256],
                           dropout=0.1, param_dim=0, cond_extra_dim=0)
        ckpt_path = ckpt_dir / "stage1.ckpt"
        self._save_lightning_ckpt(ckpt_path, model, "direct_unet")

        # Explicit config_path, not auto-discovery.
        loaded, loaded_cfg = load_model(str(ckpt_path), config_path=str(resolved_path))

        assert isinstance(loaded, DirectUNet)
        assert loaded_cfg.model.model_type == "direct_unet"
        src, dst = model.state_dict(), loaded.state_dict()
        assert set(src) == set(dst)
        for k in src:
            assert torch.allclose(src[k], dst[k]), k

    def test_overrides_reach_the_model_on_auto_discovered_config_path(self, tmp_path):
        """overrides must reach the model when built via train.model_factory
        too, not just the legacy flat/shape-inferred path -- L2b trains
        train_tau_0_only=true; overriding it False must actually flip the
        constructed VanillaCFM's flag.
        """
        from models.vanilla_cfm import VanillaCFM

        cfg = _compose("L2b_vanilla_cfm_s0s1")
        exp_dir = tmp_path / "L2b_vanilla_cfm_s0s1"
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        OmegaConf.save(cfg, str(exp_dir / RESOLVED_CONFIG_FILENAME), resolve=True)

        model = VanillaCFM(state_dim=24, hidden_channels=[64, 128, 256],
                           param_dim=0, train_tau_0_only=True)
        ckpt_path = ckpt_dir / "stage1.ckpt"
        self._save_lightning_ckpt(ckpt_path, model, "vanilla_cfm")

        m_default, cfg_default = load_model(str(ckpt_path))
        assert m_default.train_tau_0_only
        assert cfg_default.model.vanilla_cfm.train_tau_0_only

        m_over, cfg_over = load_model(str(ckpt_path), overrides={"train_tau_0_only": False})
        assert not m_over.train_tau_0_only
        assert not cfg_over.model.vanilla_cfm.train_tau_0_only

    def test_load_checkpoint_returns_auto_discovered_cfg_directly(self, tmp_path):
        cfg = _compose("L1_direct_unet_s0s1")
        exp_dir = tmp_path / "L1_direct_unet_s0s1"
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        OmegaConf.save(cfg, str(exp_dir / RESOLVED_CONFIG_FILENAME), resolve=True)

        model = DirectUNet(state_dim=24, hidden_channels=[64, 128, 256],
                           dropout=0.1, param_dim=0, cond_extra_dim=0)
        ckpt_path = ckpt_dir / "stage1.ckpt"
        self._save_lightning_ckpt(ckpt_path, model, "direct_unet")

        _, loaded_cfg = load_checkpoint(str(ckpt_path))
        assert loaded_cfg.model.direct_unet.hidden_channels == [64, 128, 256]

    def test_no_resolved_config_falls_back_to_shape_inference(self, tmp_path):
        """No resolved_config.yaml anywhere -> legacy shape-inference path,
        unchanged behavior (regression guard for pre-existing checkpoints).
        """
        from models.vanilla_cfm import VanillaCFM

        model = VanillaCFM(state_dim=24, hidden_channels=[32, 64, 128], param_dim=0)
        ckpt_path = tmp_path / "stage1.ckpt"
        self._save_lightning_ckpt(ckpt_path, model, "vanilla_cfm")

        _, cfg = load_checkpoint(str(ckpt_path))
        assert cfg.model.get("model_type") is None
        assert cfg.model.type == "vanilla_cfm"
        assert list(cfg.model.hidden_channels) == [32, 64, 128]

    def test_explicit_config_path_is_not_treated_as_resolved(self, tmp_path):
        """An explicitly-passed --config that is NOT named resolved_config.yaml
        (possibly an incomplete raw preset) must keep going through the
        tolerant partial-merge path, not the strict train.model_factory path
        -- only a file literally named resolved_config.yaml is trusted to be
        fully field-complete, regardless of auto-discovery vs explicit path.
        """
        from models.vanilla_cfm import TweedieCFM

        model = TweedieCFM(state_dim=24, hidden_channels=[32, 64, 128],
                           K_inner=1, N_outer=10, sigma_prior=0.2)
        ckpt_path = tmp_path / "stage1.ckpt"
        self._save_lightning_ckpt(ckpt_path, model, "tweedie_cfm")

        # Deliberately incomplete: missing time_emb_dim/dropout/cond_extra_dim,
        # which train.model_factory would require via strict attribute access.
        cfg_path = tmp_path / "incomplete_preset.yaml"
        cfg_path.write_text(
            "model:\n"
            "  model_type: tweedie_cfm\n"
            "  state_dim: 24\n"
            "  tweedie_cfm:\n"
            "    hidden_channels: [32, 64, 128]\n"
            "    K_inner: 1\n"
            "    N_outer: 10\n"
            "    sigma_prior: 0.2\n"
            "    train_tau_0_only: false\n"
        )
        m, _cfg = load_model(str(ckpt_path), config_path=str(cfg_path))
        assert isinstance(m, TweedieCFM)
        assert m.K_inner == 1
