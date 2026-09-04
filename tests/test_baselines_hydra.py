"""Tests for Hydra-based baseline config and eval_baselines.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conf.schema import BaselinesConfig, Weak4DVarConfig, Strong4DVarConfig, EnKFConfig


class TestBaselinesConfigDefaults:
    def test_default_values(self):
        bc = BaselinesConfig()
        assert bc.da_window_steps == 300
        assert bc.batch_size == 128

    def test_weak4dvar_defaults(self):
        w = Weak4DVarConfig()
        assert w.opt_steps == 150
        assert w.lr == 0.02

    def test_strong4dvar_defaults(self):
        s = Strong4DVarConfig()
        assert s.max_iter == 40
        assert s.lr == 0.1

    def test_enkf_defaults(self):
        e = EnKFConfig()
        assert e.N_ensemble == 50
        assert e.inflation == 1.0


class TestBaselinesConfigYaml:
    def test_baselines_default_group(self):
        """The s0/s1 baseline profile is the project's sole default (no separate variant)."""
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        import hydra
        with hydra.initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "config")):
            cfg = hydra.compose(config_name="lorenz63")
            assert cfg.baselines.da_window_steps == 50
            assert cfg.baselines.enkf.N_ensemble == 50
            assert cfg.baselines.enkf.inflation == 2.0
            assert cfg.baselines.etkf.N_ensemble == 50
            assert cfg.baselines.etkf.inflation == 2.0

    def test_dws_override(self):
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        import hydra
        with hydra.initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "config")):
            cfg = hydra.compose(
                config_name="lorenz63",
                overrides=["baselines.da_window_steps=50"],
            )
            assert cfg.baselines.da_window_steps == 50

    def test_batch_size_override(self):
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        import hydra
        with hydra.initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "config")):
            cfg = hydra.compose(
                config_name="lorenz63",
                overrides=["baselines.batch_size=256"],
            )
            assert cfg.baselines.batch_size == 256
