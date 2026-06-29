"""Tests for Hydra-based baseline config and eval_baselines.py."""
import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conf.schema import BaselinesConfig, Weak4DVarConfig, Strong4DVarConfig, EnKFConfig


class TestBaselinesConfigDefaults:
    def test_default_values(self):
        bc = BaselinesConfig()
        assert bc.da_window_steps == 500
        assert bc.N_ensemble == 30
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
        assert e.inflation == 1.0


class TestBaselinesConfigYaml:
    def test_dws20_preset(self):
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        import hydra
        with hydra.initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "config")):
            cfg = hydra.compose(config_name="baselines/dws20")
            assert cfg.baselines.da_window_steps == 20

    def test_dws100_preset(self):
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        import hydra
        with hydra.initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "config")):
            cfg = hydra.compose(config_name="baselines/dws100")
            assert cfg.baselines.da_window_steps == 100

    def test_dws_override(self):
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        import hydra
        with hydra.initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "config")):
            cfg = hydra.compose(
                config_name="lorenz63_default",
                overrides=["baselines.da_window_steps=50"],
            )
            assert cfg.baselines.da_window_steps == 50

    def test_batch_size_override(self):
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        import hydra
        with hydra.initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "config")):
            cfg = hydra.compose(
                config_name="lorenz63_default",
                overrides=["baselines.batch_size=256"],
            )
            assert cfg.baselines.batch_size == 256
