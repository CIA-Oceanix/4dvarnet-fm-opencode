import pytest
import hydra
from omegaconf import OmegaConf, DictConfig
from conf.schema import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig, BaselinesConfig, CS1Config, CS2Config


def test_schema_imports():
    """All schema classes can be imported and instantiated."""
    dc = DataConfig()
    assert dc.dt == 0.01
    assert dc.T_max == 3.0
    assert dc.system == "lorenz63"

    mc = ModelConfig()
    assert mc.state_dim == 3
    assert mc.hidden_channels == [64, 128, 256]

    tc = TrainingConfig()
    assert tc.stage1.epochs == 200
    assert tc.stage2.epochs == 400

    bc = BaselinesConfig()
    assert bc.da_window_steps == 300
    assert bc.weak4dvar.opt_steps == 150

    cs1 = CS1Config()
    assert cs1.param_bias == 0.0

    cs2 = CS2Config()
    assert cs2.param_bias == 0.15
    assert cs2.forcing_coupling == "quartic"


def test_config_yaml_loads():
    """Config YAML can be loaded via Hydra."""
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose("lorenz63_default")
    assert cfg is not None
    assert cfg.data.dt == 0.01
    assert cfg.data.system == "lorenz63"
    assert cfg.model.state_dim == 3
    assert cfg.training.stage1.epochs == 200
    assert cfg.baselines.da_window_steps == 300


def test_config_all_keys_present():
    """All expected top-level keys exist in the default config."""
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose("lorenz63_default")
    expected_keys = {"data", "model", "training", "paths", "baselines", "cs1", "cs2"}
    assert set(cfg.keys()) == expected_keys, f"Missing keys: {expected_keys - set(cfg.keys())}"


def test_data_section_keys():
    """All expected data keys exist."""
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose("lorenz63_default")
    data_keys = {
        "system", "dt", "T_max", "obs_interval", "R_var", "B_var",
        "num_windows", "window_spacing", "spinup_steps", "seed",
        "sigma_true", "rho_true", "beta_true", "gamma", "W_L_bar",
        "c1", "c2", "sigma_0", "sigma_L", "tau_eta", "sigma_eta",
        "forcing_state_bias", "forcing_coupling", "param_bias", "case",
    }
    assert set(cfg.data.keys()) == data_keys, f"Missing: {data_keys - set(cfg.data.keys())}"


def test_model_section_keys():
    """All expected model keys exist."""
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose("lorenz63_default")
    model_keys = {"state_dim", "hidden_channels", "time_emb_dim", "K_inner", "N_outer", "nu", "use_obs", "use_energy", "dropout"}
    assert set(cfg.model.keys()) == model_keys


def test_overrides_compose_correctly():
    """Command-line overrides correctly modify the config."""
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose("lorenz63_default", overrides=["data.dt=0.02", "data.case=2", "data.param_bias=0.1"])
    assert cfg.data.dt == 0.02
    assert cfg.data.case == 2
    assert cfg.data.param_bias == 0.1


def test_override_hidden_channels():
    """List-type overrides work."""
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose("lorenz63_default", overrides=["model.hidden_channels=[128,256]"])
    assert cfg.model.hidden_channels == [128, 256]


def test_dataclass_properties_match_old_config():
    """DataConfig properties match the old Lorenz63Config behavior."""
    dc = DataConfig(case=1, dt=0.01, T_max=5.0, param_bias=0.0)
    assert dc.num_steps == 500
    assert dc.use_corrupted_forcing is False
    sig, rho, bet = dc.da_params
    assert sig == 10.0
    assert rho == 28.0
    assert abs(bet - 8 / 3) < 1e-10

    dc2 = DataConfig(case=2, dt=0.01, T_max=5.0, param_bias=0.05)
    assert dc2.use_corrupted_forcing is True
    sig_b, rho_b, bet_b = dc2.da_params
    assert abs(sig_b - 10.0 * 0.95) < 1e-10
    assert abs(rho_b - 28.0 * 0.95) < 1e-10
    assert abs(bet_b - (8 / 3) * 1.05) < 1e-10


def test_experiment_config_roundtrip():
    """ExperimentConfig can be created from a DictConfig and contains all expected fields."""
    dc = DataConfig(dt=0.02, case=2)
    ec = ExperimentConfig(data=dc)
    assert ec.data.dt == 0.02
    assert ec.data.case == 2
    assert ec.model.state_dim == 3
    assert ec.training.batch_size == 32
    assert ec.paths.checkpoint_dir == "checkpoints"
    assert ec.baselines.da_window_steps == 300
    assert ec.cs1.param_bias == 0.0
    assert ec.cs2.param_bias == 0.15
