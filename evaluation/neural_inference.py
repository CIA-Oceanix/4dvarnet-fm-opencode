"""Standalone neural model inference and evaluation for L96."""
import logging
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.lorenz96 import Lorenz96Config
from models.direct_unet import DirectUNet, JointDirectUNet
from models.vanilla_cfm import JointCFM, PredictStateCFM, TweedieCFM, VanillaCFM


class BatchDict:
    """Simple wrapper for batch dict access."""
    def __init__(self, d):
        self.__dict__.update(d)


def collate_eval(batch):
    """Collate function for RandomParamLorenz96Dataset (Dict format)."""
    states = torch.stack([b["true_state"] for b in batch])
    obs = torch.stack([b["obs"] for b in batch])
    masks = torch.stack([b["obs_mask"] for b in batch])
    forcing = torch.stack([b["forcing_corrupted"] for b in batch])
    return {"true_state": states, "obs": obs, "obs_mask": masks, "forcing": forcing, "params": None}


L96_JOINT_PARAM_NAMES = ("F", "c1", "hx", "eps", "w1", "w2", "w3", "w4")


def _window_param_vector(bd, prefix=""):
    """Extract the 8-param vector from a window dict, handling both cache formats.

    Newer datasets flatten fast_weights into scalar `w1..w4` (with `true_w1..`),
    matching the joint config's `param_names`. Older cached datasets store
    `fast_weights` / `true_fast_weights` as a length-4 list. Read the scalar
    keys when present, otherwise split the list.
    """
    vec = [float(bd[f"{prefix}{n}"]) for n in ("F", "c1", "hx", "eps")]
    list_key = f"{prefix}fast_weights"
    scalar_keys = [f"{prefix}w{j}" for j in range(1, 5)]
    if all(k in bd for k in scalar_keys):
        vec += [float(bd[k]) for k in scalar_keys]
    elif list_key in bd:
        fw = list(bd[list_key])
        if len(fw) < 4:
            fw = fw + [0.0] * (4 - len(fw))
        vec += [float(x) for x in fw]
    else:
        raise KeyError(f"missing fast_weights in window ({scalar_keys[0]}/{list_key})")
    return vec


def collate_joint_eval(batch):
    """Collate for joint models: also stack the 8 L96 params + true_params.

    Reads the 8-param vector via ``_window_param_vector``, transparently
    handling both the flattened (w1..w4) and legacy (fast_weights list) cache
    formats.
    """
    states = torch.stack([b["true_state"] for b in batch])
    obs = torch.stack([b["obs"] for b in batch])
    masks = torch.stack([b["obs_mask"] for b in batch])
    forcing = torch.stack([b["forcing_corrupted"] for b in batch])
    forcing_true = torch.stack([b["forcing_true"] for b in batch])
    params = torch.tensor([_window_param_vector(bd) for bd in batch])
    true_params = torch.tensor([_window_param_vector(bd, prefix="true_") for bd in batch])
    return {
        "true_state": states, "obs": obs, "obs_mask": masks, "forcing": forcing,
        "forcing_true": forcing_true, "params": params, "true_params": true_params,
    }


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, config_path: Optional[str] = None) -> tuple:
    """Load model checkpoint and config."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Handle Lightning .ckpt files
    if "hyper_parameters" in ckpt:
        # Lightning checkpoint: extract model_type from hyper_parameters
        model_type = ckpt["hyper_parameters"].get("model_type", "direct_unet")
        is_joint = "joint" in model_type

        # Infer architecture parameters from state_dict
        inferred_params = {}

        # The state/output UNet appears as `model.unet` for DirectUNet,
        # VanillaCFM, JointCFM, JointDirectUNet, PredictStateCFM and as
        # `model.velocity_unet` for the two-stage TweedieCFM.
        unet_key = "unet" if "model.unet.enc_out.2.weight" in state_dict else "velocity_unet"
        enc_out_key = f"model.{unet_key}.enc_out.2.weight"
        proj_key = f"model.{unet_key}.cond_encoder.proj.weight"
        downs_key = f"model.{unet_key}.downs."

        # enc_out.2 is Conv1d(in_c, output_dim, 3)
        if enc_out_key in state_dict:
            output_dim = state_dict[enc_out_key].shape[0]

        # proj_in = state_dim + 1 + output_dim for joint models
        # proj_in = 2*state_dim for non-joint
        if proj_key in state_dict:
            proj_in = state_dict[proj_key].shape[1]

        if is_joint:
            # JointCFM (refactored) and JointDirectUNet (refactored) both have a
            # single state UNet (output_dim = state_dim, cond_extra_dim = 1 for
            # [obs, forcing]) plus a separate param head: JointCFM uses
            # `param_flow` (a tau-flow CNN), JointDirectUNet uses `param_head`
            # (a deterministic regression CNN). `param_dim` is read from the
            # head's output channels. The legacy dual-head joint layout falls
            # back to the old inference below.
            param_flow_key = "model.param_flow.head.0.weight"
            param_head_key = "model.param_head.head.weight"
            if param_flow_key in state_dict or param_head_key in state_dict:
                # state UNet: cond_extra_dim = proj_in - 2*state_dim, output_dim = state_dim
                cond_extra_dim = proj_in - 2 * output_dim
                state_dim = output_dim
                head_key = param_flow_key if param_flow_key in state_dict else param_head_key
                param_dim = state_dict[head_key].shape[0]
            else:
                # Legacy JointDirectUNet / old joint dual-head layout:
                # state_dim = proj_in - 1 - output_dim, cond_extra_dim = 1 + param_dim
                state_dim = proj_in - 1 - output_dim
                param_dim = output_dim - state_dim
                cond_extra_dim = 1 + param_dim
            inferred_params["state_dim"] = state_dim
            inferred_params["param_dim"] = param_dim
            inferred_params["cond_extra_dim"] = cond_extra_dim
        else:
            # Non-joint: output_dim = state_dim
            state_dim = output_dim
            param_dim = 0
            if model_type == "tweedie_cfm":
                # TweedieCFM's velocity UNet uses obs_dim = 2*state_dim
                # (context = cat([obs_clean, mean])); proj_in = state_dim + 2*state_dim + cond_extra_dim.
                cond_extra_dim = proj_in - 3 * state_dim
            else:
                # cond_extra_dim = proj_in - 2*state_dim for obs_dim = state_dim
                cond_extra_dim = proj_in - 2 * state_dim

            inferred_params["state_dim"] = state_dim
            inferred_params["param_dim"] = param_dim
            inferred_params["cond_extra_dim"] = cond_extra_dim

        # Infer the param-flow CNN hidden channels for JointCFM from its conv
        # blocks (param_flow.blocks.N.conv1.weight: [out, in, 3]). Walk all
        # blocks present so a depth-3 param flow/head ([32,64,128], the L7/L8/L9
        # default) is recovered, not truncated to the first two layers.
        for head_name in ("param_flow", "param_head"):
            hb_blocks = [0]
            n = 1
            while f"model.{head_name}.blocks.{n}.conv1.weight" in state_dict:
                hb_blocks.append(n)
                n += 1
            if len(hb_blocks) > 1:
                inferred_params[f"{head_name}_channels"] = [
                    state_dict[f"model.{head_name}.blocks.{i}.conv1.weight"].shape[0]
                    for i in hb_blocks
                ]

        # Infer hidden_channels from downs layers
        # downs.N.block.conv1: [hidden[N], hidden[N-1], 3] -> read N=1 and N=2 so
        # the full triple is recovered for any depth-3 UNet (small nets included;
        # hardcoding the last channel as 256 broke [32,64,128] architectures).
        if f"{downs_key}1.block.conv1.weight" in state_dict:
            conv1 = state_dict[f"{downs_key}1.block.conv1.weight"]
            hidden = [conv1.shape[1], conv1.shape[0]]
            if f"{downs_key}2.block.conv1.weight" in state_dict:
                hidden.append(state_dict[f"{downs_key}2.block.conv1.weight"].shape[0])
            else:
                hidden.append(256)
            inferred_params["hidden_channels"] = hidden

        # Use inferred params or defaults
        cfg_dict = {
            "model": {
                "type": model_type,
                "state_dim": state_dim,
                "hidden_channels": inferred_params.get("hidden_channels", [64, 128, 256]),
                "time_emb_dim": 64,
                "param_dim": param_dim,  # 0 for obs-only; >0 for joint models
                "cond_extra_dim": inferred_params.get("cond_extra_dim", 0),
                "param_flow_channels": inferred_params.get("param_flow_channels", None),
                "param_head_channels": inferred_params.get("param_head_channels", None),
                "device": "cpu",
            },
            "deterministic": False,
        }
        cfg = OmegaConf.create(cfg_dict)
    else:
        # Custom checkpoint format
        cfg_dict = ckpt.get("config", {})
        if config_path:
            cfg = OmegaConf.load(config_path)
            OmegaConf.update(cfg, "model", cfg_dict.get("model", {}), merge=True)
        else:
            cfg_dict["model"]["device"] = cfg_dict.get("device", "cpu")
            cfg = OmegaConf.create(cfg_dict)

    # Recover tunable sampling params from a source training YAML when supplied.
    # Lightning checkpoints do not store the model config (LitModel saves
    # hyper_parameters with `ignore=["model"]`), so K_inner/sigma_prior/N_outer
    # must be read back from the experiment YAML via --config, else create_model
    # silently uses defaults and ablation evals (K_inner=1, sigma_prior=0.2)
    # sample with the wrong values.
    if config_path:
        try:
            yaml_cfg = OmegaConf.load(config_path)
            tc = yaml_cfg.model.get("tweedie_cfm", {})
            if hasattr(tc, "get"):
                m = cfg.model
                for key in ("K_inner", "N_outer", "sigma_prior", "train_tau_0_only",
                            "cond_extra_dim", "time_emb_dim", "dropout"):
                    if m.get(key) is None and key in tc:
                        m[key] = tc[key]
                if len(tc) > 0:
                    m.tweedie_cfm = tc
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not merge tweedie_cfm from {config_path}: {e}")

    return state_dict, cfg


def resolve_model_class(cfg: Any) -> tuple:
    """Resolve model class from config."""
    model_type = cfg.model.get("type", "DirectUNet")
    
    # Normalize model type (handle both "direct_unet" and "DirectUNet")
    model_type = model_type.replace("_", "").replace("-", "").upper()
    
    if model_type == "DIRECTUNET":
        return DirectUNet, cfg
    elif model_type == "VANILLACFM":
        return VanillaCFM, cfg
    elif model_type == "JOINTDIRECTUNET":
        return JointDirectUNet, cfg
    elif model_type == "JOINTCFM":
        return JointCFM, cfg
    elif model_type == "TWEEDIECFM":
        return TweedieCFM, cfg
    elif model_type == "PREDICTSTATECFM":
        return PredictStateCFM, cfg
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model(model_class, cfg: Any) -> torch.nn.Module:
    """Create model instance from config."""
    hidden = cfg.model.get("hidden_channels", cfg.model.get("hidden", [64, 128, 256]))
    
    if model_class == DirectUNet:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            cond_extra_dim=cfg.model.get("cond_extra_dim", 0),
        )
    elif model_class == VanillaCFM:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            time_emb_dim=cfg.model.get("time_emb_dim", 64),
            N_outer=cfg.model.get("N_outer", 10),
            sigma_prior=cfg.model.get("sigma_prior", 0.5),
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            train_tau_0_only=cfg.model.get("train_tau_0_only", False),
            cond_extra_dim=cfg.model.get("cond_extra_dim", 0),
        )
    elif model_class == JointCFM:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            time_emb_dim=cfg.model.get("time_emb_dim", 64),
            N_outer=cfg.model.get("N_outer", 10),
            sigma_prior=cfg.model.get("sigma_prior", 0.5),
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            param_loss_weight=cfg.model.get("param_loss_weight", 0.1),
            param_flow_channels=cfg.model.get("param_flow_channels", None),
            train_tau_0_only=cfg.model.get("train_tau_0_only", False),
        )
    elif model_class == JointDirectUNet:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            param_loss_weight=cfg.model.get("param_loss_weight", 0.1),
            param_head_channels=cfg.model.get("param_head_channels", None),
        )
    elif model_class == TweedieCFM:
        tc = cfg.model.get("tweedie_cfm", {})
        tc_get = tc.get if hasattr(tc, "get") else None

        def _tc(key, default):
            val = tc_get(key) if tc_get is not None else None
            if val is None:
                val = cfg.model.get(key, default)
            return val

        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            time_emb_dim=_tc("time_emb_dim", 64),
            K_inner=_tc("K_inner", 5),
            N_outer=_tc("N_outer", 10),
            sigma_prior=_tc("sigma_prior", 0.5),
            dropout=_tc("dropout", 0.1),
            train_tau_0_only=_tc("train_tau_0_only", False),
            cond_extra_dim=_tc("cond_extra_dim", 0),
        )
    elif model_class == PredictStateCFM:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            time_emb_dim=cfg.model.get("time_emb_dim", 64),
            N_outer=cfg.model.get("N_outer", 10),
            sigma_prior=cfg.model.get("sigma_prior", 0.5),
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 0),
            train_tau_0_only=cfg.model.get("train_tau_0_only", False),
            cond_extra_dim=cfg.model.get("cond_extra_dim", 0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_class}")
    
    return model


def load_model(checkpoint_path: str, config_path: Optional[str] = None, **kwargs) -> tuple:
    """Load model from checkpoint.

    ``overrides`` (optional dict) is merged into the constructed ``cfg.model``
    before instantiation. Needed for ``train_tau_0_only`` on tau=0-trained CFM
    checkpoints: Lightning hyper_parameters do not record it, and sampling a
    tau=0 model with multi-step integration adds residual noise to estimates.
    """
    overrides = kwargs.pop("overrides", None)
    state_dict, cfg = load_checkpoint(checkpoint_path, config_path)
    if overrides:
        for key, value in overrides.items():
            cfg.model[key] = value
    model_class, cfg_model = resolve_model_class(cfg)
    model = create_model(model_class, cfg_model)
    
    # Move model to correct device
    device = kwargs.get("device", "cpu")
    model.to(device)
    
    # Strip "model." prefix if present (Lightning wrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v  # Remove "model." prefix
        else:
            new_state_dict[k] = v
    
    # Filter out mismatched keys (e.g., output_dim mismatch)
    model_sd = model.state_dict()
    filtered_state_dict = {}
    for k, v in new_state_dict.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            filtered_state_dict[k] = v
        else:
            logger.warning(f"Skipping key {k}: checkpoint shape {v.shape} vs model shape {model_sd[k].shape if k in model_sd else 'missing'}")
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model, cfg


def prepare_dataset(
    cfg: Any,
    test_dataset_path: Optional[str] = None,
    num_test_windows: int = 200,
    obs_interval: int = 100,
    obs_var_indices: tuple | None = None,
    **kwargs,
) -> tuple:
    """Prepare S0/S1 test dataloaders for evaluation.

    Returns ``(dataset, dataloaders, obs_var_indices)`` where ``dataset`` is the
    cached S0/S1 dict and ``dataloaders`` maps ``"s0"``/``"s1"`` to ``DataLoader``
    over the matching test split. The cached dataset (as used by the DA baselines)
    is loaded when ``test_dataset_path`` is given; otherwise the S0/S1 train/val/
    test split is generated. ``obs_var_indices`` (the observed subspace of the
    full 40D state) is resolved if not supplied and returned to the caller so the
    model's 24D predictions can be compared against the correct truth columns.
    """
    # Try to load cached dataset
    if test_dataset_path:
        logger.info(f"Loading cached test dataset from {test_dataset_path}")
        dataset = torch.load(test_dataset_path, weights_only=False)
    else:
        # Generate test dataset
        logger.info(f"Generating test dataset: {num_test_windows} windows, obs_interval={obs_interval}")
        from data.lorenz96 import make_l96_s0_s1_trainval
        cfg_test = Lorenz96Config(
            system="lorenz96",
            NO=cfg.data.system_config.NO,
            J=cfg.data.system_config.J,
            obs_j=cfg.data.obs_j,
            obs_interval=obs_interval,
            num_test_windows=num_test_windows,
            seed=42,
        )
        split_dict = make_l96_s0_s1_trainval(cfg_test)
        dataset = {
            "test_s0": split_dict["test_s0"],
            "test_s1": split_dict["test_s1"],
        }

    # Resolve the observed-subspace indices of the full state. The cached DA
    # dataset stores the full 40D true_state with obs already subsampled to the
    # observed dims. When the model predicts a 24D observed state we must compare
    # against true_state[..., obs_var_indices] (a non-contiguous subset), NOT the
    # first `state_dim` columns.
    if obs_var_indices is None:
        obs_j = int(kwargs.get("obs_j", cfg.get("data", {}).get("obs_j", 2)))
        try:
            NO = int(cfg.data.system_config.NO)
        except Exception:
            NO = 8
        try:
            J = int(cfg.data.system_config.J)
        except Exception:
            J = 4
        from evaluation.run_l96 import make_obs_j_indices
        obs_var_indices = make_obs_j_indices(NO, J, obs_j)
        if obs_var_indices is None:
            obs_var_indices = tuple(range(NO + NO * J))
    obs_var_indices = tuple(obs_var_indices)

    # Create dataloaders for both the S0 and S1 test splits
    is_joint = bool(kwargs.get("is_joint", False))
    collate = collate_joint_eval if is_joint else collate_eval
    dataloaders = {}
    for key, case in (("test_s0", "s0"), ("test_s1", "s1")):
        split = dataset[key]
        dataloaders[case] = DataLoader(
            split,
            batch_size=kwargs.get("batch_size", 200),
            shuffle=False,
            collate_fn=collate,
            num_workers=kwargs.get("num_workers", 0),
        )

    return dataset, dataloaders, obs_var_indices


def _run_case_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    obs_var_indices: tuple | None = None,
    n_members: int = 1,
    n_outer: int = 1,
) -> dict:
    """Run a model on a single case dataloader and return state estimates.

    For stochastic samplers (VanillaCFM) each call to ``sample`` draws a fresh
    initial condition, so ``n_members > 1`` produces an ensemble of independent
    members; ``n_outer`` is the number of Euler integration steps. The returned
    dict holds numpy arrays ``{"trajectories": (W, T, D), "truth": (W, T, D)}``
    where ``D`` is the observed subspace; with ``n_members > 1`` it also holds
    ``"members": (W, T, D, M)`` (float32) and ``trajectories`` is the member
    mean. No metrics are computed here — that is the job of the generic
    evaluator. For joint models the batch also carries ``params``; each
    window's predicted params are returned in ``"params_pred"`` (W, P) and the
    ground-truth in ``"params_true"`` (W, P).
    """
    model.eval()
    is_joint = isinstance(model, (JointCFM, JointDirectUNet))
    member_preds: list[list] = [[] for _ in range(n_members)]
    member_param_preds: list[list] = [[] for _ in range(n_members)] if is_joint else None
    all_true = []
    param_trues = []
    x0_all = []
    forcing_true_all = []

    with torch.no_grad():
        for batch in dataloader:
            # Convert tensors to device, skip None values
            batch = {k: v.to(device) if v is not None else v for k, v in batch.items()}
            batch_obj = BatchDict(batch)

            for m in range(n_members):
                if isinstance(model, JointCFM):
                    pred, params = model.sample(batch_obj, N_outer=n_outer, return_params=True)
                elif isinstance(model, JointDirectUNet):
                    pred, params = model.sample(batch_obj, return_params=True)
                elif isinstance(model, DirectUNet):
                    pred = model(batch_obj)
                elif isinstance(model, VanillaCFM):
                    pred = model.sample(batch_obj, N_outer=n_outer)
                elif isinstance(model, (PredictStateCFM, TweedieCFM)):
                    pred = model.sample(batch_obj, N_outer=n_outer)
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")
                member_preds[m].append(pred.detach().float().cpu())
                if is_joint:
                    member_param_preds[m].append(params.detach().float().cpu())
            all_true.append(batch["true_state"].detach().cpu())
            if is_joint:
                param_trues.append(batch["true_params"].detach().cpu())
                # full initial state + clean forcing for the forecast-skill metric
                x0_all.append(batch["true_state"][:, 0].detach().cpu())
                forcing_true_all.append(batch["forcing_true"].detach().cpu())

    # Concatenate
    per_member = [torch.cat(mp, dim=0).numpy() for mp in member_preds]
    all_true = torch.cat(all_true, dim=0)

    # The neural model predicts the observed subspace while the cached truth is
    # the full state. Subsample truth to the observed dims for a fair comparison.
    # IMPORTANT: obs_var_indices is a NON-CONTIGUOUS subset (e.g. X1-X8 then
    # Y1,Y2 of each node), so we must use those exact columns, NOT the first
    # `state_dim` columns (which would mix in unobserved fast vars).
    d_pred = per_member[0].shape[-1]
    if all_true.shape[-1] > d_pred:
        if obs_var_indices is not None and len(obs_var_indices) == d_pred:
            all_true = all_true[..., list(obs_var_indices)]
        else:
            all_true = all_true[..., :d_pred]

    out = {
        "truth": all_true.detach().cpu().numpy(),
    }
    if n_members == 1:
        out["trajectories"] = per_member[0]
    else:
        members = np.stack(per_member, axis=-1).astype(np.float32)
        out["members"] = members
        out["trajectories"] = members.mean(axis=-1)
    if is_joint:
        # Each member predicts a (W, P) param vector; stack+mean to get the
        # per-window ensemble-mean params (W, P), matching params_true (W, P).
        per_member_params = [torch.cat(pp, dim=0).numpy() for pp in member_param_preds]
        out["params_pred"] = np.mean(np.stack(per_member_params, axis=0), axis=0)
        out["params_true"] = torch.cat(param_trues, dim=0).numpy()
        # Full-state initial conditions + clean forcing for the forecast-skill
        # metric (these must NOT be subsampled to the observed subspace).
        out["x0"] = torch.cat(x0_all, dim=0).detach().cpu().numpy()
        out["forcing_true"] = torch.cat(forcing_true_all, dim=0).detach().cpu().numpy()
    return out


def run_inference(
    model: torch.nn.Module,
    dataloaders: dict,
    device: torch.device,
    obs_var_indices: tuple | None = None,
    n_members: int = 1,
    n_outer: int = 1,
) -> dict:
    """Run inference on both S0 and S1, returning per-case estimates.

    Returns ``{"s0": {...}, "s1": {...}}`` where each entry holds the numpy
    ``trajectories``/``truth`` arrays (plus ``members`` when ``n_members > 1``;
    no metrics). To produce scores, pass these to the generic evaluator
    (``evaluate_estimates`` / ``evaluate_ensemble_estimates``).
    """
    return {
        case: _run_case_inference(model, dl, device, obs_var_indices, n_members, n_outer)
        for case, dl in dataloaders.items()
    }
