# Plan: Clean conditioning separation (Option B1) — L1/L2 only

## Confirmed decisions
- Option B1 (minimal refactor: `cond_extra_dim` in `UNet1D`/`ConditionEncoder`)
- New param: `cond_extra_dim` (default 0)
- Delete old L1/L2 checkpoints
- Scope: L1/L2 (L96) only; L63 configs untouched
- Forcing not always present: L1/L2 get `cond_extra_dim=0` (no forcing, no params)
- `cond_extra_dim` lives under sub-config in YAML (`direct_unet.cond_extra_dim` / `vanilla_cfm.cond_extra_dim`)
- PR workflow: Option A (GitHub PR via `scripts/open_pr.sh`)
- Retrain AFTER PR merge

## Design
- Backbone `UNet1D`/`ConditionEncoder` gains `cond_extra_dim` (default 0).
- `proj_in = state_dim + obs_dim + cond_extra_dim` where `obs_dim = state_dim` (24 for L96).
- Models build `cond` internally; pass `cond_extra_dim` to backbone at init.
- L1/L2 (τ=0, no forcing, no params): `cond_extra_dim = 0` → backbone sees `[x_zeros, obs]` only → `proj_in = 2*state_dim = 48`.

## Steps (pre-PR, on branch feat/l96-cond-extra-dim)
1. `models/unet.py` — add `cond_extra_dim` to `ConditionEncoder` + `UNet1D`; `proj_in += cond_extra_dim`
2. `models/direct_unet.py` — `__init__` takes `cond_extra_dim` (replaces `param_dim` in backbone sizing); `forward` builds `cond = obs` when `cond_extra_dim == 0` else `[obs, forcing, params]`
3. `models/vanilla_cfm.py` — same for `VanillaCFM`; `JointCFM` uses `cond_extra_dim = 1 + param_dim`, keeps `output_dim = state_dim + param_dim`
4. `conf/schema.py` — add `cond_extra_dim: int = 0` to `DirectUNetConfig`, `VanillaCFMConfig`; `JointCFMConfig` computes `1 + param_dim`
5. `train.py:model_factory` — pass `cond_extra_dim` from sub-config; drop `param_dim` arg for `DirectUNet`/`VanillaCFM` (keep for `JointCFM`)
6. Configs (sub-config placement):
   - `L1_direct_unet_s0s1.yaml` → `direct_unet.cond_extra_dim: 0`
   - `L2_vanilla_cfm_s0s1.yaml` → `vanilla_cfm.cond_extra_dim: 0`
   - `L1b`/`L2b` → same (`cond_extra_dim: 0`)
   - L63 configs untouched
7. `evaluation/neural_inference.py` — remove `obs_dim=24` hardcode + `model.unet.obs_dim` hack; pass `cond_extra_dim` from config; keep weight-shape inference as fallback
8. Tests — `test_direct_unet.py`/`test_vanilla_cfm.py`: add `cond_extra_dim=0` and `>0` cases; assert `proj.weight.shape == (hidden, 2*state_dim + cond_extra_dim)`
9. `CHANGELOG.md` — append dated entry

## PR
10. `scripts/open_pr.sh` → auto-review by `rfablet-review` → CI `pytest` gate → merge

## Post-merge (on master)
11. `rm -rf experiments/L1_direct_unet_s0s1/ experiments/L2_vanilla_cfm_s0s1/`
12. `sbatch batch/run_l96_neural_training.sbatch` (array 0-1 = L1, L2)
13. Wait, then evaluate:
    - `python eval_neural_l96.py --checkpoint experiments/L1_direct_unet_s0s1/checkpoints/stage1_best.ckpt --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt --output experiments/L1_direct_unet_s0s1/neural_eval.json`
    - Same for L2
14. `python reports/benchmark_table_l96.py`

## Verification
- `ruff check .`, `mypy .`, `pytest tests/ -m "not slow"`
