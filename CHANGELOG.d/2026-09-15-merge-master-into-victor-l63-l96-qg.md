## 2026-09-15: Merge master (L96/QG, 61 commits) into victor (L63)

**Summary:** Reconciled two independently-evolved architectures: master's L96/QG training
stack (checkpoint resume, monai backbones, param-head/SDA/FourDVarNet model types, cosine
LR + gradient loss, the `config/experiment`+`config/case_study` Hydra layout every L96
sbatch script still uses) with the L63 branch's `build_datasets()` refactor, ensemble
CRPS/R2 eval, and the newer flat `config/models`+`config/<system>.yaml` Hydra layout.
Verified via a three-way test comparison (this branch, the pre-merge L63 commit, and a
clean `origin/master` worktree) that found and fixed two real conflict-resolution
regressions: ETKF/EnKF's `assimilate()` had stopped updating the per-timestep
`analysis`/`ens_var` arrays (making DA baselines invariant to observation noise R), and
`train.py`'s `model_factory` had stopped forwarding an explicit `cond_extra_dim` config
override for several model types.
**Files modified:** `train.py`, `conf/schema.py`, `models/vanilla_cfm.py`,
`models/direct_unet.py`, `data/lorenz96.py`, `evaluation/baselines.py`,
`training/pipeline.py`, `training/lightning_module.py`, `training/losses.py`,
`config/experiment/*.yaml` (restored, previously deleted by an L63 config cleanup),
`config/lorenz96_default.yaml` (restored) — see commits `fef666a`, `aa1adcd`, `20ef7a6`.
**Rationale:** bring the L63 work and the 61 L96/QG commits it had diverged from back onto
one branch without regressing either side's experiment suite.
**Verification:** full `pytest tests/` run: 693 passed / 28 failed / 10 skipped, with every
one of the 28 failures cross-checked as pre-existing on both parent branches (not
introduced by the merge); all 81 `config/{experiment,models,baselines}/*.yaml` still
compose via Hydra.
