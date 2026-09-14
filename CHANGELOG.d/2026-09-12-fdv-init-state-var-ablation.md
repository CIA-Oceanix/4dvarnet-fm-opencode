## 2026-09-12: L96 — configurable non-zero-variance solver initial condition

**Summary:** New `FourDVarNetSolver` param `init_state_var` (default `0.0`,
backward-compatible): when `0.0`, the unroll starts from `x_0 = 0` (all
prior behavior, unchanged); when `>0`, `x_0 ~ N(0, init_state_var)` (a
VARIANCE, not a std -- e.g. `0.1` means std `~0.316` in this normalized
state space), sampled fresh on every `forward()` call, independent of
`update_input`. New experiment config
`config/experiment/FDV1_obsstate_monai_l96_initvar01.yaml` (`init_state_var:
0.1`, otherwise identical to `FDV1_unrolled_monai_unet_l96`) + matching
sbatch script.
**Files modified:** `models/fourdvarnet.py` (`FourDVarNetSolver.__init__`,
`_unrolled_blocks`'s `x_0` line); `conf/schema.py`
(`FourDVarNetConfig.init_state_var`); `train.py` (model_factory threading);
`tests/test_fourdvarnet.py` (new `TestInitStateVar`: default reproduces
exact zero-init, positive variance samples a statistically-correct nonzero
Gaussian init, resampled every forward call, zero-variance-with-N_outer>0
reproduces the pre-existing unroll bit-for-bit); new
`config/experiment/FDV1_obsstate_monai_l96_initvar01.yaml`,
`batch/run_l96_fdv1_obsstate_monai_initvar01_train.sbatch`.
**Rationale:** User asked directly whether FDV1's solver starts from a zero
initial condition (confirmed: yes, `x_0=0` for every FDV1/FDV2 config) and
requested a variant training FDV1 from a random, non-zero-variance initial
condition instead, as a sanity/robustness check on whether FDV1's healthy
fast-Y reconstruction (variance ratio ~0.93) depends on the deterministic
zero start in some way.
**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_lightning_module.py` -- 117 passed. `ruff check` clean. Manual
smoke test: built the model via `model_factory` from the new config,
confirmed `init_state_var=0.1`, ran one `compute_loss()`+`backward()` step,
confirmed two consecutive `forward()` calls on the same batch produce
different outputs (fresh random init each call).
