## 2026-09-11: L96 — fix subgrad+state normalization dilution, soft-clip the grad term

**Summary:** Root-caused a severe fast-Y reconstruction collapse (variance
ratio ~0.6, correlation ~0.77 vs ~0.9+/~0.95 for every healthy scheme)
observed identically across all three trained gradient-conditioned FDV2
checkpoints (job 52926 `grad+state`, job 53011 `subgrad+state` equal-
capacity, job 53045 `subgrad+state` priorS+tbptt) to `_normalize_channels`'
global whole-tensor RMS norm being diluted by the sparse (1%-density,
`obs_interval=100`) `g_obs` channel: measured directly on the checkpoint,
RMS over all positions was 0.2656 vs 2.6558 over only observed positions --
exactly the `sqrt(1/0.01)=10x` dilution ratio, which then inflated the
sparse nonzero entries by that same factor at every observation. Two fixes:
(1) `subgrad+state`'s `g_obs`/`g_prior` no longer go through
`_normalize_channels` at all -- they're raw differences of already-
comparable-scale quantities (matches `ronan_devs`' own
`GradSolver_withStep`, which never normalizes them either). (2)
`grad-only`/`grad+state`'s normalization is kept (per prior published
results, Fablet et al. JAMES, on its importance for this solver class) but
its post-normalization bound is now a smooth `tanh` soft-clip
(`clip_range*tanh(t/clip_range)`, new `_soft_clip` helper) instead of a hard
`torch.clamp`, via a new independent `grad_clip_range` field (default `None`
-> falls back to `clip_range`, backward-compatible) so tightening it doesn't
also tighten the unrelated state-branch hard clamp. Also fixed a latent
wiring bug found along the way: `_build_update_input`'s `_normalize_channels`
call never threaded `clip_range` through, silently always using the
function's own `50.0` default regardless of the model's configured value.
**Files modified:** `models/fourdvarnet.py` (`_soft_clip`, `_normalize_channels`,
`_build_update_input`, `_solver_iteration`, `FourDVarNetSolver`/
`FourDVarNetPredictStateCFM.__init__`/`forward`); `conf/schema.py`
(`grad_clip_range` on both FDV configs); `train.py` (model_factory
threading); `tests/test_fourdvarnet.py` (new `TestSoftClip`,
`TestGradClipRange`; updated subgrad/normalization tests and tolerances);
`config/experiment/FDV2_grad_state_monai_l96.yaml` (`grad_clip_range=5.0`);
new `batch/run_l96_fdv2_{grad,subgrad}_state_monai*.sbatch` launch scripts.
**Rationale:** User pointed out `subgrad+state`'s structural similarity to
`FDV1` (obs recoverable from `g_obs`+`x`) should preclude a "gradient
issue" and asked directly whether normalization was applied there --
confirmed yes, and that normalization (not an architecture problem) was the
actual cause. `grad+state`'s own normalization was separately confirmed
load-bearing (kept, not removed) but its hard-clamp mechanism was replaced
with a smooth one per explicit request.
**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_lightning_module.py tests/test_l96_normalization_configs.py` --
all passing. `ruff check` clean. 1-epoch smoke tests for both fixes ran
cleanly before each real 400-epoch retrain (jobs 53077 [killed by user
request pending 53078's outcome], 53078, 53104) was launched via
`+fresh=true` (archiving, not resuming, the old now-stale checkpoints).
