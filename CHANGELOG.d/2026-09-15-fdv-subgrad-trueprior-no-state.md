## 2026-09-15: FourDVarNetSolver subgrad+trueprior (true-ODE prior, no raw-state channel)

**Summary:** New `update_input="subgrad+trueprior"` mode -- identical to
`"subgrad+state+trueprior"` (the true L96 ODE as Phi) except the raw state
`x` is dropped from the update UNet's fed tensor: `cat([g_obs, g_prior])`
instead of `cat([g_obs, g_prior, x])`. Mirrors the existing `"grad-only"`
(no state) vs `"grad+state"` (with state) pair for the real-autograd-
gradient family -- an ablation testing whether the two residuals alone
carry enough information for the solver, since both already implicitly
encode `x` (`g_obs` directly at observed times, `g_prior` via the ODE
residual).
**Files modified:**
- `models/fourdvarnet.py` -- `_IMPLEMENTED_UPDATE_INPUTS`,
  `_UPDATE_INPUT_CHANNEL_MULTIPLIER["subgrad+trueprior"] = 2`,
  `_FULL_STATE_UPDATE_INPUTS` gains the new mode (all the full-state-target
  construction/validation/embedding logic already keys off this tuple
  generically, so no other code changes were needed there),
  `_build_update_input`'s trueprior branch now handles both modes (shared
  assert, `g_prior`/`g_obs` computation identical, only the final
  `torch.cat` differs).
- `tests/test_fourdvarnet.py` -- `TestFourDVarNetSolverSubgradTrueprior`
  (channel multiplier is 2, `_build_update_input` output excludes the state
  channel and starts with `g_obs`, forward shape/finiteness, gradients
  reach `unet`, checkpoint/no-checkpoint parity).
- `config/experiment/FDV2_subgrad_trueprior_monai_l96_perfectmodel.yaml`,
  `batch/run_l96_fdv2_subgrad_trueprior_monai_perfectmodel_train.sbatch` --
  same "perfect-model" S0-targeted training recipe as the sibling
  `subgrad+state+trueprior` config (state_dim=40, S-tier MonaiUNet1D, 400
  epochs), just the new `update_input`.
**Rationale:** User asked for a `subgrad+trueprior` variant "not providing
the state as inputs to the solver" -- a direct ablation of the just-shipped
`subgrad+state+trueprior` run (job 53564: S0 RMSE 0.459, worse than every
learned-prior scheme but comfortably better than all three DA baselines).
**Verification:** `pytest tests/test_fourdvarnet.py -q -k "SubgradTrueprior
or Trueprior"` (12 passed); `pytest tests/test_fourdvarnet.py -q` (full
regression, 129 passed, `fdv-monai-proto` env); `ruff check
models/fourdvarnet.py tests/test_fourdvarnet.py` clean.
