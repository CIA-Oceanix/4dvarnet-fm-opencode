## 2026-09-14: FourDVarNetSolver prior_dropout decoupling knob

**Summary:** Added `prior_dropout` (default `None` -> falls back to `dropout`,
backward-compatible) to `FourDVarNetSolver`, decoupling `prior_unet`'s
dropout rate from the main solver unet's.

**Files modified:**
- `models/fourdvarnet.py` — `FourDVarNetSolver.__init__` gains
  `prior_dropout=None`; `self.prior_dropout = dropout if prior_dropout is
  None else prior_dropout`, used for `self.prior_unet`'s construction
  instead of the shared `dropout` arg. `self.unet` is unaffected (still
  built with `dropout`).
- `conf/schema.py` — `FourDVarNetConfig.prior_dropout: Optional[float] = None`.
- `train.py` — `model_factory` threads `fdv.get("prior_dropout", None)`.
- `tests/test_fourdvarnet.py` — new `TestPriorDropout` class (3 tests):
  default shares the main dropout, explicit value decouples it (checked via
  `model.prior_unet.bottleneck.drop.p` vs `model.unet.bottleneck.drop.p`),
  and the no-`prior_unet`-built case doesn't raise.
- `config/experiment/FDV2_grad_state_monai_l96_Stier_priorresidual_N5.yaml`,
  `batch/run_l96_fdv2_grad_state_monai_Stier_priorresidual_N5_train.sbatch`
  — new diagnostic config/launcher: grad+state (combined single-gradient
  mode, closer to the previously-working ocean4dvarnet/4dvarnet-global-
  mapping setup than gradsplit+state), `trainable_prior_weight: true`,
  `prior_residual: true`, `prior_dropout: 0.0` (main solver unet keeps
  `dropout: 0.1`), `N_outer: 5` (down from 10, faster + shorter unroll).

**Rationale:** `prior_unet`'s forward is repeatedly re-run inside
`torch.autograd.grad(..., create_graph=True)` for
grad-only/grad+state/gradsplit+state -- dropout there injects a fresh random
mask into that higher-order (double-backward) computation at every unrolled
iteration, on top of whatever noise the double-backward itself already
contributes (a distinct, separately-hypothesized source of training
difficulty from the Jacobian-magnitude issue `prior_residual` targets — see
`2026-09-14-fdv-prior-residual-diagnostic.md`). The main solver unet's own
forward is always a single, ordinary (first-order) backward, so dropout
there carries none of that risk and can be kept/increased freely as a
regularizer. This knob lets a config add dropout for the solver only,
without also adding mask noise to the fragile double-backward path.

**Verification:** `pytest tests/test_fourdvarnet.py -q` — new
`TestPriorDropout` tests pass alongside the full existing suite.
