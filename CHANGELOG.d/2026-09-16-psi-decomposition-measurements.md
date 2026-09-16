## 2026-09-16: Psi_mean/Psi_G/Psi_NG measurements on L96 + a retraction

**Summary:** Measured the flow-matching operator decomposition on trained L96
checkpoints, two ways. (1) Query-based, as a function of fast-Y observation
density: `Psi_NG`'s share of the operator rises as observations thin (V2rerun
+74% on average; 2.6x at `tau=0.1`), but its share *relative to* `Psi_G` falls
(1.70 -> 0.62), because thinning widens the posterior and that is primarily a
variance effect. Three architectures converge to `NG/G = 0.62` at zero fast-Y
density. (2) Sample-based, from stored ensembles via the exact Gaussian-mixture
score (Tweedie collapses to a softmax-weighted member average): every scheme is
over-confident relative to its own error, with the ordering roughly inverse to
accuracy -- the warm-started hybrid has the best RMSE (0.425) and a quarter of
SDA3's spread (0.110 vs 0.423). Also a first look at two alternative conditional
samplers (variance-weighted PiGDM-style guidance; twisted SMC), both unresolved.

**Files modified:**
- `docs/psi_decomposition_results.md` — new results note (sections 1-3 + reproduction, environment and metric notes)
- `docs/cfm_affine_velocity_decomposition.md` — **retracts conclusion 5** (see Rationale)
- `reports/l96/probe_psi_decomposition_vs_density.py` — new; query-based probe, `--keep-k` density sweep, handles both the `PredictStateCFM`-style and two-stage `TweedieCFM` interfaces
- `reports/l96/probe_psi_from_members.py` — new; sample-based probe from stored `members_*.npz`
- `evaluation/sda_samplers_experimental.py` — new, **EXPERIMENTAL, not production**; PiGDM-style variance-weighted guidance (works, under-tuned) and twisted SMC (degenerate at this dimension, documented in the module docstring)
- `reports/l96/sweep_pigdm_guidance.py` — new; gamma sweep with step instrumentation and an unguided reference

**Rationale:** `cfm_affine_velocity_decomposition.md` recommended concentrating
network evaluations near `tau -> 1`, based on a late-tau residual blow-up seen in
V3 and FDV1CFM. V2rerun does **not** show it (0.204 at `tau=0.95`, falling to
0.174 under sparsity, vs V3's 0.57-0.62) despite near-identical mean quality
(0.522 vs 0.515), so the blow-up is a property of the residual/velocity stage,
not of the posterior. That recommendation is retracted rather than silently
dropped, since it was actionable and wrong.

Two further corrections recorded in the results note: `evaluate_estimates`
computes the **mean over dimensions of the per-dimension RMSE**, not pooled RMSE
(0.3786 vs 0.4087 for FDV1+SDA3-monai on S0), so
`l96_consolidated_benchmark.md`'s stated convention does not match its own
numbers; and the repo has two backbone-split environments (`fdv` for UNet1D-family,
`fdv-monai-proto` for every monai checkpoint) whose mismatch fails late inside
`train.py::model_factory`.

**Verification:** `ruff check` clean on all four new Python files; each script's
`--help` and the module import exercised. Measurement-only — no model, training,
config or test code touched, so the CI gate test list is unaffected (last full
run on this tree: 454 passed, 4 skipped).
