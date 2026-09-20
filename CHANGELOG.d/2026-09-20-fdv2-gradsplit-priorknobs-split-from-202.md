## 2026-09-20: Split the FDV2 gradsplit + prior-knob layer out of PR #202

**Summary:** Second extraction from PR #202, after `L96Weak4DVar` (#229). This
lands the **foundation layer** of the FDV2 work: the `gradsplit+state`
update_input, the `subgrad+state+xtau` mode, the five prior-operator diagnostic
knobs, and the S-tier experiment configs.

**Why this layer and not the true-ODE prior, which was the original target.**
The trueprior commit (`c87e582`) does not stand alone. Cherry-picking it onto
master conflicts in four files (11 hunks in `models/fourdvarnet.py`) because its
code references five symbols that exist only further down #202's stack:
`prior_residual` (15 uses), `detach_var_cost_grad` (9), `subgrad+state+xtau` (7),
`gradsplit+state` (6), `gradsplit_prior_scale` (4). Its `_build_update_input`
branch calls `_prior_ae(..., residual=prior_residual)` and its mode tables
enumerate the gradsplit/xtau modes. Stripping those out would mean authoring a
variant of the trueprior code that was never trained or tested — unacceptable for
code whose output (S0 0.459) is meant to bear on H2. So #202's remainder is
landed bottom-up: this layer, then trueprior, then `loss_type="var_cost"` (which
was already known to depend on trueprior, being valid only for
`_FULL_STATE_UPDATE_INPUTS` modes).

**Files modified:** `models/fourdvarnet.py` (`gradsplit+state` in
`_IMPLEMENTED_UPDATE_INPUTS`/`_AUTOGRAD_MODES`/`_PRIOR_MODES`/the channel
multiplier, `subgrad+state+xtau` and `_CFM_ONLY_UPDATE_INPUTS`, and the
`gradsplit_prior_scale`/`prior_residual`/`prior_dropout`/`prior_output_init_std`/
`detach_var_cost_grad` constructor knobs); `models/monai_unet_adapter.py`
(`monai_output_init_std`); `conf/schema.py`, `train.py`,
`evaluation/neural_inference.py`, `eval_neural_l96.py`, `eval_fdv1_l96.py`
(`--batch-size` was never forwarded); 10 experiment configs + batch scripts;
`tests/test_fourdvarnet.py`, `tests/test_fourdvarnet_monai.py`,
`tests/test_neural_inference.py`.

**Merge conflicts resolved (3, all unions).** Master gained `qg_T`/`qg_ny`/`qg_nx`
parameters on `_build_backbone_unet` and `FourDVarNetSolver.__init__` after #202
forked, in the same signatures this layer extends. All three were resolved by
keeping both sets; no behaviour from either side was dropped.

**Excluded deliberately: the consolidated-benchmark rows.** #202's commit
`3050cf3` adds `grad+state-Stier-priorresidual-N5/tbptt2x5(monai)` rows to
`reports/l96/outputs/l96_consolidated_benchmark.md`, written against the
**pre-refactor** report. #219/#223 have since rebuilt that report to recompute
every row from archived trajectory arrays under one scoring formula per column,
and `tests/test_l96_report_consistency.py` guards it against the archive.
Hand-re-adding rows against that design is exactly what the refactor removed, so
the benchmark file here is byte-identical to master and its changelog fragment is
dropped. **Those two rows must be regenerated through the archive pipeline, not
transcribed** — until then the runs exist but are not benchmark rows.

**Not run:** `gradsplit+state` has configs, tests and batch scripts but no
training run. It is the control that separates "closed-form gradient beats
autodiff" from "unmixed beats summed" for the H4 question in
`docs/scoping/da_paper_structural_hypotheses.md` §10 decision 6.

**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_neural_inference.py -q -m "not slow"` (fdv-monai-proto) — **178
passed**. `ruff check` on every touched module — **8 errors, all pre-existing**,
verified by running ruff over master's own copies of the two offending files
(`evaluation/neural_inference.py` F541 x5, `tests/test_fourdvarnet_monai.py`
E402 x3) and getting an identical 8-for-8 result; this layer introduces none.
QG regression check (`-k qg`) run separately because the `qg_T`/`qg_ny`/`qg_nx`
union was the one resolution that could plausibly break an unrelated case study.
