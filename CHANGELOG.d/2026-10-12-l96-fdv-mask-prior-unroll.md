## 2026-10-12: L96 FDV M tier — observation mask, frozen prior and deep-supervision fine-tuning

**Summary:** Makes FDV2 at least as good as FDV1 at the M tier.
- Adds an explicit observation-mask channel to the residual inputs.
- Trains Φ as a prior: frozen in the solver input, and 10× slower than the solver for stability.
- Fine-tunes the unrolled solver with deep supervision beyond its 10-iteration horizon.

Over two seeds the result is 2–4% better than FDV1-M (pooled paired), and stable when iterated past 10 steps. Results note: `docs/results/l96_fdv_m_mask_prior_unroll.md`.
**Files modified:**
- `models/fourdvarnet.py`: update inputs `resid+mask+state`, `residnorm+mask+state`, `subgrad+mask+state`; update heads `gain` and `gated`; `freeze_prior_in_solver`; `prior_solver_grad_scale`; `tau_schedule` `exp` / `clamp`; burn-in with warm-up and ramp; `train_n_blocks`. All default off.
- `training/lightning_module.py`: per-epoch `set_epoch` hook.
- `training/resume.py`, `train.py`, `conf/schema.py`: `training.stage1.init_from` (weights-only fine-tuning) and the new fdv options.
- `batch/run_l96b_fdv_subgrad_aux.sbatch`: runs from the submitting worktree.
- `config/experiment/L96B_fdvresidmask*`, `L96B_fdvsubgradmask*`, `L96B_fdv1gated_monaiM.yaml`: ablation configs.
- `tests/test_fdv_*.py`: 5 new test files.
- `docs/results/l96_fdv_m_mask_prior_unroll.md`, `docs/README.md`: results note and index.

**Rationale:**
- The masked residual cannot signal "observed and already fitted"; the mask works as a gate.
- A Φ coupled to the solver's loss is a task feature, not a prior. FDV2 also trains unstably: 3 of 5 runs diverged at the full Φ learning rate.
- No pretrained solver stays stable past its trained horizon.
- The negative results (gain head, gated heads, β coupling, burn-in from scratch) are recorded in the note.

**Verification:** `pytest -m "not slow"` on the FDV, Lightning, resume, config and docs-layout tests; ruff on all touched files. Results: about 40 M-tier training runs and fine-tunes plus diagnostic probes. Their numbers are in the note.
