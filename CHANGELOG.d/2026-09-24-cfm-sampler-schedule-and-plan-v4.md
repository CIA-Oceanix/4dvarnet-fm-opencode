## 2026-09-24: S1 sampler test (stochastic / step-schedule) + τ-consistency plan v4

**Summary:** Adds `reports/l96/probe_stochastic_sampler.py`, which runs DDIM-η stochastic sampling and arbitrary step schedules on trained flows with no retraining, and runs it on the P1 PSC-M and VanillaCFM-M checkpoints. Rewrites the τ-consistency plan as v4, targeted at the variance collapse.
**Files modified:**
- `reports/l96/probe_stochastic_sampler.py`: new probe.
- `tests/test_stochastic_sampler.py`: 17 tests, including η=0 = Euler, marginal preservation for every η, and the fair CRPS.
- `reports/l96/outputs/cfm_tau_consistency/sampler/*.json`: outputs.
- `docs/results/cfm_sampler_schedule.md`: results.
- `docs/scoping/cfm_tau_consistency_next_steps.md`: v4.
- `docs/scoping/README.md`: index row.

**Rationale:**
- Measures how much of the flows' under-dispersion is fixable at sampling time.
- Stochasticity lowers the spread and is rejected.
- Uniform Euler spread/RMSE saturates with N (≈ 0.64 / 0.81), which bounds the sampler share.
- Early-fine steps (`tau_k = 1 − (1 − k/N)^0.5`) improve CRPS by 4.4% / 2.0% at the same cost.
- v4 therefore keeps the second-order loss T5 as the main lever.

**Verification:**
- `pytest tests/test_stochastic_sampler.py`: 17 passed.
- `ruff check` is clean on the touched files.
- Grids ran as SLURM jobs 55216, 55225 and 55233.
