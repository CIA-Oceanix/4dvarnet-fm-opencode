## 2026-09-26: L96 flow blends and velocity ensembles; PredictStateCFM-M x3 benchmark row

**Summary:** Sampling-time combinations of trained flows: a τ-varying blend of a VanillaCFM and a PredictStateCFM velocity field, and N-network velocity ensembles with equal or random τ-varying weights. Equal-weight averaging of independently trained networks is what helps (PredictStateCFM-M regular S0 0.341 -> 0.330 -> 0.327 for 1/2/3 networks); τ-dependent weights add nothing. A `PredictStateCFM-M x3` row is added to the extended benchmark report.
**Files modified:**
- `models/cfm_blend.py` — `TauBlendedFlow` (two flows, λ(τ) schedules), `MeanVelocityFlow` (N flows, equal or τ-varying weights), `random_weight_schedule`
- `eval_neural_l96.py` — `--blend-with` (one or more checkpoints), `--blend-schedule`, `--blend-weights`; recorded under `sampling`
- `tests/test_cfm_blend.py` — endpoint schedules reproduce each flow exactly, convex combination, same-family averaging, N-flow mean, seeded simplex schedules
- `batch/run_l96_cfm_blend.sbatch`, `batch/run_l96_cfm_blend_test.sbatch`, `batch/run_l96_cfm_ensemble3.sbatch`, `batch/run_l96_cfm_ens3_random_weights.sbatch`
- `reports/l96/generate_l96_benchmark_extended_report.py` — `PredictStateCFM-M x3` row, finding 8, cost note; outputs re-rendered
- `docs/results/l96_cfm_velocity_ensembles.md` (+ index in `docs/README.md`)

**Rationale:** VanillaCFM and PredictStateCFM are the same velocity regression with weights 1 and (1-τ)²; the blend tested whether a τ-varying mix combines PSC's accuracy with VC's spread. Controls (two seeds of one family, random τ-varying weights) show the gain is model averaging, maximal at equal weights.
**Verification:** `pytest tests/test_cfm_blend.py` (9 passed); λ = 0 reproduces plain VanillaCFM sampling exactly on real validation data; schedules selected on validation, scored once on test.
