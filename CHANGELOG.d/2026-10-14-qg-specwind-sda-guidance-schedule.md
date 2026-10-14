## 2026-10-14: QG gyrostat SDA — guidance-weight schedule option

**Summary:** `evaluation/run_qg_specwind_neural.py --guidance-schedule {constant, linear-decay}` (sbatch `GSCHED=`): `linear-decay` makes the SDA guidance weight 2 w (1 − τ) along the flow — the same total push as a constant w, concentrated early on the large scales, leaving the late small-scale steps (which carry q) to the prior.
**Files modified:**
- `evaluation/run_qg_specwind_neural.py` — `guidance_schedule`, `GUIDANCE_SCHEDULES`, `--guidance-schedule`, recorded in the run meta
- `batch/run_qg_specwind_neural_eval.sbatch` — `GSCHED`
- `tests/test_qg_specwind_neural_eval.py` — schedule values and a scheduled-weight sampler run
**Rationale:** SDA started from pure noise peaks at ~0.3 on val (2D and 3D priors) against ~0.86–0.88 warm-started: weak guidance does not fit the obs, strong guidance fits ψ1 but leaks small-scale noise into ψ2/q. The 3D-prior sweep over the total push (weight × steps ∈ {10, 20, 50, 100}, steps ∈ {10, 20, 50}, which also covers a dt-scaled weight) adds the decaying schedule at 20 steps.
**Verification:** `pytest tests/test_qg_specwind_neural_eval.py` — 16 passed; ruff clean; sweep launched on the cluster.
