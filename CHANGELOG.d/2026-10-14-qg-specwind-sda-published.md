## 2026-10-14: QG gyrostat SDA — guidance as in the SDA paper (from pure noise)

**Summary:** `evaluation/run_qg_specwind_neural.py --sda-published` (sbatch `PUBLISHED=1`, `GAMMA`, `NCORR`, `LMC_TAU`, `OBS_SIGMA`) samples the SDA prior from pure noise with the guidance of Rozet & Louppe (2023) mapped to our linear flow x_τ = (1 − τ) x0 + τ x1, x0 ~ N(0, σ_p²): likelihood N(y | H x̂1, σ_y² + γ ((1 − τ) σ_p / τ)²), its score added to the prior's (velocity + (1 − τ) σ_p² / τ · ∇ log p), and Langevin corrector steps (step lmc_tau / mean(score²)) after each Euler step.
**Files modified:**
- `evaluation/run_qg_specwind_neural.py` — `PublishedSDA`, `published_sda_members`, `estimate_members(published=)`, `evaluate(published=)`, CLI options, recorded in the run meta
- `batch/run_qg_specwind_neural_eval.sbatch` — `PUBLISHED`, `GAMMA`, `NCORR`, `LMC_TAU`, `OBS_SIGMA`
- `tests/test_qg_specwind_neural_eval.py` — no-information limit equals the prior's Euler sample, guidance reduces the obs misfit with the corrector on, refuses a warm start
**Rationale:** our normalized-step guidance from noise peaks at ~0.36 on val (2D and 3D priors) against ~0.88 warm-started; the paper's noise-aware likelihood variance and Langevin correction are the published remedies for guidance acting on an uninformative x̂1 early in the flow, so this gives SDA a fair "as published" row. Steps kept within 30–100.
**Verification:** `pytest tests/test_qg_specwind_neural_eval.py` — 19 passed; ruff clean; val sweep launched (3D prior, γ ∈ {0.01, 0.1, 1} × steps ∈ {30, 100}, 1 corrector, N = 10).
