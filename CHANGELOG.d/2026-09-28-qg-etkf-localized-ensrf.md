## 2026-09-28: Localized ETKF update — two defects diagnosed, exact EnSRF mode added (default unchanged)

**Summary:** The localized ETKF update (`loc_mode="square_root"`, the default)
has two problems:
1. its gain uses unnormalized sample covariances against an unscaled R, so R
   is effectively divided by N − 1;
2. it applies the full gain to the anomalies, dropping the +KRKᵀ term.

The new `loc_mode="ensrf"` is the exact localized EnSRF: normalized
covariances and the Andrews / Whitaker–Hamill modified gain for the
anomalies. `run()` now records per-window spread/RMSE per layer.

**Files modified:**
- `evaluation/baselines.py`: `_ensrf_localized_analysis`; the `ensrf`
  branch in both `ETKF.assimilate` and `ETKF.assimilate_batch`.
- `evaluation/run_qg_baselines.py`: `etkf_loc_mode`, `spread_ratio_list`.
- `evaluation/run_qg_specwind_da.py`: `--etkf-loc-mode`; spread ratios in
  the per-window metrics.
- new `evaluation/qg_specwind_etkf_check.py`,
  `batch/run_qg_specwind_etkf_check.sbatch`,
  `tests/test_etkf_ensrf_localized.py` (3 tests),
  `docs/results/qg_specwind_etkf_loc_update.md`.

**Rationale:** Under S1 the EnKF beat the ETKF on test. The val check (20
windows) confirms the over-confidence: legacy spread/RMSE in S1 is 0.58 /
0.76 (q₁ / q₂), against 0.69 / 0.92 for the EnKF.
- **EnSRF, radius 8, ridge 0.1:** S0 0.764 (legacy 0.754, EnKF 0.753) and
  S1 0.580 (legacy 0.551, EnKF 0.593). The S1 ETKF–EnKF gap shrinks from
  0.042 to 0.013, and S1 q₂ rises by +0.08 to +0.12.
- **In S0 the tuned ridge had masked the defects.**
- **The default is unchanged**, so every published result reproduces.
  Switching the QG gyrostat benchmark is a separate decision.

**Verification:**
- `pytest -m "not slow" tests/test_etkf_ensrf_localized.py
  tests/test_qg_specwind_da.py tests/test_qg_specwind_s1.py
  tests/test_qg_baselines.py tests/test_da_golden_l96.py`;
- `ruff check` on the touched files.
