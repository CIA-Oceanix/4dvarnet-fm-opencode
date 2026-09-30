## 2026-09-30: QG EnKS time taper (τ = 8 days) replaces the hard lag; CRPS and spread columns in the report

**Summary:**
- **E1:** re-tuning the lag at the S1-tuned R × 6 (val) keeps lag 12 as the
  best hard lag. Lag 24 costs −0.014 and the whole window −0.040; the
  unobserved ψ₂ degrades.
- **E2:** a time taper is added to the localized EnKS:
  `EnKS(taper_steps)`, `run(enks_taper_steps)`, `--enks-taper-days`,
  sbatch `ENKSTAPER`. It is Gaspari–Cohn in time, zero beyond 2τ. On val,
  τ = 8 days with no lag cutoff beats lag 12 in S0 (+0.008) and S1-tuned
  (+0.007) and has the best CRPS in both.
- **E4:** it is now the benchmark EnKS. Test, taper − lag 12:
  - S0: +0.009 / +0.006 (0.812 / 0.819);
  - S1 with S0-tuned filters: +0.011 / +0.010 (0.612 / 0.620);
  - S1-tuned: a tie (0.665 / 0.673);
  - CRPS is better in all six cases.
- **E3:** the report's scenario tables gain CRPS (q, lowest-best, ranked)
  and median spread/RMSE q₁ columns. The EnKS has the best CRPS everywhere.

**Files modified:**
- `evaluation/baselines.py`: `_enks_smooth_localized(taper_steps)`,
  `EnKS(taper_steps)`.
- `evaluation/run_qg_baselines.py`, `evaluation/run_qg_specwind_da.py`:
  EnKS defaults are taper 8 d with no lag.
- `batch/run_qg_specwind_da.sbatch`.
- new `evaluation/qg_specwind_enks_tuning.py` and
  `batch/run_qg_specwind_enks_tuning.sbatch`.
- `reports/qg/generate_qg_specwind_da_report.py`: EnKS selection by
  lag/taper; CRPS and spread columns.
- `reports/qg/generate_qg_specwind_da_figs.py`; regenerated report and
  figures.
- `tests/test_enks_localized.py` (+1 test).
- `docs/results/qg_specwind_enks.md`,
  `docs/results/qg_specwind_da_s0_s1_test.md`,
  `docs/results/qg_specwind_s1_tuning.md`.

**Rationale:** E1–E4 of the smoother plan approved on 2026-09-30.

**Verification:**
- SLURM 57338 (val) and 57359–57362 and 57374–57375 (test): all completed.
- `pytest -m "not slow"` on the EnKS, specwind DA, S1 and docs/report tests;
  `ruff check` on the touched files.
