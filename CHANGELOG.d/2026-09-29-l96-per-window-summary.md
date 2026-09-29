## 2026-09-29: Per-window RMSE / EV / CRPS summary in the three current L96 reports

**Summary:** A shared section, one table each for RMSE, EV and CRPS (per window, 24 observed channels, mean ± sd over the 200 P1 test windows; regular / random × S0 / S1), covering the current benchmark rows:
- DA: ETKF / EnKF at the #295 inflation, the ETKS (#299), Strong-4DVar;
- SDA1/2/3-fix-M (gw 25, 400 ep);
- DirectUNet-M and PredictStateCFM-M / VanillaCFM-M (1200 ep);
- the best hybrid, for reference.

The same section is appended to `l96_benchmark_extended.md`, `l96_benchmark_default.md` and `p1_l96_benchmark.md`.
**Files modified:**
- `reports/l96/per_window_summary.py` (new) — reads the `benchmark_extended` bundle; EV = per-channel 1 − SSE/SST about the window mean; best value per column in bold, second best in italics.
- `reports/l96/generate_l96_benchmark_extended_report.py`, `generate_l96_benchmark_default_report.py`, `generate_p1_l96_benchmark.py` — call it before "Caveats"; outputs regenerated.
- `reports/README.md` — note on the shared section. `l96_consolidated_benchmark.md` is deliberately not updated (old protocol; digits quoted by the scoping docs).
- Bundle: the random-layout ETKF/EnKF trajectories of #298 and the ETKS random-layout trajectories of #299 added to `benchmark_extended/here` (they were only in worktree `experiments/`).
**Rationale:** EV and per-window CRPS were not reported side by side for every family; the three current reports now carry one identical, current summary.
**Verification:** Reports regenerated; `pytest tests/test_l96_report_consistency.py tests/test_docs_layout.py tests/test_reports_index.py tests/test_report_inputs.py`.
