## 2026-10-21: L96 benchmark and P1 tables on the 2D (time x sector) backbone

**Summary:** The global L96 benchmark report gains the 2D circular-backbone rows (M tier with 3 planned seeds, S/L tiers, FDV1/FDV2-2D and their SDA hybrids), and the P1 report and the paper's L96 tables now use the 2D counterpart of every 1D U-Net row; rows with fewer than 3 seeds carry a note, rows without a 2D checkpoint yet (PredictStateCFM, SDA2, SDA3-fix, DirectUNet -> SDA3-fix) render as pending. Also lands the VanillaCFM-2D / SDA1-2D models and configs and the `--members-file scores` option of the FDV hybrid eval from `feature/l96-fdv-2d-circular`.
**Files modified:**
- `reports/l96/generate_l96_benchmark_extended_report.py` — `rows_2d()` (groups `2D backbone`, `2D SDA`, `2D hybrid`, `2D tiers`, `2D 4DVarNet`), finding 11, protocol note, 2D per-window summary, caveat
- `reports/l96/per_window_summary.py` — `rows(backbone=)` / `section(backbone=)`: the 2D variant swaps every U-Net row for its 2D counterpart
- `reports/l96/p1_benchmark_sections.py` — sections 2-4 read the 2D groups, seed/pending notes, 2D hybrids as reference rows, 2D-safe cross-family reading, 2D tier annex; the 1D tau=0 loader is gone
- `reports/l96/generate_p1_l96_benchmark.py` — 2D intro/protocol, pooled-skill section, FMS_τ and tau=0 pending for 2D, reconstruction figures from the 2D seed-1 models, the 1D P1-protocol annex replaced by the 2D tiers
- `reports/l96/p1_skill.py` (new) — pooled EV / CRPSS of the paper tables, ported from the uncommitted script behind the 1D numbers (its DA rows reproduce the paper's exactly)
- `reports/l96/outputs/{l96_benchmark_extended.md,p1_l96_benchmark.md,p1_l96_benchmark.json,figures/p1_l96_hovm_*.png}` — regenerated
- `reports/README.md` — index entries
- `docs/papers/p1_structural_hypotheses/sections_v4/05_results.tex` — L96 learned rows of `tab:s0`, `tab:s1`, `tab:prior` on the 2D backbone (markers: seed 1 only, SDA1 standing in for SDA2/SDA3, pending SS/FMSS), plus a `\todo` that the L96 prose, figures and method text still describe the 1D backbone
- Input bundle: 44 2D result directories added to `experiments/l96/report_inputs/benchmark_extended/here` with `scripts/bundle_report_inputs.py add` (each passed `check_learned`)
**Rationale:** The 2D backbone cuts every family's error (DirectUNet regular S0 0.340 -> 0.248, VanillaCFM 0.351 -> 0.220 per-window RMSE); P1 moves to it. Step 1 shows the current 2D results; the missing seeds and models are step 2 (training).
**Verification:** both reports regenerated from the bundle (every 1D number of `l96_benchmark_extended.md` unchanged); `pytest -m "not slow" tests/test_reports_index.py tests/test_report_inputs.py tests/test_docs_layout.py tests/test_ensemble_window_scores.py tests/test_members_store.py`; `ruff check` on the touched report modules; the paper compiles (latexmk).
