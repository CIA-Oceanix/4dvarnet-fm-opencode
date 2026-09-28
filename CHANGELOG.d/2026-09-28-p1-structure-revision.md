## 2026-09-28: P1 revised paper structure (scoping)

**Summary:** Scoping doc for a revised P1 structure: design matrix (F1 prior–likelihood split, F2 dynamics structure, F3 sequential inference, U-state, U-model) replaces the conditional-flow partition as the paper's instrument; restriction-vs-misspecification thesis; two-pass results (S0 then S1).
**Files modified:** `docs/plans/paper/l96_p1_structure_revision.md` — new SCOPING doc
**Rationale:** C4 and C5 of the current draft are not supported; the learned rows' S1 robustness is confounded by training on the true system.
**Verification:** `pytest tests/test_docs_layout.py -q`

**Addendum (same day): pooled calibration.** `reports/l96/probe_pooled_calibration.py` re-samples an L96 ensemble eval with members kept in memory and writes pooled spread/RMSE (E[var] vs E[sq. error], windows × time × channels); outputs for PSC/Van-M 1200 ep, SDA1/2/3-fix-M and the DU→SDA3-fix hybrid (seed 1, regular grid) in `reports/l96/outputs/pooled_calibration/`. The benchmark's per-window ratio understated the flows' spread (0.63–0.72 → pooled 0.86–0.96) and hid the filters' S0 over-dispersion (0.98 → 1.08); the structure doc's §3 table and K6 are updated. Verification: every row reproduces the report's RMSE and per-window ratio.

**Addendum 2: seeds 2–3 and the random layout.** SLURM array 56368 (`batch/run_l96_pooled_calibration.sbatch`, 30 tasks) adds seeds 2–3 on the regular grid and seeds 1–3 on the canonical random layout for every learned row; the structure doc's §3 table is now 3-seed means with ranges on both layouts (flows 0.84–0.94 pooled, SDA/hybrid 0.52–0.76, SDA2 0.90/0.86 at S1). Also in §3: the per-scheme error-component table (EnKF, EnKS, Strong-4DVar, SDAx) and the correction that the benchmark Strong-4DVar cycles six 500-step sub-windows.

**Addendum 3: DA after #291.** ETKF/EnKF rerun on master (SLURM 56384, `batch/run_l96_da_post291.sbatch` on `feature/l96-da-post291`), both layouts, per-time variance kept; pooled DA rows in `reports/l96/outputs/pooled_calibration/da_post291.json` and the §3 table. The fixed ETKF is 3% (regular) / 8% (random) worse at S0 at the default inflation 1.5 and equal/better at S1: the S0 inflation was tuned for the buggy square root and needs retuning.
