## 2026-10-04: Weak-constraint 4D-Var benchmark rows (L96, regular grid)

**Summary:** `L96Weak4DVar` (whitened controls, per-step model-error increment scaled by q, LBFGS 40 iterations) gets benchmark rows on the 200-window regular test set:
- **S0** (q 0.03): 0.695, level with Strong-4DVar (0.703).
- **S1** (q 0.3): 1.064, −26% vs Strong-4DVar (1.436) — the best DA row at S1, ahead of the ETKS (1.338).

The rows go into the extended report (main-table row, finding 10) and into section 1 of the P1 report. There is no random-layout row yet: the observation cost does not mask missing channels.

**Files modified:**
- `scripts/weak4dvar_sweep_l96.py` — the sweep / benchmark driver, with `W_START` / `CASES` / `KEEP_DIR` options.
- `batch/run_l96_weak4dvar_{sweep,diag,sweep_it40,bench}.sbatch` — the runs.
- `scripts/assemble_weak4dvar_bench.py` — assembles the 40 chunks and rescores them against the test truth.
- `reports/l96/generate_l96_benchmark_extended_report.py` — Weak-4DVar row, `check_weak4dvar()`, finding 10; DA rows may lack random-layout columns.
- `reports/l96/p1_benchmark_sections.py` — DA row; tables accept rows without random columns.
- `docs/results/l96_weak4dvar.md` — new: validation sweeps, LBFGS budget, per-case selection, benchmark rows.
- `docs/README.md`, `reports/README.md` — indexes.
- The assembled rows are hard-linked into the bundle at `benchmark_extended/here/weak4dvar_2026-10-04/`.

**Rationale:** Weak-4DVar was missing from the benchmark: the generic `Weak4DVar` diverges on L96. It is the model-uncertainty arm of the hard constraint (P1 claim K4/K5). The model-error scale and the LBFGS budget are tuned on the validation windows, for tuning parity.

**Verification:** the assembled rows cover windows 0–199 once per case and match the chunks' RMSE to 1e-4, with no NaN resets. `bundle_report_inputs.py run` for benchmark_extended and p1_benchmark. `pytest tests/test_p1_benchmark_sections.py tests/test_l96_report_consistency.py tests/test_report_inputs.py tests/test_reports_index.py tests/test_docs_layout.py tests/test_batch_node_tmp.py -m "not slow"`: 166 passed.
