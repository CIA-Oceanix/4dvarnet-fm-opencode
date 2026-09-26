## 2026-09-26: L96 benchmark default raised to 1200 epochs; report follow-ups; bundle `add`

**Summary:** `config/l96_benchmark_default.yaml` now trains 1200 epochs (was 400); the 400-epoch benchmark configs pin `epochs: 400`, so every existing run resolves exactly as before. Reports: the extended report's rows are relabelled (400-epoch "recipe" vs the 1200-epoch default), its by-obs-count / fast-channel binned tables use the best scheme (DirectUNet-M(1200) -> SDA3-fix-M hybrid) instead of the 400-epoch SDA2 hybrid, and finding 2 states the new default; the P1 report gains an SDA3-fix-M row (the original SDA3-M never had its bias conditioning active); the benchmark-default report notes that it documents the 400-epoch runs. New `scripts/bundle_report_inputs.py add` links a new result directory into a report input bundle now that the legacy worktrees `record` ran against are pruned.
**Files modified:**
- `config/l96_benchmark_default.yaml`, `config/experiment/L96B_{directunet,vanillacfm,predictstatecfm}_monaiM.yaml` — 1200-epoch default; 400 pinned in the 400-epoch configs
- `AGENTS.md` — convention entry
- `reports/l96/generate_l96_benchmark_extended_report.py`, `generate_p1_l96_benchmark.py`, `generate_l96_benchmark_default_report.py` + outputs
- `scripts/bundle_report_inputs.py` — `add` subcommand; `tests/test_report_inputs.py` — its test
**Rationale:** the training-budget study showed 400 epochs is far from converged; the report sections still pointed at superseded SDA variants.
**Verification:** Hydra resolution of every `L96B_*` config unchanged (400 / 1200 as before; default 1200); `pytest tests/test_report_inputs.py` passed; the three reports regenerated from their bundles (`bundle_report_inputs.py run`).
