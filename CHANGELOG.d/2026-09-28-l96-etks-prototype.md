## 2026-09-28: L96 ETKS prototype results; ETKS plan revised

**Summary:** First measurement of the ETKS against the benchmark ETKF, on all 200 P1 test windows (regular grid, benchmark inflation), from a throwaway prototype.
- S0: RMSE −17.7%, CRPS −14.5% (`correct`, L = 2).
- S1: RMSE −6.1%, CRPS −2.3% (`correct`, L = 1).
- The smoother is better in every window. Almost all of the gain comes from the next analysis.
- The ensemble RTS is never better, and it diverges in S0 without heavy truncation of P_f⁺.
- The S0 calibration identity fails at λ = 1.5 (over-dispersed filter).
**Files modified:**
- `docs/results/l96_etks_prototype.md` — new RESULTS note.
- `docs/plans/tech/l96_etks_smoother.md` — "Prototype outcome" section. `correct` is the default, `none` is limited to L ≤ 2, `inflate` and the RTS (PR 1b) are dropped. The plan now uses the pre-inflation state at analysis times, and PR 2 scores with the benchmark metric and tunes λ on validation first. Tests 9–10 revised.
- `docs/README.md` — index rows for the plan and the note.
**Rationale:** Settle the smoother design on L96 evidence before writing PR 1. No benchmark or report table is changed. The prototype is scratch code, it uses a different RMSE aggregation (per-window RMS vs the benchmark's mean per-variable RMSE), and it has no validation tuning.
**Verification:** `pytest tests/test_docs_layout.py -q`. Prototype scripts and per-window scores are archived under `experiments/l96/etks_prototype_2026-09-28/` in the main checkout (not tracked).
