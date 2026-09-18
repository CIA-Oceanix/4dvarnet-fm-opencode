## 2026-09-18: L96 — one scoring formula per column

**Summary:** the ES / CRPS columns mixed a proper ensemble score with an N=1 MAE
proxy, with only a trailing `*` to distinguish them. The proxy is removed. Each
cell is now either a real ensemble score, `—` (not defined for a deterministic
point estimator), or `pending` (ensemble-capable, members not yet produced).

**Files modified:**
- `reports/l96/generate_l96_consolidated_report.py` — `DETERMINISTIC_METHODS`; three-state ES/CRPS; `N1_ES_METHODS` and the `per_window_deterministic_crps` fallback deleted
- `tests/test_l96_report_consistency.py` — assert the taxonomy is declared and the proxy cannot return
- `reports/l96/outputs/l96_consolidated_benchmark.md` — regenerated

**Rationale:** a proper ensemble score credits spread; the N=1 proxy cannot.
Putting both in one column made rows non-comparable, and specifically made the
only two rows with real members look better partly by convention rather than by
skill — the exact apples-to-apples failure this table exists to prevent.

Scheme class comes from `model_type` in each run's resolved config, not from
name matching: `monai_direct_unet` and `fourdvarnet` are deterministic;
`monai_vanilla_cfm` and `monai_sda_*` are stochastic.

**Strong-4DVar also carried a proxy.** Its ES came from the DA cache and looked
like a peer of ETKF/EnKF's, but `_ESAccumulator` is fed
`analysis[t].reshape(1, -1)` — a single-member "ensemble" — so it was the N=1
proxy all along. `DETERMINISTIC_METHODS` is therefore tested *before*
`DA_METHODS`, so DA membership cannot override the taxonomy. ETKF/EnKF are
genuine N=30 ensembles and keep their scores.

**Resulting ES/CRPS column:** 4 real values (ETKF, EnKF, and the two
Stier+SDA3 hybrids), 10 `—`, 14 `pending`.

RMSE and EV per-window mean±std were already complete and homogeneous for all 27
scoring schemes and are unchanged.

**Known limitation:** the table currently has no probabilistic metric that
compares neural schemes with one another — the only real ensemble scores belong
to 2 DA methods and 2 hybrids. Resolving the 14 `pending` rows (re-evaluate with
`--n-members 30 --n-outer 10`, ~5 GPU-hours) is the second step. ETKF/EnKF
per-window CRPS additionally needs the DA driver to persist members, which is a
code change rather than a re-run.

**Verification:** 32 tests pass; report regenerates clean.
