## 2026-09-10: QG — revised S1 (model-error) reference case, full ETKF/EnKF N=100 campaign

**Summary:** Redefined S1 to share S0's initial-uncertainty setup exactly (lag=5.0d,
noise_frac=0.05) with three model-error sources on top (param bias, corrupted wind,
`da_nx=32` structural resolution mismatch). Bias level (0.1/0.1, not the 0.15/0.15
project default) chosen via an isolated per-factor sensitivity sweep: wind bias is
nearly harmless to the DA analysis even at 0.30, param bias is dangerous (ETKF
diverges outright at 0.30). An ETKF inflation sweep testing whether under-inflation
explains EnKF's edge over ETKF found the opposite — any inflation above 1.0 causes
immediate catastrophic divergence, not gradual degradation; `inflation=1.0` looks
necessary for stability here, not undertuned. Full N=100 ETKF/EnKF results committed;
EnKF is the clear S1 winner (stays positive on q; both 4DVar methods collapse harder
under S1 than S0). See PLAN.md's "Revised S1 (model-error) reference case" section for
the full writeup, config, and results table.
**Files modified:** `reports/qg/outputs/qg_repro_validation_s1/{etkf,enkf}.json` (new,
N=100) + `etkf_n10*.json` (new, N=10 sanity check + inflation sweep);
`reports/qg/outputs/qg_repro_validation_s1_sensitivity/*.json` (new, per-factor
sensitivity sweep); `PLAN.md`.
**Rationale:** the old S1 (project defaults, 0.15/0.15 bias) was never actually run at
N=100 with the revised lag/noise setup; this establishes it as a real, validated,
non-degenerate reference case alongside S0.
**Verification:** all committed JSONs are real `run()` output from
`qg_da_s1_scratch.py`/`qg_da_s1_factor_sensitivity_scratch.py` (scratch drivers, not
committed) on the production 100-window test cache.
