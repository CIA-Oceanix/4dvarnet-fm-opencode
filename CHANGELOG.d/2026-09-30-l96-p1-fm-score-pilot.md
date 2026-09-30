## 2026-09-30: L96 P1 flow-matching operator score (pilot)

**Summary:** New proper score FMS_τ = E‖E_q[x₁|x_τ] − x₁*‖² (Gaussian closed form, empirical/KDE ensemble operator, snapshot blocks) and a pilot on the P1 rows with stored members (400-epoch CFM/SDA) plus ETKF/ETKS (Gaussian) and DirectUNet-M at τ ∈ {0, 0.25, 0.5, 0.75, 0.95}, with a variance-rescaling calibration-loss decomposition.
**Files modified:** `evaluation/fm_score.py` — score functions; `tests/test_fm_score.py` — Gaussian optimum at v = MSE, closed-form/MC agreement, propriety check; `reports/l96/probe_fm_score.py`, `reports/l96/probe_fm_score_scale.py` — probes; `reports/l96/outputs/fm_score/` — outputs; `docs/results/l96_p1_fm_score_pilot.md` — results note.
**Rationale:** P1 needs a proper score that separates mean accuracy (τ = 0) from calibration (τ > 0); DA calibration loss grows S0 → S1 while learned rows' is invariant.
**Verification:** `pytest tests/test_fm_score.py` (9 passed); `ruff check` on touched files clean; benchmark RMSEs reproduced (ETKF 0.610/1.409, ETKS 0.497/1.338).
