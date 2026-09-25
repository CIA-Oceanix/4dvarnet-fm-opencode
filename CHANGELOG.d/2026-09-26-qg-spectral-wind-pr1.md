## 2026-09-26: Spectral (Option B) wind forcing for QG — PR-1: Fourier basis, gyrostat driver, first simulations

**Summary:** PR-1 of the Option B wind-forcing plan. It adds:
- **`models/qg_wind_modes.py`:** `FourierWindBasis` (spectral and grid curl,
  projection, translation, and the analytic Mexican-hat spectrum) and
  `generate_spectral_wind` for the drivers `spectral_ou`, `gyrostat` and
  `gyrostat_surrogate`;
- **`models/gyrostat_driver.py`:** `GyrostatSystem`, the presets `l63` and
  `l63ring4` with measured attractor constants, a Lyapunov estimate, and the
  multivariate phase-randomized surrogate;
- **`reports/qg/simulate_qg_spectral_wind.py`:** four 120-day QG simulations
  (current storm, spectral OU, gyrostat, surrogate) from one shared
  spin-up, with animations, forcing and PV figures, a comparison figure, and
  `reports/qg/outputs/qg_spectral_wind_report.md`.

No QG dynamics, data or DA code is changed. The simulations use a local
`QGDynamics` subclass until PR-2 adds the hooks.

**Files modified:**
- `models/qg_wind_modes.py`, `models/gyrostat_driver.py` — new;
- `tests/test_qg_wind_modes.py`, `tests/test_gyrostat_driver.py` — new, 28
  tests (4 marked slow);
- `reports/qg/simulate_qg_spectral_wind.py` — new;
- `reports/qg/outputs/qg_spectral_wind_report.md` and
  `reports/qg/outputs/figs/qg_specwind_*` — generated;
- `docs/results/qg_spectral_wind_calibration.md` — new;
- `reports/README.md` — index row (CURRENT, illustrative).

**Rationale:** implements the user-approved Option B plan with its default
decisions: `|k| <= 2`, the storm's spatial spectrum, the 15-day memory for
E1, and the imposed drift. Findings along the way:
- **The planned 13-mode gyrostat chain collapses to a fixed point**, and a
  fan of pairs sharing one mode synchronizes. They are replaced by
  `l63ring4`: four Lorenz-63 gyrostats on a ring with energy-conserving
  coupling (Lyapunov exponent 1.25, 12 modes, one per amplitude).
- **The Mexican-hat storm's Fourier spectrum is analytic**
  (`π σ⁴ k² e^{-σ²k²/2}`), which gives the drivers' target spectrum exactly.
- **On 40-year series all drivers match the storm's memory** (14.8–15.8
  days) and spectrum (within 3%). The gyrostat amplitudes are bimodal
  (kurtosis 1.98) against a Gaussian surrogate (2.93), and they carry a
  long-memory tail absent from OU.
- **The demo gives the current storm a 60-day wind lead-in**, because
  `generate_wind_state` starts the OU amplitude at zero.

**Verification:**
- `pytest tests/test_qg_wind_modes.py tests/test_gyrostat_driver.py`:
  28 passed, slow tests included.
- `ruff check` on the new files: clean.
- Fast suite: see the PR.
