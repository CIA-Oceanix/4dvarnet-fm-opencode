# Verification scripts

Standalone, manually-run verification scripts. **These are not pytest tests.**

They previously lived under `tests/` with `test_*.py` names, which made pytest
try to collect them. None of them defines a test function -- all of their work
happens at module import time -- so collection executed the whole script and
crashed, which in turn made `pytest tests/` fail outright. They were moved here
so that `pytest tests/ -m "not slow"` is a clean, runnable merge gate.

| script | what it checks | requirements |
|---|---|---|
| `check_numerical_equivalence.py` | `Lorenz63Dynamics.step` / `EnKF` / `ETKF` / 4DVar against the pre-refactor inline L63 code | CPU or GPU |
| `check_equiv_report.py` | current L63 S0/S1 baseline RMSEs against the published `s0_s1_synthesis.md` values | **GPU required**, runs a full DA suite (minutes) |
| `compare_rmse_slices.py` | full-window vs edge-trimmed RMSE | CPU |

Run them directly, not through pytest:

```bash
python scripts/verification/check_numerical_equivalence.py
```

They print `PASS`/`FAIL` lines rather than asserting. The automated equivalence
coverage lives in `tests/test_refactoring_equivalence.py`.
