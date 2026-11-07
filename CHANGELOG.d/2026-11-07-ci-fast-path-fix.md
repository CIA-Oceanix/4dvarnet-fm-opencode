## 2026-11-07: CI docs-only fast path — install pyyaml, skip conftest

**Summary:** The docs-only fast path of `.github/workflows/ci.yml` installs `pyyaml` with pytest and runs the docs tests with `--noconftest`.
**Files modified:** `.github/workflows/ci.yml`.
**Rationale:** Its first use (#371) failed: `tests/conftest.py` imports torch, and `tests/test_reports_index.py` needs PyYAML. The docs tests use no conftest fixture.
**Verification:** in a clean venv with pytest and pyyaml only, `pytest --noconftest tests/test_docs_layout.py tests/test_reports_index.py` passes (116 tests).
