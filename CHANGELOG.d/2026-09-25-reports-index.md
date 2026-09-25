## 2026-09-25: reports/README.md index + config-freshness check for CURRENT reports

**Summary:** Adds `reports/README.md`, the single index of the 32 checked-in
reports. Each row gives the question, a status (CURRENT / FROZEN / STALE /
SUPERSEDED), the protocol and the generator, and the index names the headline
report per case study. `tests/test_reports_index.py` enforces the index, and
fails when a CURRENT report's model configs have changed since the report was
last regenerated.

**Files modified:**
- `reports/README.md` (new). QG and L63 are marked "rework planned", with
  provisional statuses.
- `tests/test_reports_index.py` (new). It checks four things:
  - every report is indexed;
  - every indexed report exists;
  - every status is from the vocabulary, and every listed generator exists;
  - no CURRENT L96 report is older than its model configs.

  For the last check, a report's dependencies are the `config/experiment/*.yaml`
  files its generator names, plus the `config/*.yaml` paths it names, followed
  through Hydra `defaults:`. They are compared as parsed YAML against the
  version at the report's last commit, so comment-only edits pass.
- `.github/workflows/ci.yml`: the pytest job checks out with `fetch-depth: 0`,
  so the freshness check can read history. The test skips on a shallow clone.

**Rationale:** Step R1 of the docs/reports refactor. There was no entry point
to the results, four overlapping L96 benchmark reports had no marked headline,
and nothing told a reader which reports were still valid. The user also set the
rule that reports must be regenerated when their model configs change. The
check turns that rule from a convention into a CI failure.

**Verification:**
- `pytest tests/test_reports_index.py tests/test_docs_layout.py`: all pass.
- Appending a real key to `config/experiment/A1_vanillacfm_monaiM_l96.yaml`
  makes the check fail for all three CURRENT reports that use it.
- Appending a comment only does not trip it.
