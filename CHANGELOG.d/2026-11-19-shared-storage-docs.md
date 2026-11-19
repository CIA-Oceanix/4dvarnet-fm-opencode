## 2026-11-19: Document the shared Odyssey artifact folder

**Summary:** The repo docs now point contributors to `/Odyssey/public/4dvarnet-fm/` for checkpoints, configs, baselines and report bundles that other people need. That folder's own README (outside git) is the reference for layout and rules, and it gained an "L63 specifics" section.
**Files modified:** `README.md` — new "Shared storage" section; `docs/CONTRIBUTING.md` — pointer in "Things that will surprise you"; `AGENTS.md` — pointer in Hygiene
**Rationale:** The shared folder has existed since 2026-10-06, but nothing in the repo mentioned it, so new contributors (e.g. the L63 work from #356) had no way to find where to archive results.
**Verification:** `pytest tests/test_docs_layout.py tests/test_reports_index.py` (docs-only change).
