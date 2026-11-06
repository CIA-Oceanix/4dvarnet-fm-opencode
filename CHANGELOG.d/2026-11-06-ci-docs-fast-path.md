## 2026-11-06: CI — docs-only fast path

**Summary:** `.github/workflows/ci.yml`: the ruff lint becomes a step of the required `pytest` job, and a docs-only change (every file `*.md`, `*.tex`, `*.bib`, under `docs/`, or an md/png/pdf/gif under `reports/*/outputs/`) installs pytest only and runs the docs tests. `changelog fragments policy` stays a separate job, since the `master` branch protection requires it by name. AGENTS.md documents the fast path, explains why the manual `open_pr.sh review` approval stays required (unattributed commits need an extra approval under the `master` ruleset), and exempts pure paper syncs from the changelog fragment.
**Files modified:** `.github/workflows/ci.yml` — ruff folded into `pytest`, change classification, docs fast path, concurrency cancel, pip cache; `AGENTS.md` — docs-only rules, why the manual approval stays.
**Rationale:** Text-only PRs (paper syncs) waited 8-11 min for the full suite and ran three billed jobs (now two). Modelled on CIA-Oceanix/da-error-budget's CI.
**Verification:** YAML parses; classifier checked on representative paths; this PR (workflow change) runs the full suite; the next docs-only PR exercises the fast path.
