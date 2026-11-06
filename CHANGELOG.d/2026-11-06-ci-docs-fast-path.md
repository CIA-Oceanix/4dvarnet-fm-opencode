## 2026-11-06: CI — one job with a docs-only fast path

**Summary:** `.github/workflows/ci.yml` runs a single `pytest` job (the only required check): the changelog policy and ruff become steps, and a docs-only change (every file `*.md`, `*.tex`, `*.bib`, under `docs/`, or an md/png/pdf/gif under `reports/*/outputs/`) installs pytest only and runs the docs tests. AGENTS.md documents the fast path, explains why the manual `open_pr.sh review` approval stays required (unattributed commits need an extra approval under the `master` ruleset), and exempts pure paper syncs from the changelog fragment.
**Files modified:** `.github/workflows/ci.yml` — consolidated job, change classification, docs fast path, concurrency cancel; `AGENTS.md` — docs-only rules, why the manual approval stays.
**Rationale:** Text-only PRs (paper syncs) waited 8-11 min for the full suite and ran three billed jobs. Modelled on CIA-Oceanix/da-error-budget's CI.
**Verification:** YAML parses; classifier checked on representative paths; this PR (workflow change) runs the full suite; the next docs-only PR exercises the fast path.
