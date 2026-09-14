## 2026-09-10: Enforce the CHANGELOG.d/ policy in CI (not just documentation)

**Summary:** The CHANGELOG.d/ fragment convention (see the same-day "Adopt CHANGELOG.d/
fragment files" entry) was only documented in `AGENTS.md`, not enforced -- an agent or
human could still hand-edit `CHANGELOG.md` directly by mistake, silently reintroducing
the cross-PR conflict it exists to prevent. Also, `master` currently has no GitHub
branch protection at all, so even the existing `pytest` merge gate is purely a
documented convention every agent chooses to follow, not something GitHub itself
blocks. Added a new blocking CI job (`changelog-policy`) that fails the build if
`CHANGELOG.md` is modified without a paired `CHANGELOG.d/*.md` deletion in the same
diff (the signature of a legitimate `scripts/assemble_changelog.py` run) -- with a
one-time bootstrap exception for the PR that introduces the `AUTO-ASSEMBLED` marker
itself (this PR).
**Files modified:** `.github/workflows/ci.yml` — new `changelog-policy` job.
**Rationale:** a convention that's only written down gets violated eventually;
turning it into a real, automated check is what actually makes it mandatory.
**Verification:** validated the workflow YAML parses; simulated the check's bash logic
locally against this branch's own diff (correctly identified as the bootstrap case,
would pass) -- the check will run for real once this PR's own CI executes.
