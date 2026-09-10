## 2026-09-10: Adopt CHANGELOG.d/ fragment files to eliminate cross-PR conflicts

**Summary:** Every PR was required to add its entry to the top of the single shared
`CHANGELOG.md`, so any two PRs open at the same time always conflicted on that same
insertion point -- hit repeatedly during today's QG PR-splitting session (3 rebases,
each needing a manual 3-line conflict resolution). Replaced with the standard
changelog-fragments pattern (as used by e.g. Python's `towncrier`): each PR now adds
its own new file to `CHANGELOG.d/` instead -- new files never conflict with each other
in git. `scripts/assemble_changelog.py` folds pending fragments into `CHANGELOG.md`
(newest first, above an `AUTO-ASSEMBLED` marker) as a separate, occasional maintenance
step, not part of landing any individual PR.
**Files modified:** `CHANGELOG.d/README.md` (new) — convention; `scripts/
assemble_changelog.py` (new) — assembly script; `CHANGELOG.md` — added the
`AUTO-ASSEMBLED` marker after the header (existing entries otherwise untouched);
`AGENTS.md` — changelog convention now points at `CHANGELOG.d/` instead of direct edits.
**Rationale:** eliminates the conflict class entirely rather than just making each
occurrence cheaper to resolve.
**Verification:** dry-run tested `assemble_changelog.py` with a throwaway fragment --
folded correctly above the marker, fragment consumed, rest of `CHANGELOG.md` preserved
byte-for-byte; reverted the dry run before committing.
