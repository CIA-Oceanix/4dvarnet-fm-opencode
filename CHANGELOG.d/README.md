# CHANGELOG.d/

Add one new file here for every merged change, instead of editing
`CHANGELOG.md` directly. Editing a single shared file's top section is
exactly what makes every concurrently-open PR conflict with every other one
(each adds its entry at the same spot). New files never conflict with each
other in git, so this removes that conflict class entirely.

## Convention

- Filename: `YYYY-MM-DD-<short-slug>.md` (e.g. `2026-09-10-qg-cross-res-fix.md`).
- Content: exactly one changelog entry, same format `CHANGELOG.md` has always used:

```
## YYYY-MM-DD: Short Title

**Summary:** 1-2 sentence description of changes.
**Files modified:** `path/to/file.py` — brief note
**Rationale:** Why this change was made.
**Verification:** Test command run and result.
```

Add the fragment file as part of the same PR that makes the change (same as
the old convention of editing `CHANGELOG.md` directly), just in this
directory instead of that file.

## Assembling into CHANGELOG.md

Run `python scripts/assemble_changelog.py` to fold every fragment here into
`CHANGELOG.md` (newest first, above the `AUTO-ASSEMBLED` marker) and delete
the consumed fragment files. This is a separate, occasional maintenance
step — not part of landing any individual PR — so run it whenever
convenient (e.g. after a batch of PRs has merged), not on every single one.
