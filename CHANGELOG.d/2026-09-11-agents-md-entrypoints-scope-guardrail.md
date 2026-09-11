## 2026-09-11: AGENTS.md — document top-level entry points, add scope guardrail

**Summary:** Two small additions to `AGENTS.md`, cherry-picked from a Claude-app-drafted
`docs/CLAUDE.md` proposal (pushed to `rfablet-patch-1`) after most of that draft was found
to be stale or actively wrong for this repo (fabricated Hydra CLI examples, a
`data/`-is-off-limits note that contradicts how `data/lorenz96.py`/`data/qg.py` are
actually worked on, and a worktree table that repeats `docs/worktrees.md`'s existing
staleness). Only two items survived verification against the real repo.
**Files modified:** `AGENTS.md` — Project Structure section now lists the top-level entry
points (`train.py`, `run_experiment(s).py`, per-case-study `eval_*.py`/`evaluate_all*.py`),
which were previously undocumented; Git/PR Workflow's Hygiene section gains a guardrail
against reformatting/restyling unrelated code in the same change.
**Rationale:** Close a real documentation gap (root-level scripts were never mentioned)
and make an existing scope-discipline expectation explicit, without importing the rest of
the draft's inaccuracies.
**Verification:** Doc-only change; no tests affected. Confirmed the listed entry-point
scripts exist (`ls *.py`) and confirmed the discarded parts of the draft were actually
wrong (checked `train.py`'s real `@hydra.main` config groups, `config/` contents, and
that `data/*.py` are routinely edited per `AGENTS.md`'s own "Data is generated on-the-fly"
convention).
