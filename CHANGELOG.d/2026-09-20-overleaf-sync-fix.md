## 2026-09-20: Fix the Overleaf sync script against a real Overleaf project; set the paper's author

**Summary:** The `sync-overleaf.sh` shipped in #233 does not work against
Overleaf. Two facts, both established by running it against a live project,
break it — and both break the obvious `git subtree push` approach generally.

1. **The Overleaf branch is `main`, not `master`.** The script hardcoded
   `master`, so `push` and `pull` both targeted a branch that does not exist.
2. **Overleaf refuses force pushes, server-side:** `remote: hint: You can't git
   push --force to a Overleaf project.`

(2) is fatal to `git subtree push`. A subtree split shares no ancestor with
whatever Overleaf already has — a fresh project is not empty, it ships with a
starter `main.tex` — so the split can only land via a force, which is rejected,
or a merge Overleaf has no way to perform.

**Files modified:**
- `docs/papers/p1_structural_hypotheses/sync-overleaf.sh` — `push` now takes the
  split's **tree**, commits it with Overleaf's current head as parent, and
  pushes that: an ordinary fast-forward, always legal, and it survives Overleaf
  autosaves landing between syncs. It short-circuits when the trees already
  match. `pull` checks the remote tree into the prefix instead of `git subtree
  pull`, which would merge Overleaf's per-keystroke autosave history into this
  repo — we want its content, not its log — and leaves the result staged for
  review. `status` gained a token check and a drift indicator.
- `scripts/overleaf-askpass.sh` — new `GIT_ASKPASS` helper reading a `0600`
  token file (`~/.config/overleaf-token`, overridable via
  `OVERLEAF_TOKEN_FILE`), mirroring the existing
  `~/.config/opencode/reviewer-token` convention. The token never reaches a
  command line, shell history, or `.git/config`. It lives under `scripts/`
  rather than beside the paper because `sync-overleaf.sh` refuses to run with
  uncommitted changes under the paper prefix — a helper stored there could not
  be edited and used in the same breath.
- `docs/papers/p1_structural_hypotheses/main.tex` — author set to Ronan Fablet;
  affiliation and co-authors remain a visible `\todo` in the rendered PDF.
- `README.md` — Option A rewritten around the token file and the two quirks.

**Why a credential prompt was not an option.** Running the push from the agent
session failed with `could not read Username ... ENXIO`: there is no controlling
terminal, so `credential.helper cache` never gets populated and any interactive
`read` fails too. A file-based `GIT_ASKPASS` is the only route that works
non-interactively without embedding the secret somewhere durable.

**Two further bugs, found only by finally running the thing (2026-09-21).**

1. **`pull` was destructive.** It did `rm -rf "$PREFIX"` before re-populating
   from the remote tree. The `rm` failed -- "directory not empty", this repo is
   on NFS where an open file leaves a `.nfs*` entry -- `set -e` aborted, and the
   paper directory was left **gutted with nothing restored**; all 12 files were
   recovered with `git reset --hard`. The rm was never needed: the index can be
   repointed at the remote tree and checked out without deleting anything first.
   Fixed, and the fix immediately proved itself -- a later failure mid-`pull`
   left the worktree completely intact.
2. **The sync was unscoped, so it mirrored `sync-overleaf.sh` itself.** That
   script lives inside `$PREFIX`, so a `pull` would overwrite it with whatever
   Overleaf held: it clobbered itself. `README.md` and `.gitignore` were also
   being pushed into a LaTeX project for no reason.

Both are fixed by `SYNC_PATHS=(main.tex refs.bib sections)`: `push` mirrors only
those, `pull` only writes or deletes within them. That also removes `git subtree`
entirely -- `HEAD:$PREFIX` is already a tree rooted at the paper directory, so
filtering it is enough; no split, no synthetic history to reconcile.

A third, smaller one: `mktemp` creates a 0-byte file and git rejects that as an
index ("index file smaller than expected"), so the scratch index needs
`mktemp -u`.

**Verification — end to end against the live Overleaf project.**

| test | result |
|---|---|
| `init`, `status` | remote, token and drift reported correctly |
| `push` | `feacffd..429de3f`, then `94950bc..01c6e45` |
| `push` again | short-circuits: "already matches ... nothing to push" |
| `pull`, trees equal | short-circuits: "nothing to pull" |
| `pull`, trees differ | applied an Overleaf-side edit to `main.tex`; **`sync-overleaf.sh`, `README.md`, `.gitignore` untouched** |
| remote contents | exactly the 9 LaTeX files -- no script, no README, no `.gitignore` |
| empty token file | `status` now reports `(EMPTY -- auth will fail)` rather than "present" |

The Overleaf project and the repo are reconciled: `drift: none`. `bash -n` clean
on both scripts; `pdflatex` still gives 11 pages.
