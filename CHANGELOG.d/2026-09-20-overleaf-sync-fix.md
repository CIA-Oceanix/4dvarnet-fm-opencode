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

**Verification — read this carefully, it is partial.**
- `status` and `init`: exercised against the live project; `status` correctly
  reported the remote, the token file, and drift.
- The **push mechanism** is verified: running its exact commands by hand
  (`subtree split` → `rev-parse ^{tree}` → `commit-tree -p overleaf/main` →
  `push`) is how the 11-page draft reached the Overleaf project, confirmed by
  diffing the remote's `main.tex` against the repo's (identical) and listing all
  11 source files on the remote.
- **`push` and `pull` as implemented in the script are NOT verified end to end.**
  The token was rotated mid-test — correctly, because it had been pasted into a
  chat transcript — and the replacement was not available before this landed.
  The failure mode observed afterwards is an *auth rejection*, not a prompt
  failure, which does confirm the askpass path reaches Overleaf and presents a
  credential. **Someone should run `push` and `pull` once with a valid token
  before relying on them.**
- `bash -n` clean on both scripts; `pdflatex` still gives 11 pages with the new
  author line.
