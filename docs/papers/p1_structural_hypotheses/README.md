# P1 draft — "The structural hypotheses that bound data assimilation"

First draft of the DA-venue diagnosis paper. Target: **JAMES**.

**Source of truth is `docs/scoping/da_paper_structural_hypotheses.md` (SCOPING
v2), not this directory.** The draft renders that document's argument; it does
not supersede it. If a number changes in the scoping doc or the underlying
report, change it here too.

## Build

```bash
cd docs/papers/p1_structural_hypotheses
latexmk -pdf main.tex        # or: pdflatex main.tex  (x2)
```

Compiles standalone with the `article` class — 11 pages as of the first draft,
no undefined references.

## Getting it into Overleaf

The wrinkle: this paper lives in a **subdirectory** of a large repo, but
Overleaf needs `main.tex` at the **project root**. `git subtree` bridges that —
it rewrites this directory's history as though it were its own repo. Verified:
a subtree split of this prefix yields `main.tex`, `refs.bib`, `sections/`,
`README.md` at top level, which is exactly what Overleaf expects.

### Option A — two-way git sync (recommended; needs a paid Overleaf plan)

Overleaf's git bridge is **not on the free tier**. If you have it:

1. In Overleaf, create a **blank project** (don't upload anything yet).
2. *Menu → Git* → copy the URL, `https://git.overleaf.com/<project-id>`.
3. Get a git token: *Account Settings → Git integration*. It is used as the
   **password**; any username works.
4. From anywhere in this repo:

```bash
docs/papers/p1_structural_hypotheses/sync-overleaf.sh init https://git.overleaf.com/<project-id>
docs/papers/p1_structural_hypotheses/sync-overleaf.sh push     # repo  -> Overleaf
docs/papers/p1_structural_hypotheses/sync-overleaf.sh pull     # Overleaf -> repo
```

Edit in Overleaf, `pull` to bring changes back; edit here, `push` to send them.
`pull` uses `--squash`, so one merge commit per sync rather than every Overleaf
autosave.

**Do not put the token in the URL** — it would land in `.git/config` in
plaintext. Let git prompt for it and cache it:

```bash
git config --global credential.helper 'cache --timeout=86400'
```

### Option B — zip upload (free tier; one-way)

```bash
cd docs/papers/p1_structural_hypotheses
zip -r ../p1.zip . -x '.git*' -x '*.aux' -x '*.log' -x '*.pdf'
```

Then Overleaf *New Project → Upload Project*. To update, re-upload — which
**discards Overleaf-side edits**, so only use this if Overleaf is a read-only
view or you are the sole editor.

### Option C — copy-paste

`main.tex` plus `sections/` into a blank project. Fine for a one-off look;
no sync.

### Which to choose

If co-authors will edit in Overleaf, you need **A** — otherwise their edits
have no route back and will be overwritten. If Overleaf is just for rendering a
PDF to circulate, **B** is enough.

To switch to the AGU/JAMES template, replace the `\documentclass` line in
`main.tex` with `\documentclass{agujournal2019}` (Overleaf has the template
built in) and move `\title`/`\author` into AGU's macros. Everything else is
portable — no custom class dependencies.

## Drafting macros

Three markers flag what is not yet settled. Grep for them before circulating:

| macro | meaning |
|---|---|
| `\todo{...}` | work to do before submission |
| `\needsrun{...}` | a claim **blocked on an experiment not yet run** |
| `\caveat{...}` | evidenced, but carries a stated limitation |

`\needsrun` is the important one. The draft is written so that every claim
resting on an unrun experiment says so in the rendered PDF, in red. Do not
circulate externally until those are resolved or removed.

## Known gaps in this draft

- **C1 is blocked on D0** (weak-constraint 4D-Var benchmark). The
  implementation landed in #229 but is wired to no driver and its `optimizer`
  still defaults to `"adam"`, which the investigation measured at RMSE ~24
  against L-BFGS's ~0.98. Until D0 runs, C1 is argued against strong-constraint
  and filtering methods only.
- **`refs.bib` is entirely unverified** — written from memory, every entry
  marked `UNVERIFIED`. Nothing in it has been checked against the actual
  publication, and no `\cite` commands are used in the text yet.
- **No figures.** All evidence is currently in tables.
- **Rank histograms (D5) do not exist** anywhere in the codebase; they are the
  primary posterior-diagnostic figure for this readership.
