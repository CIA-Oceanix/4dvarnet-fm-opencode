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

Three options, best first.

1. **Overleaf git sync** (needs a paid Overleaf account). Create an empty
   Overleaf project, take its git URL from *Menu → Git*, and push this
   directory's contents to it. Two-way sync afterwards.
2. **Zip upload.** `zip -r p1.zip . -x '*.git*'` from this directory, then
   Overleaf *New Project → Upload Project*. One-way; re-upload to update.
3. **Copy-paste** `main.tex` plus the `sections/` files into a blank project.

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
