## 2026-09-17: Fill the five conditioning-ladder rows in the L96 consolidated benchmark

**Summary:** `L2b`, `L6`, `SDA1`, `SDA2-mixed` and `SDA2-nominal` were `—`
placeholders in all three variable-group tables despite their evaluations
existing on disk. Filled from those evaluations. Numbers only; no other row,
section or file touched.

**Files modified:** `reports/l96/outputs/l96_consolidated_benchmark.md` — 15
rows (5 labels × RMSE / EV / ES by variable group).

**Rationale:** These five carry §4.1.1 and R1 of
`docs/scoping/da_paper_structural_hypotheses.md`. They are the conditioning
ladder: `L2b` (no conditioning) vs `L6` (corrupted-forcing conditioning) is a
matched pair on one architecture, and `SDA1` (unconditional) vs `SDA2-mixed`
(params+forcing, trained with S1-level error) vs `SDA2-nominal` (same, trained
at `forcing_state_bias=0.0`) is the ladder plus the training-exposure control.
The scoping doc cites all five; until now a reader could not check them against
the canonical table.

What they show, now visible in the table: conditioning on model information
moves S0 skill by at most ~1.5% and not even consistently in sign
(L2b 0.6329 → L6 0.6389, +0.9% worse; SDA1 0.7185 → SDA2-mixed 0.7074, −1.5%
better). And SDA2-nominal — the only configuration never exposed to S1-level
error in training — still gives S1/S0 = 0.998 against the S1-exposed 0.997, so
training-time exposure does not explain the S1 invariance.

**Verification:** Values read programmatically from each run's
`neural_eval.json` (`ens30_no10/` for the three SDA rows), all at
`num_samples = 200`, matching the table's window count. The fill script asserts
each target line is an existing placeholder before replacing it and that each
label resolves to exactly one row per table; `git diff --stat` is 15 insertions
/ 15 deletions. `*` (N=1 ensemble) markers were already present in the `L2b`/`L6`
placeholders and match `N1_ES_METHODS` in the generator.

**Deliberately not done:** the report was **not** regenerated. Regeneration
would have dropped `FDV1-Stier(monai)`, `FDV1-Stier+SDA3(monai)`,
`subgrad+state-Stier(monai)` and `subgrad+state-Stier+SDA3(monai)` — the four
best rows in the table — because none of them is in the generator's
`NEURAL_EXP_DIRS` and the script does not produce them. The committed report is
a hybrid of generated and manually-inserted rows, i.e. it is not reproducible
from its own generator, and `batch/run_l96_consolidated_report.sbatch` will
silently lose those rows for whoever runs it. That defect is being addressed in
a separate session; this change stays out of the generator entirely to avoid
conflicting with it.

**Also noted, not fixed:** the remaining non-monai rows (`L1b`, `L3`, `L4`,
`L5`, `V2`, `V3`, `FDV1`, `FDV1CFM`, `FDV1+SDA1`, `FDV1+SDA2`,
`FDV1+FDV1CFM`) and `FDV2(monai)` are still placeholders. Their data exists
across sibling worktrees; filling them belongs with the generator fix, not here.
