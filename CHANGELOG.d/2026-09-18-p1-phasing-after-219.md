## 2026-09-18: Correct P1's phasing and close its ML-split decision, after #219

**Summary:** Two statements in the DA diagnosis scoping doc were made false by
PR #219 (consolidated benchmark regenerated from stored checkpoints) and PR
#218/#220 (P2 scoping). Corrected. Docs only.

**Files modified:** `docs/scoping/da_paper_structural_hypotheses.md` — §8
phasing (P0a withdrawn, P0b renumbered to P0, note added); §10 decision 1
closed.

**Rationale:**

1. **P0a is withdrawn, not rescheduled.** It described `L2b`/`L6`/`SDA2-mixed`/
   `SDA2-nominal` as `—` placeholders in `l96_consolidated_benchmark.md` needing
   only "an evaluation-to-table pass". #219 rebuilt the report to recompute every
   row from archived trajectory arrays and scoped `NEURAL_EXP_DIRS` to the
   monai-backbone schemes plus DA baselines, so those rows were **removed, not
   left as placeholders** (`L6` and `SDA2-nominal` now have zero occurrences in
   the report). Re-adding them would mean editing a freshly-rewritten generator
   against its evident design. The doc now says to cite those five **by run
   artifact** (`neural_eval.json`, resolvable via `evaluation/archive.py`) and
   never as benchmark rows.

   An earlier recommendation in session — archive `SDA2_cond_nominal` so it
   appears in the table — is **retracted**: archiving does not add a row, since
   the row list is explicit and monai-only, and `--names SDA2_cond_nominal_l96`
   reports its artifacts already resolvable, so there is nothing to repair.

2. **A real asymmetry is now recorded rather than hidden.** `SDA2_cond_nominal`
   is the only configuration trained at `forcing_state_bias=0.0` — R1.3's
   training-exposure control — and has **no monai counterpart**. So R1.3's claim
   rests on a run the canonical table does not and will not show. The doc says to
   state that plainly, or to train a monai nominal variant if the claim needs
   more weight.

3. **§10 decision 1 closed.** It still described the mean-slot comparison as "the
   ML paper's headline"; that moved to P2 (M2) in #218/#220. The three-way split
   is recorded, with the two boundaries that must not blur (calibration; a shared
   obs-density protocol).

**Verification:** Docs only, no code touched. Claims checked against
`origin/master`: `NEURAL_EXP_DIRS` contains no non-monai entries and zero
occurrences of `SDA2_cond_nominal`; `grep -c` on the report gives 0 for both `L6`
and `SDA2-nominal`; `scripts/consolidate_l96_archive.py --names
SDA2_cond_nominal_l96` dry-run reports 3 artifacts already resolvable and 0
missing; `config/experiment/` has no monai nominal variant.
