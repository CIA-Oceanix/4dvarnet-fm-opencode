## 2026-09-18: L96 — make the consolidated benchmark reproducible from stored checkpoints

**Summary:** `reports/l96/generate_l96_consolidated_report.py` could not run from
the master worktree at all, and had drifted 5 PRs behind the markdown it
produces. Both are fixed: artifact location is now resolved by one module, the
scattered arrays are consolidated, and every published method has a generator
path again.

**Files modified:**
- `evaluation/archive.py` — new; the single resolver for run artifacts
- `scripts/consolidate_l96_archive.py` — new; `--check` audit / `--apply` hard-link consolidation
- `tests/test_archive.py` — new; 19 tests on a synthetic tree (no dependency on gitignored data)
- `docs/archive.md` — new; layout, conventions, the parent-walking anti-pattern
- `reports/l96/generate_l96_consolidated_report.py` — resolve by run name; drop 16 pre-monai methods; add 7 monai-era ones; declare `ESTIMATE_FILENAMES` and `UNAVAILABLE_METHODS`
- `eval_monai_l96.py` — norm stats resolved from config, not path arithmetic

**Rationale:** three independent failures, all of the same shape — a convention
assumed in several places instead of declared in one.

1. *Artifacts unreachable.* The 2026-09-10 consolidation moved `checkpoints/` +
   configs into `experiments/l96/` but left the ~100 MB `estimates_*.npz` in
   whichever worktree trained each run. 24 of 34 methods resolved to paths that
   do not exist in master, so the report died on its first row. 62 artifacts
   were hard-linked in from four sibling worktrees (same filesystem: no extra
   disk, and unlike symlinks a hard link survives worktree pruning and cannot be
   walked off by `Path.resolve()`).

2. *Path arithmetic.* `eval_monai_l96.py` derived the normalization stats as
   `Path(ckpt).resolve().parents[2] / "l96_norm_stats_obsj2.pt"`. Correct for
   `experiments/<run>/checkpoints/`, silently wrong for every archived
   checkpoint once `experiments/l96/` added a level.

3. *Generator drift.* PRs #188/#192/#194/#200/#204 added rows to
   `l96_consolidated_benchmark.md` by hand without touching the generator, which
   `docs/scoping/README.md` explicitly forbids. Regenerating therefore dropped 7
   methods, including the table's headline result. Each of those PRs documented
   why: the generator's full path needed caches "only available in the training
   worktree" — the very problem item 1 removes.

**Scope decision (2026-09-17, user):** the 16 pre-monai neural methods
(`L1b_direct_unet_s0s1`, `L2b`/`L3`/`L4`/`L5`/`L6`, `V2_tweedie_cfm_l96`,
`V3_predict_state_cfm_l96`, `SDA1_prior_l96`, `SDA2_cond_*`,
`FDV1_unrolled_unet_l96`, `FDV1CFM_predict_state_l96`, `FDV1_SDA*_hybrid_l96`)
are superseded and no longer need reproducing. The CFM arm remains represented
by 4 monai rows. DA baselines are unchanged.

**Verification:** every ported method re-scores to its published value.

| method | published S0 | re-scored |
|---|---|---|
| subgrad+state-Stier(monai) | 0.3728 | 0.372765 |
| DirectUNet(aug)+SDA3 | 0.3890 | 0.3890 |
| FDV1-Stier(monai) | 0.4011 | 0.401114 |
| DirectUNet-M(monai,cos,obsdensity) | 0.4727 | 0.472707 |
| CFM-M(monai,flat,obsdensity) | 0.4654 | 0.465437 |
| FDV1-Stier+SDA3(monai) | 0.3587 | 0.3588 (re-run, ens30 stochastic) |

`pytest tests/test_archive.py tests/test_l96_report_consistency.py` -> 30 passed.

**Regenerated table vs published**, 16 pre-monai rows removed as agreed and no
row lost unexpectedly:

- 44 cells shift by <=0.0011 -- the hand-added rows were computed in float32 via
  direct `evaluate_estimates` calls, the generator casts to float64.
- 18 cells on **`DirectUNet-M(monai,flat)`** change substantively (slow 0.2671
  -> 0.2720). That row's group columns were transcribed from its
  `neural_eval.json` sidecar, which disagrees with its own stored arrays;
  regeneration corrects it. It is the only such row -- every other pre-existing
  monai row reproduces exactly.

**Generator bugs found and fixed while porting** (each silently degraded a
result rather than failing):

- `L3_ENS30_DIR`, a module-level constant that was never read, still crashed
  import once its `ENS30_DIRS` entry was dropped.
- `load_members` and `_ens30_es` built `experiments/<run>/...` paths directly,
  so an archived row's members were not found and the caller fell back to the
  deterministic CRPS proxy.
- there was **no path to compute pooled ensemble ES from `members_*.npz`** --
  only from the DA cache or an ens30 `neural_eval.json` -- so a hybrid evaluated
  with `--n-members 30` silently reported the N=1 MAE proxy (0.2077 vs the
  proper 0.1708). This is the structural reason the two Stier+SDA3 rows had to
  be computed by hand.
- `MONAI_ROWS` is a second method list driving the per-window tables; methods
  added only to `NEURAL_EXP_DIRS` appeared in 3 of the 6 tables.
- `load_members` promoted a 1.6 GB (W,T,D,M) array to float64, doubling it and
  doubling again in the ES accuracy term; kept in float32, which is also what
  produced the published numbers.

`pytest tests/test_archive.py` → 19 passed.
