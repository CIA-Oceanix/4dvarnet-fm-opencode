## 2026-09-24: Ensemble member dumps default to node-local /tmp

**Summary:** The six L96 eval scripts that write `members_<case>.npz` now write it
under the GPU node's local /tmp by default (`/tmp/$USER/members_$SLURM_JOB_ID/<abs
output dir>`). `--keep-members` or `FDV_KEEP_MEMBERS=1` keeps the old location next
to `--output`. The sbatch scripts that produce report-read benchmark rows (P1,
benchmark-default and consolidated) export `FDV_KEEP_MEMBERS=1`.

**Files modified:**
- `evaluation/members_store.py` — new; decides where the dump goes.
- `eval_neural_l96.py`, `eval_sda_l96.py`, `eval_sda_mean_hybrid_l96.py`,
  `eval_sda_fdv1_hybrid_l96.py`, `eval_sda_directunet_hybrid_l96.py`,
  `eval_fdv1_fdv1cfm_hybrid_l96.py` — go through `members_store`; new
  `--keep-members` flag.
- 15 `batch/*.sbatch` scripts — `FDV_KEEP_MEMBERS=1` for benchmark rows. In
  `run_l96b_seeds.sbatch` this applies only to the regular test set; the
  `rand30_fast8` diagnostic goes to /tmp.
- `tests/test_members_store.py` — new.
- `docs/archive.md` — notes that members are not archived by default.

**Rationale:** On 2026-09-24 the 30 TB /Odyssey volume was full (131 GB free).
Members dumps were about 475 GB of this project's ~490 GB, at 1.6–3.3 GB per
case, and 91 GB of them were written on 2026-09-22 alone. The user deleted
194 GB of dumps that no report reads (sweeps, diagnostics, pre-monai rows) and
set the rule that sweeps and diagnostics do not store full ensembles on
/Odyssey. Metrics are unaffected: every eval script computes them in-process
from the in-memory members before the dump is written.

**Interplay with #258's `--members-file`:**
- `--members-file scores` (KB-sized per-window scores) still writes next to
  `--output`.
- `--members-file full` (the default) goes through `members_store.members_dir`,
  so it lands on node-local /tmp unless kept.
- #258's new sweep scripts already pass `scores` and need no flag.

**Caveats:**
- Node-local /tmp is not visible from other nodes and may not outlive the
  job, so any re-reading of the members, e.g. by
  `reports/l96/probe_psi_from_members.py`, must run in the same job.
- The sidecar's `<case>_members` path now points at the node-local file.
- New benchmark sbatch scripts must export `FDV_KEEP_MEMBERS=1` explicitly.

**Verification:** `pytest tests/test_members_store.py` plus the fast suite; `ruff
check` on the touched Python files; `bash -n` on every `batch/*.sbatch`.
