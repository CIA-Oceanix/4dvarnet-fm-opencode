## 2026-10-15: QG neural eval sbatch scores one shard of the windows

**Summary:** `batch/run_qg_specwind_neural_eval.sbatch` takes `SHARD`/`N_SHARDS` (default 0/1) and passes them to `evaluation.run_qg_specwind_neural --shard/--n-shards`, writing `shard_<SHARD>_of_<N_SHARDS>.json` like the DA rows.
**Files modified:** `batch/run_qg_specwind_neural_eval.sbatch` — shard env vars and output name.
**Rationale:** the published-SDA test row (3D prior, 100 steps + 1 corrector, N=30) costs about 40 GPU-hours on 100 windows, too long for one job; the Python driver already shards.
**Verification:** `bash -n` on the script; test row submitted as 8 shards (61664–61671). Default output path unchanged (`shard_0_of_1`).
