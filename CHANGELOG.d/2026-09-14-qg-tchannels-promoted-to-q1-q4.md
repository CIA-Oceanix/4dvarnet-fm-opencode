## 2026-09-14: T-channels DirectUNet family promoted to canonical Q1-Q4

**Summary:** Q7-noise05/Q8/Q9/Q10 (the T-channels bench refresh's obs-only/
oracle-cond./noisy-cond./noisy-cond.+IC schemes) are promoted to be the new
canonical `Q1`/`Q2`/`Q3`/`Q4` in `reports/qg/outputs/qg_neural_report.md`'s
benchmark table, retiring the older batch-folding-backbone family
(Q1/Q3/Q4/Q3-noise0.05/Q5) from that report.

**Files modified:**
- `config/experiment/Q1_direct_unet_tchannels_s0.yaml` (was
  `Q7_direct_unet_tchannels_s0_noise05.yaml`),
  `Q2_direct_unet_tchannels_s0_oracle_cond.yaml` (was `Q8_...yaml`),
  `Q3_direct_unet_tchannels_s1_noisy_cond.yaml` (was `Q9_...yaml`),
  `Q4_direct_unet_tchannels_s1_noisy_ic_cond.yaml` (was `Q10_...yaml`) --
  `git mv` + `experiment_id`/header comment updated. The older
  `Q1_direct_unet_s0.yaml`/`Q2_vanilla_cfm_s0.yaml` (unrelated
  scheme)/`Q3_direct_unet_s0_oracle_cond*.yaml`/
  `Q4_direct_unet_s1_noisy_cond.yaml`/`Q5_direct_unet_s1_noisy_ic_cond*.yaml`
  are untouched -- kept as historical, reproducible experiments.
- `batch/run_qg_q{1,2,3,4}_direct_unet_tchannels*.sbatch` -- `git mv` +
  updated `--exp-id`/`--exp-dir`/job-name to match.
- `eval_qg_neural_s0_s1.py` -- `SCHEMES` rebuilt with just the 4 new
  entries (all 10 old entries -- Q1/Q3/Q4/Q3-noise0.05/Q5/Q7/Q7-noise05/
  Q8/Q9/Q10 -- removed); module docstring/CLI help text updated.
- `reports/qg/generate_qg_neural_report.py` -- `SCHEMES`/`NEURAL_SCHEMES`
  updated to the 4 new rows (all `matched=True` now, since all four train
  directly at the DA-matched config -- the † caveat marker no longer
  applies to any row); module docstring and table prose updated.
- `tests/test_qg_neural.py` -- 4 config-sanity tests renamed
  (`test_q7_noise05_...`/`test_q8_...`/`test_q9_...`/`test_q10_...` ->
  `test_q1_direct_unet_tchannels_s0_...`/`test_q2_..._oracle_cond_...`/
  `test_q3_..._noisy_cond_...`/`test_q4_..._noisy_ic_cond_...`) pointing
  at the new filenames. `test_q7_direct_unet_tchannels_yaml_config`
  (checks the original, untouched `Q7_direct_unet_tchannels_s0.yaml`) is
  unchanged.
- `reports/qg/outputs/qg_neural_s0_s1_cross_scenario/
  results_lag5_noise0.05_bias0.1.json`, `reports/qg/outputs/
  qg_neural_report.md` -- regenerated.

**Checkpoint archiving:** the 4 winning experiment directories were moved
to the master worktree's `experiments/qg/{Q1,Q2,Q3,Q4}/` (mirroring L96's
`experiments/l96/` convention, each with a `config.yaml` copy alongside
for full reproducibility), symlinked back into this worktree's own
`experiments/` under the new names. This branch's own local (gitignored)
checkpoint directories for the retired old family (old Q1/Q3/Q4/
Q3-noise0.05/Q5, plus the superseded old-default Q7 and assorted smoke/
validation/seed-ablation debris tied to them) were deleted to free disk
-- their config YAMLs (git-tracked recipes) are untouched, so those
experiments remain reproducible by rerunning training.

**Rationale:** See PLAN.md's 2026-09-14 "T-channels bench refresh" ->
"Promoted to the canonical Q1-Q4" section for the full history and the
final S0/S1 comparison numbers.

**Verification:** After the file moves, each new symlinked checkpoint
path loads correctly (`torch.load` succeeds); re-running
`eval_qg_neural_s0_s1.py --schemes Q1 Q2 Q3 Q4` reproduced the exact same
EV numbers as before the migration (regression check). `ruff check` clean
on all touched files. `pytest tests/test_qg_neural.py
tests/test_fourdvarnet.py tests/test_monai_unet_qg2d.py -m "not slow"`
in both `fdv` and `fdv-monai-proto` envs.
