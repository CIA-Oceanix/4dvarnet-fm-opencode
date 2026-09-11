## 2026-09-10: QG — Strong-4DVar/Weak-4DVar N=100 results for the revised S1 reference case

**Summary:** Completes the revised-S1 (model-error) campaign started in the "revised S1
(model-error) reference case, full ETKF/EnKF N=100 campaign" entry (same day) -- that
entry's own committed data only had ETKF/EnKF, since the completed Strong-4DVar/Weak-4DVar
runs were lost to a `git stash -u` accident before being committed (see
`feedback_stash_u_deletes_untracked_scratch_files` memory) and had to be re-run. Full N=100
results, all 4 methods, S0 vs revised S1 (ψ EV / q EV): ETKF 0.921/0.405 → 0.874/0.307,
EnKF 0.947/0.481 → 0.896/0.331, Strong-4DVar 0.971/-0.126 → 0.931/-0.857, Weak-4DVar
0.966/-0.035 → 0.947/-0.501. EnKF remains the clear S1 winner -- both 4DVar methods'
already-negative S0 q EV gets much worse under S1's real model error, while ψ stays
comparatively robust for every method. See PLAN.md's "Revised S1 (model-error) reference
case" section for the full writeup.
**Files modified:** `reports/qg/outputs/qg_repro_validation_s1/{strong4dvar,weak4dvar}.json`
(new, N=100).
**Rationale:** the revised-S1 reference case isn't complete without all 4 DA baselines.
**Verification:** re-ran via `qg_da_s1_scratch.py` (SLURM jobs 52994/52995) after fixing the
stale-`.pyc`-bytecode-cache issue that broke the first two re-run attempts (cleared
`__pycache__` project-wide); results are bit-for-bit identical to the originally-computed
(and lost) numbers reported earlier in this session, confirming reproducibility.
