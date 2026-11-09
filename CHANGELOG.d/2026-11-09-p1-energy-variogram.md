## 2026-11-09: Energy and variogram scores of the P1 L96 ensembles

**Summary:** Energy and variogram scores (p = 0.5), with skill scores against a climatological ensemble, for the P1 L96 ensembles: the 30-member ETKF, EnKF and ETKS, and the 2D CFM, SDA1 and D-Unet+SDA (seed 1). Regular test set, S0 and S1, next to FMSS_0.75. This answers the P1 discussion `\needsrun`.
**Files modified:**
- `reports/l96/p1_energy_variogram.py`: `score` step, run in the job on the node-local member dumps, writes per-window arrays; `report` step writes `reports/l96/outputs/fm_score/p1_energy_variogram.json`.
- `batch/run_l96_p1_energy_variogram.sbatch`: DA re-run with `--save-members` and the 2D learned re-evaluation with `--members-file full`. Members stay on node-local `/tmp`; only the per-window scores are kept, in `experiments/l96/p1_energy_variogram/`, which is added to the `benchmark_extended` bundle.
- `reports/l96/outputs/fm_score/p1_energy_variogram.json`: new.
**Rationale:** P1 compares FMS_tau with the standard multivariate proper scores.
- The energy score is sensitive mostly to the mean.
- The variogram score reads the dependence structure: on snapshots of the 24 observed variables, and on space-time patches of 24 variables x 10 times 30 steps apart.
- The climatological reference is one fixed set of 30 truth samples per case, shared by every scheme.
**Verification:** A synthetic smoke test reproduced the expected ensemble-mean RMSE (1/sqrt(30) for unit-noise members). The ensemble-mean RMSE of every row matches the stored benchmark RMSE within the sampling noise of a new ensemble realisation. `ruff check` passes.
