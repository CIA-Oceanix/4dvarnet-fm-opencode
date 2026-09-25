## 2026-09-25: L96 rank histograms (P1 C4) + opt-in DA member output

**Summary:** Adds opt-in ETKF/EnKF member storage and a rank-histogram probe, then runs the experiment P1 §5 flags as `\needsrun` for C4. Once spread and per-channel mean bias are corrected, the rank histograms do **not** single out the class without `Psi_NG`:
- the ensemble filters' apparent shape defect is mostly a mean bias from forward-model error;
- their residual shape error is comparable to PredictStateCFM's and smaller than SDA's, with opposite tails;
- only VanillaCFM is close to flat.

**Files modified:**
- `evaluation/baselines.py`: `store_ensemble` on the batched `ETKF.assimilate_batch` / `EnKF.assimilate_batch` (off by default; results then carry the real members).
- `evaluation/run_l96.py`, `evaluate_all_l96.py`: `save_members` / `--save-members`, writing `*_members.npz` (`members` (W, T, D, N), `truth`) in the neural members layout.
- `reports/l96/probe_rank_histograms.py`: raw, spread-corrected and spread- plus bias-corrected histograms, RI, extreme-bin ratio, rank bias.
- `batch/run_l96_rank_histograms.sbatch`: DA members produced, used and deleted on node-local `/tmp`.
- `tests/test_rank_histograms_and_da_members.py`.
- `reports/l96/outputs/rank_histograms/*`: JSON/PNG outputs and DA summaries.
- `docs/results/l96_rank_histograms_c4.md`.

**Rationale:**
- Settles the paper's largest open experiment.
- DA members are ~1.7 GB per method and Odyssey is nearly full, so nothing large is written there.

**Members location:** after #259, `--save-members` writes through `evaluation/members_store.py` (node-local /tmp by default; `--keep-members` / `FDV_KEEP_MEMBERS=1` keeps them next to the trajectories).
**Verification:**
- `pytest` over the baseline and DA suites plus the new tests: 120 passed. The DA golden values are unchanged, and the stored members' mean equals the analysis.
- The DA re-runs reproduce the stored RMSEs to within 0.4%.
- `ruff check` is clean.
