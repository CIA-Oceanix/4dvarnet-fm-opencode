## 2026-09-15: Rerun L63 DA baselines, add Monai-backbone results table

**Summary:** `eval_baselines.py` called `build_datasets(cfg)` without a `train_case`,
which (since `data.build.build_datasets` only ever returns one case's test split) silently
limited it to test_s0 -- the pre-existing `baselines_s0s1.json`/`.npz` cache must have been
produced by running it twice by hand with different overrides. Fixed to loop s0 then s1 in
one process, and reran Weak-4DVar/Strong-4DVar/EnKF/ETKF on both cases against the fixed
`evaluation/baselines.py` (see the merge-regression fix, same day). Also added a
"State estimation (Monai backbone)" table to `reports/README.md` from the
`monai_direct_unet`/`monai_tweedie*`/`monai_vanilla_cfm*` results already sitting in
`experiments/l63/` (same configs/protocol as the non-Monai table, `models/
monai_unet_adapter.py` backbone instead of `models/unet.py`).
**Files modified:** `eval_baselines.py` — loop both train_case values;
`reports/README.md` — updated Weak/Strong-4DVar/EnKF/ETKF rows (EnKF/ETKF shift ~2-5%,
consistent with the ETKF/EnKF analysis-step fix rather than noise), added the Monai table.
**Rationale:** user requested a baseline rerun in parallel with the SDA prior training, and
a new report table summarizing the already-trained Monai-backbone L63 models.
**Verification:** full baseline rerun (`eval_baselines.py`, both cases, ~24 min); Monai
table numbers read directly from each `experiments/l63/monai_*/results.json`.
