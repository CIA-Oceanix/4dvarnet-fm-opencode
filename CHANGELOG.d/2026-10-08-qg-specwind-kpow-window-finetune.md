## 2026-10-08: QG gyrostat neural — sparse-tilted density draw, 15-day network window, fine-tuning

**Summary:** Three training options for the spectral-wind gyrostat networks, each with a G1-M batch-16 config: `data.cols_per_day_power` draws the train density with P(K) ∝ K^-a (a = 1.5 over 3–30: median 6, 22 % of draws at the test density K = 3, against 17 and 3.6 % uniform); `data.train_window_days` builds the network for a shorter window (15 days) trained on random whole-day crops of the 30-day windows and scored on the 30-day test windows from overlapping sub-windows (`data.window_stride`, tent-weighted average); `training.init_from` / `--init-from` fine-tunes from a finished run with its psi normalization and q-loss weight.
**Files modified:**
- `data/qg_neural.py` — `QGNeuralDataset(cols_per_day_power, crop_days, crop_starts)`; `slice_days`, `window_starts`, `windowed_estimate`
- `train_qg_neural.py` — YAML keys above, `--init-from` (architecture check, parent norm stats copied beside the run, parent q-loss weight), `load_run_weights`, model built at the network window, val tiled by non-overlapping crops, resolved config records the new keys
- `evaluation/run_qg_specwind_neural.py` — rebuilds a shorter-window network and estimates the 30-day windows with `windowed_estimate` (`--window-stride`, default the run's)
- `config/experiment/G1_direct_unet_tchannels_specwind_b16_{kpow,kpow_ft,w15}.yaml` — new
- `tests/test_qg_specwind_kpow_window.py` — new
**Rationale:** the test observes 3 columns/day, the sparse end of the uniform 3–30 train range (median 17); the G1-L/XL capacity study saturated, so the next levers are the train density distribution and the network window. The daily-mean target's ceiling was checked first (0.998 at daily, 1.000 at 6 h), so a finer time resolution was not pursued.
**Verification:** `pytest -m "not slow" tests/test_qg_neural.py tests/test_qg_config_persistence.py tests/test_qg_specwind_neural.py tests/test_qg_specwind_neural_eval.py tests/test_qg_cols_sampling.py tests/test_qg_specwind_kpow_window.py` — 147 passed; `ruff check` clean on touched files; 1-epoch smoke runs of the w15 and kpow_ft configs on the cluster.
