## 2026-10-12: QG gyrostat SDA (G3) — prior on the batch-16 / 15-day protocol, per-step velocity composition, warm start, 3D prior

**Summary:** G3 config on the protocol that won for G1 (batch 16, 1200 epochs, 15-day crops, peak LR 3e-4); flow models built for a shorter window are now run on the 30-day test state by composing their velocity over overlapping sub-windows at every flow step (`ComposedFlow`), so each ensemble member is one coherent trajectory; SDA members can be warm-started at flow time τ0 from a DirectUNet run's estimate.
**3D prior:** `sda_prior_3d` (`MonaiSDAPrior3DQG`): the same prior on MONAI's `DiffusionModelUNet` with `spatial_dims=3` over (time, ny, nx), convolutions periodic in space and zero-padded in time (`SpaceCircularConv3d`), weights shared across days, so one network runs on any window length (`--full-window` / `FULL=1` scores it on the whole 30-day window instead of composing). Narrow widths [32, 64, 128]: 10.0 M parameters; 0.86 s/step at batch 16 on an RTX 8000 (2D M: 0.15 s), about the data pipeline's own pace, so a 1200-epoch run costs about the same wall time.
**Files modified:**
- `data/qg_neural.py` — `compose_days` factored out of `windowed_estimate` (unchanged behaviour)
- `evaluation/run_qg_specwind_neural.py` — `ComposedFlow`, `renorm_psi`, `sda_members(mean_estimate, tau0)`, `evaluate(warm_start, tau0)`, `--warm-start-run` / `--tau0`; DirectUNets keep blending sub-window estimates
- `batch/run_qg_specwind_neural_eval.sbatch` — `WARM` / `TAU0`; default `--gres=gpu:rtx8000:1` (the cluster rejects an untyped GPU request)
- `config/experiment/G3_sda_prior_tchannels_specwind_b16_w15.yaml`, `G3_sda_prior_3d_specwind_b16_w15.yaml` — new
- `models/monai_unet_qg2d.py` — `SpaceCircularConv3d` (the MONAI conv factory's dim-3 entry), `MonaiSDAPrior3DQG`
- `train_qg_neural.py` — `sda_prior_3d` in `build_model`, `FLOW_TYPES`, CLI choices; `SDA_TYPES`
- `tests/test_qg_specwind_neural_eval.py` — composition, spread, warm start, renorm tests
- `tests/test_qg_sda_prior_3d.py` — new: any window length, spatial periodicity, no wrap in time, training loss, 10 M parameters, guided sampler on full and composed windows
**Rationale:** blending independently sampled sub-windows (what `windowed_estimate` does for a deterministic estimate) averages independent members where they overlap and shrinks the ensemble spread; composing the velocity at each step is the SDA construction (Rozet & Louppe 2023) and keeps guidance acting on the full 30-day trajectory. The warm start puts an SDA ensemble around the best deterministic estimate (G1-L 15-day + K^-1.5, S0 0.856).
**Verification:** `pytest -m "not slow"` on test_qg_specwind_neural_eval, test_qg_specwind_kpow_window, test_qg_specwind_neural, test_qg_neural, test_sda_sampler — 159 passed (adding test_qg_config_persistence and test_qg_sda_prior_3d); ruff clean; 1-epoch smoke run of the 3D prior scored in both modes; 2D prior (60026) and 3D prior launched on the cluster.
