## 2026-10-12: QG gyrostat SDA (G3) — prior on the batch-16 / 15-day protocol, per-step velocity composition, warm start

**Summary:** G3 config on the protocol that won for G1 (batch 16, 1200 epochs, 15-day crops, peak LR 3e-4); flow models built for a shorter window are now run on the 30-day test state by composing their velocity over overlapping sub-windows at every flow step (`ComposedFlow`), so each ensemble member is one coherent trajectory; SDA members can be warm-started at flow time τ0 from a DirectUNet run's estimate.
**Files modified:**
- `data/qg_neural.py` — `compose_days` factored out of `windowed_estimate` (unchanged behaviour)
- `evaluation/run_qg_specwind_neural.py` — `ComposedFlow`, `renorm_psi`, `sda_members(mean_estimate, tau0)`, `evaluate(warm_start, tau0)`, `--warm-start-run` / `--tau0`; DirectUNets keep blending sub-window estimates
- `batch/run_qg_specwind_neural_eval.sbatch` — `WARM` / `TAU0`; default `--gres=gpu:rtx8000:1` (the cluster rejects an untyped GPU request)
- `config/experiment/G3_sda_prior_tchannels_specwind_b16_w15.yaml` — new
- `tests/test_qg_specwind_neural_eval.py` — composition, spread, warm start, renorm tests
**Rationale:** blending independently sampled sub-windows (what `windowed_estimate` does for a deterministic estimate) averages independent members where they overlap and shrinks the ensemble spread; composing the velocity at each step is the SDA construction (Rozet & Louppe 2023) and keeps guidance acting on the full 30-day trajectory. The warm start puts an SDA ensemble around the best deterministic estimate (G1-L 15-day + K^-1.5, S0 0.856).
**Verification:** `pytest -m "not slow"` on test_qg_specwind_neural_eval, test_qg_specwind_kpow_window, test_qg_specwind_neural, test_qg_neural, test_sda_sampler — 143 passed; ruff clean; G3 prior launched on the cluster.
