## 2026-10-19: QG hybrid SDA warm-started from an obs-conditioned flow (G2-3D)

**Summary:** `--warm-start-run` now also takes a CFM run (G2): it draws `n_members` flow samples per window and either starts each SDA member from its own sample (`--cfm-warm-start member`, default) or all members from their mean (`mean`). `sda_guided_sample` gains `member_starts` (B, T, D, n_members).
**Files modified:** `evaluation/sda_sampler.py` — per-member SDEdit starts; `evaluation/run_qg_specwind_neural.py` — `cfm_warm_starts`, `--cfm-warm-start`, CFM warm start also feeds the FM-posterior sampler's member starts, recorded in the meta; `tests/test_qg_specwind_neural_eval.py` — per-member starts at tau0=1, CFM warm starts stitched and full-window.
**Rationale:** on the obs-config stress test G2-3D is the best end-to-end scheme (beats pure-noise SDA at c6/c12/15% noise) and beats the G1-3D-warm hybrid at 15% noise; the hybrid should start from the best end-to-end estimate.
**Verification:** `pytest -m "not slow" tests/test_qg_specwind_neural_eval.py tests/test_sda_sampler.py tests/test_qg_vanilla_cfm_3d.py` — 59 passed; ruff clean.
