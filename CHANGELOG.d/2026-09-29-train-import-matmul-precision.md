## 2026-09-29: importing train.py no longer switches float32 matmuls to bfloat16

**Summary:** `train.py` called `torch.set_float32_matmul_precision('medium')` at import time, so any process importing it ran float32 matmuls in bfloat16 wherever a fast bf16 path exists:
- the test suite, where three test modules import `train` at collection;
- evaluation via `evaluation/neural_inference.py`.

The call now sits in `train.main()`, so training runs are unchanged.
**Files modified:**
- `train.py` — precision set in `main()` instead of at import.
- `tests/test_baselines_etks.py` — the exact linear-Gaussian RTS check pins `'highest'` precision (fixture, restored afterwards) and reports the active precision and the CPU's bf16/AMX flags on failure.
- `tests/test_import_side_effects.py` (new) — importing `train` / `evaluation.neural_inference` in a fresh interpreter must leave the precision unchanged; it fails on the old `train.py`.
**Rationale:** `test_linear_gaussian_matches_exact_rts_smoother` failed once on CI (#301, max error 1.8e-3 vs atol 2e-4) and passed on rerun. Locally it is deterministic with error 3e-7 (400 repeats, 1–8 threads; also on an avx512_bf16 node under 'medium'). The bfloat16 path is the only process-wide numeric switch found that differs between a full-suite CI run and a single-file run. It is consistent with the error size (bf16 ≈ 3 significant digits) and with the intermittency (varying runner CPUs). It was not reproduced on AMX hardware here (no AMX node available at the time).
**Behaviour change outside tests:** evaluation scripts that import `train` indirectly no longer inherit 'medium' precision. On Ampere-or-newer GPUs (A100/H100/L40S) and AMX CPUs their float32 matmuls are now exact float32. RTX 8000 (Turing) runs are unaffected. Training through `train.py` is unchanged.
**Verification:** `pytest -m "not slow"` on the ETKS, import-side-effect, golden DA and all train-importing test modules: 215 passed. The new guard fails on the previous `train.py`.
