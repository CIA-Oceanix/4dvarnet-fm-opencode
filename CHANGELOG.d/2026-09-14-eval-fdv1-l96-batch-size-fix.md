## 2026-09-14: eval_fdv1_l96.py --batch-size was dead code

**Summary:** `eval_fdv1_l96.py`'s `--batch-size` CLI flag was parsed and logged
but never actually forwarded to `prepare_dataset()`, so every run silently
used the hardcoded default of 200 (all 200 test windows in a single batch)
regardless of what was passed.

**Files modified:** `eval_fdv1_l96.py` — `prepare_dataset(...)` call now
passes `batch_size=args.batch_size`.

**Rationale:** Discovered while running the full S0/S1 eval for job 53501
(`FDV2_grad_state_monai_l96_Stier_priorresidual_N5`, `update_input=grad+state`):
this mode's real `torch.autograd.grad(..., create_graph=True)` at eval time
is far more memory-hungry per sample than every other mode evaluated so far
(`subgrad+state`, `gradsplit+state`, FDV1's `obs+state`, none of which call
autograd.grad at all) -- batch=200 OOM'd (CUDA out of memory, 6.71 GiB
alloc attempt against 40+ GiB already in use). Passing `--batch-size 16`
had zero effect (identical OOM, byte-for-byte the same numbers) because the
flag was never threaded through -- `prepare_dataset`'s own `batch_size`
kwarg plumbing (`evaluation/neural_inference.py`) was always correct; the
bug was purely `eval_fdv1_l96.py` failing to forward it. This bug was
latent until now: every other model type/mode evaluated through this
script was cheap enough per-sample that batch=200 never mattered.

**Verification:** Reran the eval with the fix (`--batch-size 16`) --
succeeded (S0 RMSE 0.4286, S1 RMSE 0.4230, no OOM). `pytest
tests/test_neural_inference.py -q` -- unaffected (`prepare_dataset` itself
was never broken). No dedicated CLI-script test added: no `eval_*.py`
script in this project has direct test coverage today (only the
`evaluation/` library functions they call do).
