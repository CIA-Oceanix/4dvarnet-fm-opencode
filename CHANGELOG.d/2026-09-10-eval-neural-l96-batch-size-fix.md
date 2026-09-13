## 2026-09-10: L96 — fix `eval_neural_l96.py` silently ignoring `--batch-size`

**Summary:** `--batch-size` was accepted and logged by `eval_neural_l96.py`
but never actually passed to `prepare_dataset()`, so its `DataLoader` always
used the hardcoded default of 200 regardless of the flag -- silently
defeating the documented reduced-batch-size workaround for large-model
eval-time OOMs (the same one already needed for `DirectUNet-L`). Found
while evaluating `FDV2_grad_state_monai_l96`'s checkpoint on nominal S0:
`--batch-size 16` produced an identical `CUDA out of memory` error to the
unflagged default, twice in a row, before the bug was traced.
**Files modified:** `eval_neural_l96.py` -- pass `batch_size=args.batch_size`
into the `prepare_dataset(...)` call.
**Rationale:** A CLI flag that is silently a no-op is worse than no flag at
all -- it looks like a workaround was tried and failed, when it was never
actually applied.
**Verification:** default (200) unchanged, so every prior invocation that
never passed `--batch-size` is unaffected; re-running the FDV2 grad+state
eval with `--batch-size 16` after the fix succeeded (previously OOM'd
identically twice).
