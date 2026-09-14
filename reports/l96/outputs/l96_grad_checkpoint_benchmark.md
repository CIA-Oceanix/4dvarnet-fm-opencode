# FDV Gradient Checkpointing: Memory/Time Microbenchmark

**Question:** PR #172 added `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` around each unrolled iteration of `FourDVarNetSolver`/`FourDVarNetPredictStateCFM` (see CHANGELOG 2026-09-08, `models/fourdvarnet.py::_solver_iteration`) to keep activation memory from scaling linearly with the unroll length. Gradient/output-value equivalence was verified by `tests/test_fourdvarnet.py::TestGradientCheckpointing`, but that says nothing about the actual memory savings or the recompute time cost -- this microbenchmark measures both directly.

**Setup:** synthetic batches (`states`/`obs`/`obs_mask` only -- no real L96 data generation needed to exercise `forward()`+`backward()`), sized to match `config/experiment/FDV2_grad_state_l96_fixedw.yaml`: B=16, T=500 (`da_window_steps`), D=24, `hidden_channels=[64, 128, 256]`, `time_emb_dim=64`, `dropout=0.1`. Two `update_input` modes: **obs+state** (no `prior_unet`, no `autograd.grad` -- the cheapest mode) and **grad+state** (`prior_unet` + `torch.autograd.grad(..., create_graph=True)` each iteration -- the costliest mode, and the specific one `use_reentrant=False` targets). Loss is a plain `out.pow(2).sum()` (memory/time depend only on `forward()`'s graph, not on the actual loss formula). 3 warmup + 5 timed forward+backward passes per cell, GPU: Quadro RTX 8000. Peak memory via `torch.cuda.max_memory_allocated()` (reset per cell), time via `torch.cuda.synchronize()`-bracketed `time.perf_counter()`. Checkpointing bypassed for the "off" rows the same way `tests/test_fourdvarnet.py`'s `TestGradientCheckpointing` does: `unittest.mock.patch` on `models.fourdvarnet.checkpoint` to a passthrough -- same code path, only the checkpoint mechanics differ.

---

## Results

| N_outer | mode | checkpoint | peak mem (MB) | mem reduction | time (ms) | time overhead |
|---:|---|:---:|---:|---:|---:|---:|
| 5 | obs+state | off | 603.9 | -- | 69.6 | -- |
| 5 | obs+state | on | 161.1 | 3.7x | 108.1 | 1.55x |
| 5 | grad+state | off | 2325.2 | -- | 287.3 | -- |
| 5 | grad+state | on | 518.8 | 4.5x | 481.9 | 1.68x |
| 10 | obs+state | off | 1174.3 | -- | 135.9 | -- |
| 10 | obs+state | on | 168.5 | 7.0x | 227.4 | 1.67x |
| 10 | grad+state | off | 4609.6 | -- | 572.3 | -- |
| 10 | grad+state | on | 526.1 | 8.8x | 932.3 | 1.63x |
| 20 | obs+state | off | 2315.0 | -- | 272.3 | -- |
| 20 | obs+state | on | 183.1 | 12.6x | 465.6 | 1.71x |
| 20 | grad+state | off | 9177.9 | -- | 1140.8 | -- |
| 20 | grad+state | on | 540.8 | 17.0x | 1861.1 | 1.63x |
| 40 | obs+state | off | 4596.6 | -- | 562.0 | -- |
| 40 | obs+state | on | 212.4 | 21.6x | 953.4 | 1.70x |
| 40 | grad+state | off | 18314.5 | -- | 2271.0 | -- |
| 40 | grad+state | on | 570.1 | 32.1x | 3713.9 | 1.64x |

---

## Takeaways

- **Memory**: without checkpointing, peak memory scales linearly with `N_outer` (doubling `N_outer` roughly doubles peak memory), as expected since every iteration's UNet(+prior_unet) forward stays alive for backprop. With checkpointing, peak memory is essentially flat in `N_outer` (bounded by roughly one iteration's activations plus fixed overhead) -- the reduction factor grows with `N_outer` and reaches its largest value at the largest `N_outer` tested.
- **Time**: a consistent overhead across every configuration, matching the expected cost of one extra forward recompute per iteration during backward (roughly 2 forwards + 1 backward instead of 1 forward + 1 backward) -- stable regardless of `N_outer` or `update_input` mode, not a cost that compounds with unroll length.
- **Net effect**: turns peak memory from O(N_outer) into effectively O(1), at a stable time cost -- the previous hard limiter on how far `N_outer` (or the backbone size) could be scaled was GPU memory, not compute time, so this trade is worth taking whenever memory (not wall-clock) is the binding constraint, e.g. before scaling either the unroll length or the MonaiUNet backbone prototype further.

---

