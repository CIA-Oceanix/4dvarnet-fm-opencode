# FDV speed levers beyond checkpointing: torch.compile and JAX (investigation notes, 2026-09-08)

**Status:** investigation notes, not an execution plan. Captures why `torch.compile`
was tried and shelved as a follow-up to the gradient-checkpointing work
(`[[project_fdv_gradient_checkpointing_plan]]`, PR #172/#173), and why a JAX port
wasn't pursued either, despite JAX structurally avoiding the specific bug hit here.
Revisit only if the condition in the last section changes.

---

## 1. Context

PR #172 wrapped each unrolled iteration of `FourDVarNetSolver`/`FourDVarNetPredictStateCFM`
in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` to cut peak GPU
memory from O(`N_outer`) to effectively O(1) (measured in PR #173:
`reports/l96/outputs/l96_grad_checkpoint_benchmark.md` — up to 32x reduction at
`N_outer=40`, at a consistent ~1.55-1.66x wall-clock overhead). That wall-clock
overhead is the checkpointing tax: one extra forward recompute per iteration during
backward. `torch.compile` was explored as a way to claw some of that back via kernel
fusion, without touching JAX.

## 2. `torch.compile` experiment (GPU, Quadro RTX 8000, torch 2.4.1+cu121, N_outer=10)

Compiled `models.fourdvarnet._solver_iteration` (the shared per-iteration helper) via
`torch.compile`, crossed with checkpoint on/off, for `obs+state` (cheap mode) and
`grad+state` (costliest, uses `torch.autograd.grad(..., create_graph=True)`):

| mode | checkpoint | compile | outcome |
|---|:---:|:---:|---|
| obs+state | on | off | 238.5 ms, 168.5 MB (current production default) |
| obs+state | on | **on** | **crashes**: `BackendCompilerFailed: ... Please convert all Tensors to FakeTensors first` |
| obs+state | off | off | 160.3 ms, 1183.8 MB |
| obs+state | off | on | 155.8 ms, 1038.8 MB — only ~3% faster, ~12% less memory, behind a ~50s compile warm-up |
| grad+state | on | off | 1008.2 ms, 521.0 MB (current production default) |
| grad+state | on | **on** | **crashes**, same error as above |
| grad+state | off | **on** | **crashes**: `RuntimeError: torch.compile with aot_autograd does not currently support double backward` |

## 3. Root causes (not just the local error message)

- **The checkpoint+compile crash is a known, already-fixed PyTorch bug** —
  [pytorch/pytorch#121966](https://github.com/pytorch/pytorch/issues/121966), same
  repro shape (dropout + `checkpoint(..., use_reentrant=False)` + `torch.compile`).
  Fixed by [PR #123196](https://github.com/pytorch/pytorch/pull/123196), confirmed
  working on **torch 2.5.1** (issue closed Nov 2024). Root cause per a PyTorch
  maintainer: *"compile(checkpoint) works but checkpoint(compile) does not"* — this
  experiment compiled `_solver_iteration` and fed the compiled function into
  `checkpoint(...)`, i.e. exactly the unsupported direction. Fixable in principle by
  upgrading past 2.5.1 **and** restructuring to `torch.compile` the outer
  `forward()`/loop (letting `checkpoint(...)` calls happen *inside* the compiled
  region) rather than compiling `_solver_iteration` itself.
- **Double backward is a separate, still-open, architectural limitation** —
  [pytorch/pytorch#91469](https://github.com/pytorch/pytorch/issues/91469)
  ("torch.compile with aotautograd does not support double backwards"), open, last
  updated **2026-05-15**, tracked as a feature/redesign effort (moving backward
  compilation to always happen at backward time), not a bug fix. Ed Yang's
  ["State of torch.compile" post (Aug 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
  confirms double backward remains unsupported as of that writing. No evidence found
  (as of 2026-09) that this has landed since.
- **This is a PyTorch-specific artifact, not a mathematical necessity.** `torch.compile`
  (TorchDynamo + AOTAutograd) is bolted onto an eager, define-by-run autograd system:
  AOTAutograd traces a joint forward+backward graph ahead of time by calling
  `torch.autograd.grad` internally during tracing, which collides with a forward that
  already contains its own `create_graph=True` `autograd.grad` call (our
  `grad-only`/`grad+state`), and `checkpoint`'s `fork_rng` (needed to reproduce
  dropout's mask exactly across recompute) touches real, non-fake RNG state tensors
  mid-trace. JAX's `jax.grad`/`jax.jit`/`jax.checkpoint` are first-class, composable
  transformations over the same functional IR (jaxpr) from the start — `jit(grad(grad(f)))`
  and `jit(checkpoint(f))` are ordinary usage, not special-cased combinations
  ([JAX autodiff docs](https://docs.jax.dev/en/latest/automatic-differentiation.html),
  [`jax.checkpoint` docs](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) —
  *"integrates smoothly with jax.jit"*). JAX also has no analogue of the RNG half of
  this crash: it threads an explicit `PRNGKey` instead of PyTorch's implicit global
  RNG state, so there's no `fork_rng`-style save/restore to conflict with tracing.

## 4. Why this doesn't change the conclusion

JAX would plausibly let `grad-only`/`grad+state` reach the same modest fusion speedup
the working `torch.compile` config already showed for `obs+state` (~3%) — a real
improvement over the PyTorch ceiling (currently zero for those modes), but still the
same order of magnitude, since `jax.checkpoint`'s recompute-during-backward is the
identical mathematical tradeoff (JAX doesn't remove the ~1.6x checkpointing time
cost, only the compatibility failures around it). That's not enough gain to justify
rewriting a large, mature PyTorch/Lightning/Hydra codebase (dynamics models, DA
baselines, training pipeline, Hydra configs) in JAX/Flax/Equinox.

**Don't re-attempt `torch.compile` for FDV without first checking whether
[pytorch/pytorch#91469](https://github.com/pytorch/pytorch/issues/91469) has closed.**
If it has, and if a genuinely large-`N_outer`/large-backbone training run is planned
where even the cheap-mode ~3% (or a JIT-fused equivalent) would matter at scale,
re-run the benchmark in `reports/l96/generate_l96_grad_checkpoint_benchmark.py`-style
with `torch.compile` wrapping the outer `forward()` (not `_solver_iteration`) before
deciding again.
