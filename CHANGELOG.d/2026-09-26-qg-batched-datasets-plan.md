## 2026-09-26: Plan — batched GPU generation for QG Options B/C and a train/val/test dataset framework

**Summary:** Adds `docs/plans/tech/qg_batched_generation_datasets.md` (DRAFT
v1). It covers:
- **Batched generation:** a `BatchedQGDynamics` class with per-window
  parameters, batched GPU wind (gyrostat, OU, surrogate, a split-specific
  attractor-state cache), and the Option C coupled step co-stepped inside
  the ocean's RK4 stages;
- **a dataset framework:** a versioned spec and manifest, hierarchical
  `SeedSequence` streams with a separate test entropy (so the test set does
  not depend on train size), enforced independence rules and leakage
  checks, a Sobol / Latin-hypercube factor design with regime
  stratification, a lean storage format, and optional train-time
  regeneration;
- **a compute-time assessment.**

**Files modified:** `docs/plans/tech/qg_batched_generation_datasets.md` —
new. `docs/README.md` — index row.
**Rationale:** The user asked for a plan to batch Options B and C on GPUs,
with train/val diverse and test fully independent. Measurements on an
RTX 8000:
- **Batching saturates at 0.026 ms per window-step** (batch 128–256),
  against 3.05 ms serial, i.e. about 100×.
- **The coupled step costs 1.2–1.4× a one-way step** at saturation (2.25×
  at batch size 1, from kernel-launch overhead).
- **A 1000/100/100 dataset takes about 5–10 minutes** against about 10 hours
  serial today.
- **Today's format stores 33.9 MB per window**, half of it recomputable
  fields: 41 GB for 1000/100/100. The lean format needs about 5 GB.

The legacy seed arithmetic was checked: no exact collision across splits,
but interleaved ranges and nothing enforcing independence. It stays for the
published `ou` benchmark.
**Verification:** Docs only. `pytest tests/test_docs_layout.py` passes.
