## 2026-09-24: Benchmark flow sampler → 30 members × 20 early-fine steps; P1 and benchmark-default reports re-scored

**Summary:** The L96 benchmark now scores flows (VanillaCFM, PredictStateCFM) with 30 members × 20 early-fine Euler steps (`tau_k = 1 − (1 − k/20)^0.5`), sub-directory `ens30_no20`, instead of 10 uniform steps (`ens30_no10`). Every flow row of `l96_benchmark_default.md` and `p1_l96_benchmark.md` is re-evaluated and both reports are regenerated. The P1 paper's §5/§6 flow numbers are refreshed.
**Files modified:**
- `batch/run_l96b_seeds.sbatch`, `batch/run_l96b_psc_seeds.sbatch`, `batch/run_l96_canonical_rlayout_eval.sbatch`: 20 steps.
- `reports/l96/generate_l96_benchmark_default_report.py`: `FLOW = ens30_no20`, a protocol line, and the transfer-ratio range now computed instead of hard-coded.
- `reports/l96/generate_p1_l96_benchmark.py`: `ens30_no20` for the 7 flow rows and the reconstruction figure; protocol text; paired-t claim updated.
- `reports/l96/outputs/{l96_benchmark_default,p1_l96_benchmark}.{md,json,png}` and the P1 figures: regenerated.
- `docs/papers/p1_structural_hypotheses/sections/05_results.tex`, `06_discussion.tex`: flow numbers and the under-dispersion range.
- `config/l96_benchmark_default.yaml`, `AGENTS.md`: the protocol recorded.

**Rationale:** On the benchmark-default checkpoints, early-fine N = 20 gives the best (PSC) or tied-best (Van) CRPS, beating uniform N = 80 at a quarter of the cost (`docs/results/cfm_tau_consistency_l96b.md`, #256). Effect on the reports:
- **Benchmark default (regular S0):**
  - PSC-M RMSE 0.409 → 0.403, CRPS 0.191 → 0.183, spread/RMSE 0.53 → 0.70;
  - Van-M RMSE 0.434 → 0.430, CRPS 0.201 → 0.196, spread/RMSE 0.62 → 0.79.
- **P1:**
  - VanillaCFM-M 0.3445 → 0.3412 (CRPS 0.1568 → 0.1527, spread/RMSE 0.535 → 0.614);
  - PSC-M 0.3584 → 0.3536 (spread/RMSE 0.400 → 0.487).
- **Qualitative findings:** unchanged except:
  - the under-dispersion range becomes 1.6–2.3× (was 1.8–2.4×);
  - Van vs PSC at S+ is paired t = 0.7, p = 0.48 (was 0.0), still indistinguishable.
- SDA keeps its guided 10-step protocol.

**Verification:**
- Before any change, both generators reproduced the checked-in reports byte for byte from the same inputs.
- After the change, the benchmark-default consistency check passes, and `pytest tests/test_l96_report_consistency.py`: 14 passed.
- SLURM 55361 (16 evals) and 55374 (5 evals), all completed.
