## 2026-09-25: L96 S1 evaluation fed params-conditioned models the TRUE params — fixed, SDA2/SDA3 re-evaluated

**Summary:** `collate_joint_eval` built `params` from the plain `F/c1/hx/eps/fast_weights` keys, which hold the TRUE values in S1 windows (the biased DA-model params live in `*_da`, read by the DA baselines and the training collate). Every params-conditioned model (SDA2/SDA3, alone and as DirectUNet-hybrid prior; the FourDVarNet trueprior "biased" mode) was therefore conditioned on the truth at S1. Fixed and every affected evaluation re-run; S0 reproduces bit-for-bit.
**Files modified:**
- `evaluation/neural_inference.py` — `_window_da_param_vector` (`*_da` first, plain keys as fallback), used by `collate_joint_eval`
- `tests/test_eval_da_params.py` — S1 gives the biased params, S0 the plain ones, flattened fast weights do not mask `fast_weights_da`
- `batch/run_l96_sda_da_params_rerun.sbatch` — SDA2/SDA3-fix seeds, the 400/1200-epoch DU->SDA2 hybrids, a new DU(1200)->SDA3-fix hybrid, the P1 SDA2/SDA3 gw 20 runs, and the validation re-check
- `reports/l96/generate_l96_benchmark_extended_report.py` — DU(1200)->SDA3-fix hybrid row, SDA3-fix in the validation tables, S1-conditioning protocol note, findings 1 and 6 rewritten from the data
- `reports/l96/outputs/{l96_benchmark_extended,l96_benchmark_default,p1_l96_benchmark}.*` — re-rendered
- `docs/results/l96_benchmark_default_paper_audit.md` — update note

**Rationale:** with the biased params at S1, SDA2-M degrades (+1.8% RMSE) while SDA3-fix-M, trained on noisy DA params, does not; as hybrid prior this decides robustness to model error: DU(1200)->SDA2-M goes 0.310 -> 0.333 (regular S0 -> S1), DU(1200)->SDA3-fix-M stays 0.312 -> 0.307. The SDA3-fix hybrid is the best scheme at S1 on both test sets and is also the better prior on the validation windows; guidance weight 25 and tau0 0.1 / gw 2 remain validation-optimal. "Conditioning is inert" no longer holds.
**Verification:** `pytest tests/test_eval_da_params.py tests/test_param_head.py tests/test_l96_normalization.py` (29 passed, 3 skipped); re-run S0 scores identical to the pre-fix ones for every affected run; reports re-rendered with the consistency checks passing.
