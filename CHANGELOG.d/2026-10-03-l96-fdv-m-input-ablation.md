## 2026-10-03: L96 FDV M-tier input ablations (resid+state, zero_prior_input, aux_detach_x_final)

**Summary:** Three default-off options for the FDV solver and the M-tier runs that use them. Seed-1 results:
- FDV1-M (`obs+state`) remains the best M-tier variant (0.245 / 0.291 RMSE, regular / random S0).
- Replacing y by the masked residual (`resid+state`, identical initial weights) costs about 3% on the regular grid.
- The subgrad prior channel adds nothing when Φ is untrained, and hurts by 4–7% when Φ is trained as a prior.
- The original aux prior loss destabilizes the M-tier solver (blow-up at epoch ~320); detaching x̂ in that term fixes the instability.
**Files modified:**
- `models/fourdvarnet.py`, `train.py`:
  - `update_input: resid+state` (FDV1 with y → (x − y)·mask);
  - `zero_prior_input` (subgrad+state: feed 0 instead of x − Φ(x));
  - `aux_detach_x_final` (the aux prior cost on x̂ trains Φ only).

  All default off; existing configs are unchanged.
- `tests/test_fdv_resid_state.py`, `tests/test_fdv_zero_prior_input.py` (new):
  - identical init to `obs+state`;
  - inputs exactly as specified and zero where unobserved;
  - output independent of Φ when zeroed;
  - solver gradient equal to the no-aux case when detached;
  - the flag is rejected outside subgrad+state.
- `config/experiment/L96B_fdvresid_monaiM.yaml`, `L96B_fdvsubgrad_monaiM_aux01{,det,det_noprior}.yaml`; `batch/run_l96b_fdv_subgrad_aux.sbatch` (`FDV_CFG`, `FDV_SEED`).
- `docs/results/l96_fdv_m_input_ablation.md` (new); `docs/plans/analysis/l96_p2_fdv_benchmark.md`, `docs/README.md` — status and index.
**Rationale:** Explains why FDV2 subgrad, ahead of FDV1 at the S tier, trails it at M, and which input ingredient costs accuracy.
**Verification:** see the PR body. A re-evaluation at full float32 precision confirmed the pre-#302 evals to 4 decimals.
