## 2026-09-26: PredictStateCFM linear skip connection option + results

**Summary:** Opt-in `skip_connection: linear` for PredictStateCFM (D = τ·x_τ + (1-τ)·net; default `none` is bit-identical to before) and `skip_loss: state|net` (MSE(D, x1), or the unweighted MSE(net, (1+τ)x1 - τx0); validation always MSE(D, x1)). Two 1200-epoch runs show the skip barely matters: the state-loss run matches PredictStateCFM, the net-loss run matches VanillaCFM, because the net loss is exactly VanillaCFM's velocity loss and the state loss is PredictStateCFM's (1-τ)²-weighted one.
**Files modified:**
- `models/vanilla_cfm.py` — `skip_connection`, `skip_loss` tau options; `_net` helper
- `train.py` — both keys in `PSC_TAU_OPTION_KEYS`
- `tests/test_psc_skip_connection.py` — default unchanged, blend formula, exact at τ=1, τ=0 head, velocity = net - x, net loss = unweighted target, validation loss is the state loss, config wiring, invalid options
- `config/experiment/L96B_predictstatecfm_monaiM_ep1200_skip_{state,net}loss.yaml`, `batch/run_l96b_psc_skip.sbatch`
- `docs/results/l96_psc_skip_connection.md` (+ index in `docs/README.md`)
**Rationale:** test whether an EDM-style skip parameterization improves PredictStateCFM; it identifies the loss weighting, not the parameterization, as what separates PredictStateCFM from VanillaCFM.
**Verification:** `pytest tests/test_psc_skip_connection.py tests/test_psc_tau_options.py tests/test_t5_variance_and_sampler.py` (41 passed at the time of the runs); results table from the benchmark sampler on the 200 test windows.
