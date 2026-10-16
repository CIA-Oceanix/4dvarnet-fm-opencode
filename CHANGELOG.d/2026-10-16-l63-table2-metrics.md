## 2026-10-16: L63 Table-2 metrics (EV, CRPSS, SS) for the DA and neural schemes

**Summary:** Evaluation scripts for the L63 column of the paper's Table 2 at `obs_interval=10` (now the default in these scripts): ETKF/EnKF/ETKS (30 members, inflation tuned on 50 validation windows), Strong/Weak-4DVar (L-BFGS cycled over sub-windows, sub-window and model-error scale tuned), and the learned schemes D-Unet, CFM, guided SDA2 and the D-Unet+SDA2 hybrid (guidance weight and start time tuned).
**Files modified:** `eval_da_l63.py` — ensemble DA scores, shared metric code; `eval_da_l63_4dvar.py` — L-BFGS 4DVar on the L63 dynamics; `eval_neural_l63.py` — neural/SDA/hybrid scoring.
**Rationale:** The L63 column of Table 2 was empty. The batched Adam 4DVar path did not converge on the 300-step stochastic window (Strong EV 0.23 at S0), so the 4DVar rows use cycled L-BFGS instead.
**Verification:** Full runs at `obs_interval=10` on 200 test windows (results in `experiments/l63_da_table2/`, gitignored); smoke-tested on small windows first. No unit tests added.
