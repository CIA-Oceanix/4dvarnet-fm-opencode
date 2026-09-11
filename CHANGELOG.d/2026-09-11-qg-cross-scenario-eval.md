## 2026-09-11: QG Q3/Q4 cross-scenario S0/S1 evaluation

**Summary:** Q3 and Q4 completed full 200-epoch training (Q3: 5.02h, S0 PSI
EV=0.913/PV-q EV=0.205; Q4: 9.71h, S0 PSI EV=0.913/PV-q EV=0.205). Added a
new deterministic `cond_mode="scenario"` and `eval_qg_neural_s0_s1.py` to
test whether Q3/Q4 degrade under S1 model error the way DA baselines do.

**Files modified:**
- `data/qg_neural.py` -- new `_scenario_forcing_and_params` +
  `cond_mode="scenario"`: reads whatever the window's own S0/S1 scenario
  wrapper designates as believed (`wind_state_corrupted`/`da_params`),
  unlike `"true"`/`"noisy"` which always ignore the scenario label.
- `train_qg_neural.py` -- `"scenario"` added to `--cond-mode` choices.
- `eval_qg_neural_s0_s1.py` (new) -- evaluates Q1/Q3/Q4 on both test_s0/
  test_s1; supports `--lag-days`/`--noise-frac`/`--s1-param-bias`/
  `--s1-amp-bias` overrides with a cache-reuse trick (loads the cached
  truth at its original key, then cheaply redraws obs/init-state at the
  requested lag/noise) to avoid missing the cache and triggering a
  from-scratch truth rollout.
- `tests/test_qg_neural.py` -- cond_mode="scenario" regression test
  (matches "true" on S0-scenario windows, differs/biased on S1, deterministic).

**Rationale:** Two apples-to-apples config mismatches were caught before
trusting the comparison: (1) the initial eval used the models' own
lag=1.0/noise=0.01 training distribution, not comparable to the DA
baselines' lag=5.0/noise=0.05 reference case; (2) even after matching
lag/noise, the S1 severity itself defaulted to `QGConfig`'s class default
(`s1_param_bias=s1_amp_bias=0.15`), not the DA campaign's actual explicit
override (0.1/0.1, deliberately milder). Final result, fully matched: Q4 is
essentially immune to S1 model error (Δq=-0.007, ~14x smaller than ETKF's
own degradation); Q3, at the correct bias level, is actually less sensitive
than ETKF/EnKF (an earlier wrong-bias run had suggested the opposite).

**Verification:** `pytest tests/test_qg_neural.py -m "not slow"` (fdv env)
-- 29 passed, 1 skipped (monai-gated). `ruff check` clean.
