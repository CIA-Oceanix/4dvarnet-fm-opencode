# L96 fast_weights randomization — progress log

## Status
- Phase A (refactor, legacy-compatible): [in progress — bugs fixed, tests green]
- Phase B (repro gate, 1e-3 rel): [not started]
- Phase C (S0b/S1b true fast_weights randomization): [not started]
- Phase D (closeout): [not started]

## Objective
Generalize the L96 S0/S1 per-window parameter randomization to per-parameter
control, enabling per-window randomization/corruption of `fast_weights`
(4 independent weights, 4 independent biases for S1). Keep the legacy
`randomize_params: bool` + shared-`param_bias` path intact so existing S0/S1
baselines reproduce exactly (repro gate) BEFORE enabling the new path.

## Decisions
| Date | Decision | Rationale |
|---|---|---|
| 2026-08-20 | New branch `feat/l96-fast-weights-randomization` from `feat/l96-neural-training` | isolated change |
| 2026-08-20 | Per-param `randomize` dict, legacy fields kept | repro safety |
| 2026-08-20 | `fast_weights` Dirac (noise=0) when not randomized | exact repro of fixed-weight S0/S1 |
| 2026-08-20 | 4 independent weights + 4 independent biases (S1b) | user-confirmed |
| 2026-08-20 | Repro gate tolerance 1e-3 relative | matches plan |
| 2026-08-20 | Repro gate: CPU 3-window smoke, then GPU 200-window | staged verification |
| 2026-08-20 | Keep S0/S1 naming; fast_weights-randomized variant = S0b/S1b | user-confirmed |
| 2026-08-20 | CI gate = pytest fast only; ruff lint informational | avoids blocking on 236 pre-existing ruff errors |
| 2026-08-20 | CI pytest gate scoped to 6 relevant test files | full `tests/` has pre-existing failures (broken test_numerical_equivalence.py API call, hardcoded-GPU tests, master failures) that would keep the gate red |

## Agent workflow (per-step iterative loop)

Every code change follows this cycle:

```
1. IMPLEMENTER (cortecs/deepseek-v4-flash)
   → makes the change, returns modified files + rationale

2. REVIEWER (opencode/big-pickle)
   → reads only the diff, checks correctness/edge cases/repro safety/style
   → returns PASS or ISSUES LIST

3. IF ISSUES → back to step 1 with reviewer feedback (max 2 fix rounds)

4. VERIFIER (cortecs/deepseek-v4-flash)
   → runs ruff check + pytest, returns PASS/FAIL
```

Agent assignment per step group:

| Group | Implementer | Reviewer | Verifier |
|-------|-------------|----------|----------|
| R1-R5 (review fixes) | cortecs/deepseek-v4-flash | opencode/big-pickle | cortecs/deepseek-v4-flash |
| A5-A7 (config threading) | cortecs/deepseek-v4-flash | opencode/big-pickle | cortecs/deepseek-v4-flash |
| B1-B4 (repro gate) | cortecs/deepseek-v4-flash (runner) | opencode/big-pickle (analyst) | — |
| C1-C4 (S0b/S1b) | cortecs/deepseek-v4-flash | opencode/big-pickle | cortecs/deepseek-v4-flash |

### Execution paths

Two execution paths per code step, both following the implement→review→verify loop:

**Option A — GitHub PR workflow** (requires `gh auth login` on the runner):
- Use `scripts/open_pr.sh <create|review|verify> ...` as the single wrapper:
  - `create R4 "desc"` → pushes the branch and opens a real PR
  - `review <PR#>` → reviewer reads `gh pr diff`, then `--approve` or
    `--request-changes`
  - `verify <PR#>` → `gh pr checks --watch` waits for CI, then `gh pr merge --squash`
- Underlying commands: `gh pr create --base feat/l96-*`, `gh pr review`,
  `gh pr checks`, `gh pr merge`
- Enforced by `.github/workflows/ci.yml` (pytest fast gate) + optional branch
  protection on `feat/l96-*`

**Option B — Local fallback** (works immediately, no GitHub):
- Use `scripts/agent_review_loop.sh <STEP> "<desc>"` to create a branch, then
  re-run with `--review` to commit → show diff → reviewer gate (y/n) →
  verifier (ruff + pytest) → squash merge
- Same loop as Option A, local git only


## Step tracker
| Step | Status | Agent | Notes |
|---|---|---|---|
| W1 GitHub Actions CI | ✅ | implementer | `.github/workflows/ci.yml` (ruff informational + pytest gate on `feat/l96-*`) |
| W2 local review loop script | ✅ | implementer | `scripts/agent_review_loop.sh` |
| W5 open_pr.sh helper | ✅ | implementer | `scripts/open_pr.sh` (create/review/verify gh wrapper for Option A) |
| W3 gh auth login | ✅ | user | `gh auth login` done (rfablet, repo+workflow scopes) |
| W4 branch protection | ✅ | user | ruleset `feat/l96-*: require PR review + CI` (1 approval + pytest check, active, no admin bypass) |
| R1 CHANGELOG header fix | ✅ | reviewer | `CHANGELOG.md` |
| R2 dead isinstance guard | ✅ | reviewer | `models/lorenz96_dynamics.py` |
| R3 _to_tensor_kw docstring | ✅ | reviewer | `evaluation/run_l96.py` |
| R4 da_J=None assertion | ✅ | reviewer | `evaluation/run_l96.py` (bug-fix from review) |
| R5 VanillaCFMConfig.train_tau_0_only | ✅ | reviewer | `conf/schema.py` (bug-fix from review) |
| A1 dynamics per-call fast_weights | ✅ | — | `models/lorenz96_dynamics.py` (list→tensor fixed) |
| A2 ParamRandomization + randomize dict | ✅ | — | `conf/schema.py` |
| A3 per-param draw + fast_weights list | ✅ | — | `data/lorenz96.py` (Bug 1 fixed) |
| A4 DA per-window fw + init fix + `_fw` cache | ⬜ | — | `evaluation/run_l96.py` (Bug 2 gating done; `_fw` cache suffix pending) |
| A5 train.py threading | ⬜ | — | `train.py` |
| A6 evaluate_all_l96 `--randomize` | ⬜ | — | `evaluate_all_l96.py` |
| A7 configs randomize block | ⬜ | — | `config/` |
| ✅ A8 tests (7 new) | ✅ | — | `tests/test_lorenz96_training.py` (32 pass) |
| A9 ruff + pytest | ✅ | — | only pre-existing E401/F841 remain |
| B1 CPU smoke repro | ⬜ | — | |
| B2 GPU full repro gate | ⬜ | — | diff vs existing cache |
| C1 S0b/S1b configs | ⬜ | — | `config/experiment/L{1,2}b_*` |
| C2 S0b/S1b tests | ⬜ | — | |
| C3 S0b/S1b DA run | ⬜ | — | `_fw` cache |
| D1 docs + changelog + progress | ⬜ | — | |

## Repro gate (Phase B)
- Referenced cache: `experiments/l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2.json`
- Tolerance: 1e-3 relative
- Legacy S0 EnKF/ETKF/Strong RMSE+EV: expected / got / delta
- Legacy S1 EnKF/ETKF/Strong RMSE+EV: expected / got / delta
- Verdict: [pending/passed/failed]

## Notes
- Existing S1 uses ONE shared per-window bias `b` for all 5 params
  (`params_da = params_true·(1+b)`). The new per-param dict adds independent
  per-param bias as an OPT-IN path; the legacy shared-bias path is preserved
  for exact repro of current results.
