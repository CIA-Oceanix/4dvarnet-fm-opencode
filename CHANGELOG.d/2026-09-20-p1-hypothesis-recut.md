## 2026-09-20: P1 scoping — re-cut the hypothesis set (H1 Markovianity, H3a/H3b split, H4 named)

**Summary:** Revised `da_paper_structural_hypotheses.md` from SCOPING v1 to v2.
H1 is restated as **Markovianity** (the joint problem factorizes into a sequence
of simpler ones) rather than the weaker "sequential"; H3 is split into **H3a**
(ODE/PDE state representation) and **H3b** (autoregressive error propagation),
which v1 conflated by filing §4.1.2's model-vs-posterior sensitivity argument as
a corollary; a fourth hypothesis **H4** (the value of gradients/adjoints) is
named, with only its classical corollary **C7** retained here and the
constructive half assigned to P2. No evidence was added or retracted — C2/C3/C4/C6
are re-tagged to the split hypotheses, C7 is new, and §4.2.1 is a new reading of
the existing §4.2 table.

**Files modified:**
- `docs/scoping/da_paper_structural_hypotheses.md` — title/status to v2 + a
  "what changed" box; §1 hypothesis statements and the §2 table rebuilt around
  H1/H2/H3a/H3b with an H4 row marked as P2-owned; §4.1.2/§4.3/§4.4/§4.5
  re-tagged; new **§4.2.1** (C7); §5 claims re-tagged and C7 added; **§6 D0**
  status rewritten; §8 P0 rewritten; new §10 decision 6.
- `docs/scoping/README.md` — index entry for the P1 doc updated to v2.

**Rationale:** Requested re-scoping of the first DA paper against the underlying
hypotheses of DA schemes. Three of the four proposed axes already existed under
other names; the split of H3 and the naming of H4 are the substantive changes.
C7 is stated as a claim about *marginal value* (19.7%→3.2% for Strong-4DVar, a
6.2× collapse, against 1.9× for the gradient-free filters) and explicitly not
about levels, since Strong-4DVar remains the best classical method at every
density in that config. All C7 figures come from the single config of §4.2, so
R3's no-cross-quoting rule is respected.

**D0 status corrected.** v1 recorded weak-4D-Var as unimplemented-in-practice.
Open PR #202 adds `L96Weak4DVar` (QG4DVar-style whitened controls, NaN-reset) +
14 tests and identifies Adam — not the weak-constraint parameterization — as the
failure mode, with LBFGS resolving it. §6/§8 now say the remaining P0 is landing
that PR and running the S0/S1 benchmark. The retracted ~1.52 RMSE figure is
flagged as not-to-be-cited (corrected value ~0.98, `seed=123`, 10 windows).

**Verification:** `pytest tests/ -q -m "not slow"` (fdv-monai-proto) — 851
passed, 2 skipped, 7 failed in 20m36s; all 7 failures are
`tests/test_joint_estimation.py::TestJointCFM`, which `.github/workflows/ci.yml`
deselects by name as a known pre-restructure-API gap, and this change touches no
`.py` file. `pytest tests/test_l96_report_consistency.py
tests/test_l96_fm_sampler_report.py -q -m "not slow"` — 19 passed (the guards
that cover the report values these docs cite). `ruff check .` — 14 pre-existing
errors, none in a file touched here (`evaluation/neural_inference.py`,
`reports/l96/generate_l96_grad_checkpoint_benchmark.py`,
`reports/qg/generate_qg_reconstruction_figs.py`, two test modules).
