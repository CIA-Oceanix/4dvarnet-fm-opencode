## 2026-09-21: Add the planned learned-side representation-swap table to P1

**Summary:** §5.4 gains `tab:representation-planned`, an empty-celled table for
the missing NN-side representation experiment, laid out to mirror
`tab:representation` so the two read side by side.

**The design is a 2x2, not a straight swap, and reading the code is why.**
`train_qg_neural.py`'s supervision target is the daily-mean two-layer
streamfunction, and `q` is obtained by spectrally inverting the predicted `psi`
-- structurally the same path as the DA `psi_state` variant, and subject to the
same `K^2` amplification. So **the existing network is already one arm of the
experiment**, and two of the table's cells are filled from it (Q1: psi EV 0.9805,
q EV 0.6238, identical across S0/S1 by construction since that configuration is
obs-only).

The complication is the optional auxiliary q-loss (weight `1/Var(q)`), which
supervises the inverted field directly and therefore partially decouples *output
space* from *what is supervised*. Varying only the output space would confound
the two, so the table crosses output space (psi / q) with the auxiliary loss
(on / off) at both scenarios -- six runs.

**A prediction is stated in advance**, which the table is designed to
discriminate:
- If the effect is a property of inverse problems generally, the q-output arm
  beats the psi-output arm on q, loses on psi, and the gap widens from S0 to S1
  as it does for DA.
- If instead psi-output *with* a q-loss matches q-output, the operative choice is
  what is **supervised**, not what is **represented**, and H3a is weaker than the
  DA result suggests.

**The existing numbers already hint at the second reading**, and the draft says
so: the psi-output network reaches q EV 0.62-0.67 where the DA psi-state variant
collapses to 0.583 and then -2.925, and the one structural difference between
them is that q-loss. Writing the prediction down now is what makes the experiment
informative either way rather than a confirmation exercise.

**A spread/skill column is included** so the swap is also read against §5.5: if
changing the output space moves calibration as well as accuracy, representation
is not separable from the dispersion question.

**Verification:** `pdflatex` x3 -- 16 pages, 0 undefined references or citations.
