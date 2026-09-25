## 2026-09-25: Scoping note — gyrostat-driven synoptic forcing for the QG case study

**Summary:** Adds `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`,
which proposes replacing the QG case study's kinematic OU storm with a
low-order chaotic atmosphere built from coupled gyrostats, in order to study
how the upper ocean responds to synoptic variability. It lays out five
options:
- **A:** a one-way parametric driver (recommended first);
- **B:** a one-way spectral driver;
- **C:** two-way mechanical coupling through the relative wind;
- **D:** a gyrostat reduced model of the ocean;
- **E:** thermal coupling (deferred).

The note also gives a response-analysis protocol built around a
phase-randomized surrogate control, new DA scenarios (S1-gyro-param,
S1-gyro-struct), an implementation sketch for Option A, risks and open
decisions.
**Files modified:** `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`
— new. `docs/README.md` — index row.
**Rationale:** The QG ocean's wind forcing is Gaussian, one-way and slow, and
its S1 forcing error is unstructured noise. A dynamical low-order atmosphere
lets the case study ask how forcing structure, as opposed to forcing energy,
shapes the upper-ocean response. It also gives the P1 model-error axis a
structured forcing-error scenario. The note is written to be shared with
colleagues outside the repo. The QG benchmark default is unchanged.
**Verification:** Docs only. `pytest tests/test_docs_layout.py` passes.
