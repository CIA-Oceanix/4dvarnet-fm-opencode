## 2026-09-24: Scoping note for a coupled ocean–atmosphere case study (coupled gyrostats + MAOOAM)

**Summary:** Adds `docs/scoping/cgoa_coupled_gyrostat_case_study.md`, which
proposes a fourth case study after L63, L96 and QG. It has two tiers on one
torch quadratic-tensor engine:
- **Tier A:** an energy-consistent coupled-gyrostat model. The slow ocean
  is carried by inertia, and the air–sea coupling has three switchable
  mechanisms: conservative cross triads, shared interface modes, and
  dissipative drag.
- **Tier B:** MAOOAM, with its tensor exported once from `Climdyn/qgs`.

**Files modified:** `docs/scoping/cgoa_coupled_gyrostat_case_study.md` —
new. `docs/scoping/README.md` — index row.
**Rationale:** Coupled DA adds two things L96 and QG lack: time scales that
differ by orders of magnitude, and very unequal observation of the two
components. Changing the coupling structure gives a clean structural model
error. The note surveys the gyrostat theory (Obukhov, Gluhovsky,
Seshadri–Lakshmivarahan), the reusable code (qgs, MAOOAM, DAPPER) and the
coupled-DA benchmarks (Tondeur et al. 2020, Garcia-Oliva et al. 2025). It
also argues for building the case study in this repo rather than a new one.
Nothing is implemented. Paper fit, tier order and the mode-space backbone are
left as open decisions.

**v2 (same PR):** §5 no longer mirrors the L96 stack file for file. CGOA adds
only case-specific files (a generic quadratic-tensor engine, a gyrostat
builder that exposes `component_slices`, and a `data/cgoa.py` that honours the
existing window-dict contract). The shared pieces it needs are extracted in one
no-op PR (new WP3): `L96Weak4DVar` → `WhitenedWeak4DVar` with a
per-component whitening scale, grouped `evaluate_estimates`, SCDA/WCDA via
block localization on the existing `ETKF`/`EnKF` `loc_*_t` hook, a generic DA
loop, and a `cgoa` archive entry. v2 also corrects v1's claim that `Weak4DVar`
applies unchanged: on two-scale dynamics the plain class diverges, as it did on
L96. It adds an acceptance criterion (no `eval_*_cgoa.py`, no per-run sbatch,
no copied driver functions) and decision 5 (where the extraction lands).
**Verification:** Docs only. No code or tests touched.
