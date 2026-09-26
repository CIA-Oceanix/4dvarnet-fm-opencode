# Two-way coupled QG dataset (`qg_coupled_gyrostat_v1`, train and val)

**Status:** RESULTS (2026-09-26). The first dataset from the Option C
coupled system (`models/qg_coupled.py`), at the physical coupling strength
κ_fb = 1: a two-layer QG ocean coupled both ways to the `l63ring4`
gyrostat atmosphere. It is the configuration of this case study closest to a
real coupled reanalysis. Context: `docs/plans/tech/qg_batched_generation_datasets.md`
(decision 4, §8.1).

## Spec and pairing

`qg_coupled_gyrostat_v1` has the same factor box, seed entropies, factor
design and window seeds as the forced dataset `qg_specwind_gyrostat_v1`
(`docs/results/qg_specwind_demo_dataset.md`):
- **Shared by window i of both datasets:** ocean parameters, drift, phase,
  wind level, gyrostat time unit, and the ocean and gyrostat initial
  states.
- **What differs:** the forced dataset already includes the relative-wind
  eddy drag (`r_cf` = 2.8×10⁻⁸ s⁻¹), so the two differ **only by the
  ocean → atmosphere feedback**.
- **Spin-up:** each coupled window has its own coupled two-year spin-up.

`kappa_fb` is part of the split hash only for the coupled driver, so the
forced spec's stored hashes are unchanged (checked against the v1
manifest).

## Generation

On 2026-09-26, 17:01 → 17:47, on one RTX 8000 (`sl-mee-br-202`), commit
`423dfac`. It ran as 22 shards of 250 windows at 120 s each.

| split | windows | time | on disk |
|---|---|---|---|
| train | 5000 | 2412.6 s | 13 GB |
| val | 500 | 235.9 s | 1.3 GB |

That is 0.48 s per window, 1.26× the forced dataset (0.38 s). Location:
`/SCRATCH/rfablet/qg_datasets/qg_coupled_gyrostat_v1`, on the server's
local, un-backed-up scratch disk.

**No test split yet**, as the request was for train and val. It generates in
about 4 min with the test entropy.

## Independence and diversity

| check (train vs val) | result |
|---|---|
| seeds, state hashes, duplicate factors | all pass |
| NN distance val → train (median) / within train | 0.418 / 0.422 |
| val windows below train's 1% NN quantile | 0.6% |
| max start-state \|corr\| | 0.84 (< 0.99) |

- **Calm fraction:** 20.0% (train and val).
- **Val against train:** KS p ≥ 0.90 for every factor, the rms curl, the KE
  and the feedback ratio.
- **Regimes:** codes 0 + 15 hold 52% of train windows (50% of val). The
  ring's regime synchronization persists under coupling.

## Feedback strength (κ_fb = 1)

The per-window feedback ratio is the rms of the ocean → atmosphere term over
the rms of the gyrostat's own tendency, averaged over the lead and window.
On windy windows it is:
- median **1.2%**;
- 10th–90th percentile 0.6%–3.7%;
- maximum 147%.

The maximum occurs in weak-wind windows, where the feedback is divided by a
small wind scale `s_k`. This matches decision 4's estimate (0.2–3.5%) and
the G3 benchmark (0.75% median over 60 days from an unforced spin-up).

## Paired comparison with the forced dataset (train, 5000 windows)

| quantity | value |
|---|---|
| window-mean upper-layer KE, coupled / forced: median | **1.00** |
| 10th–90th percentile | 0.73–1.42 |
| correlation of KE across windows | 0.80 |
| correlation of rms wind curl across windows | 0.96 |

**At the physical coupling strength, the feedback leaves the ocean's energy
level unchanged on average.** The window-to-window scatter comes from the
chaotic eddy field, which decorrelates two runs that differ by a small
perturbation. Separating a forced response from that scatter needs
ensembles with shared initial conditions (the Option B plan's WP2).

## Page

`reports/qg/visualize_qg_dataset.py --spec qg_coupled_gyrostat_v1
--pair-spec qg_specwind_gyrostat_v1` produces the data for an overview page
with:
- independence checks;
- distributions, including the feedback ratio;
- regime occupancy and factor coverage;
- the paired KE scatter on log–log axes;
- sample windows.
