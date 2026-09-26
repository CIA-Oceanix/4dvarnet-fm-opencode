# First generated QG spectral-wind dataset (demo instance of the G2 spec)

**Status:** RESULTS (2026-09-26). The first dataset produced by the G2
framework of `docs/plans/tech/qg_batched_generation_datasets.md`: the spec
`qg_specwind_gyrostat_demo`, which has the same factor ranges and protocol as
the production spec `qg_specwind_gyrostat_v1` at 256/64/64 windows.

The production 5000/500/500 run was generated later the same day; see the
"Production run" section below. It is stored on a server-local disk, not on `/Odyssey`.

## Generation

| split | windows | time (one RTX 8000, batch 256) | on disk |
|---|---|---|---|
| train | 256 | 96.5 s | 0.69 GB (12-hourly frames) |
| val | 64 | 57.0 s | 0.17 GB |
| test | 64 | 58.7 s | 1.01 GB (every step; read-only) |

Each window has its own forced 2-year spin-up (730 d), a 10-day lead and a
30-day window: 9240 steps. The runs also include:
- the gyrostat wind (`l63ring4`, a per-window 50-unit burn-in and time
  unit drawn in 30–90 d);
- the relative-wind eddy drag `r_cf` = 2.8×10⁻⁸ s⁻¹;
- per-window `rd`, `U1`, `rek` (±15%), level (20% calm stratum), drift
  and start phase from a Latin hypercube.

The 256-window train split runs at 0.38 s per window, matching the G1
benchmark. The small splits are dominated by the ~30 s per batch of wind
generation.

## Independence (all checks pass)

| pair | seeds disjoint | state hashes disjoint | no duplicate factors | NN not closer | start states not duplicated |
|---|---|---|---|---|---|
| train vs val | ✓ | ✓ | ✓ | ✓ (0% below train's 1% NN quantile) | ✓ (max \|corr\| 0.57) |
| train vs test | ✓ | ✓ | ✓ | ✓ (1.6%) | ✓ (0.66) |
| val vs test | ✓ | ✓ | ✓ | ✓ (3.1%) | ✓ (0.52) |

- **Nearest-neighbour distances:** the median distance from val and test
  windows to train (0.602 / 0.609) equals train's own median (0.604), as
  expected for independent samples of one design.
- **Leak detection:** `tests/test_qg_datasets.py` shows that a leaked
  window fails four of the five checks.

## Diversity (train and val)

- **Calm fraction:** 19.9% (train), 20.3% (val), against a target of 20%.
- **Val against train:** KS p-values of 0.97–1.00 for every factor, the
  rms curl and the upper-layer KE.

**Gyrostat regime occupancy is strongly non-uniform.** With the ring
coupling ε = 0.5, the four copies share a lobe (codes 0 and 15) in about
50% of windows (train 23% + 27%). The alternating patterns 5 and 10 are
nearly absent (0–1.6%). 14 of the 16 codes occur in train, and 15 across
train and val.

The coupling therefore synchronizes the copies' regimes. This bears on
decision 3 (ring coupling as a diversity factor, pending its attractor
constants):
- varying ε would change the regime mix;
- stratified selection would be needed if the rare codes must be present.

## Page

An overview page with the independence checks, distributions, regime
occupancy, factor coverage and five sample animations (train and val only;
the test split is withheld by protocol) is produced by
`reports/qg/visualize_qg_dataset.py`.

## Production run: `qg_specwind_gyrostat_v1` (5000/500/500)

Generated on 2026-09-26, 14:04 → 14:46, on one RTX 8000 (server
`sl-mee-br-202`), from commit `16a38f8`. It ran as 24 sequential shards of
250 windows, at 95 s per shard (104 s for test shards, which keep every
step).

| split | windows | time | on disk |
|---|---|---|---|
| train | 5000 | 1882.9 s | 13 GB |
| val | 500 | 188.1 s | 1.3 GB |
| test | 500 | 196.4 s | 7.4 GB (read-only) |

**Total: 37.8 min, 21.7 GB.** Location:
`/SCRATCH/rfablet/qg_datasets/qg_specwind_gyrostat_v1` on `sl-mee-br-202`.
This is the server's local scratch disk (the user asked for server-local
temporary storage; its `/tmp` had only 6.5 GB free). It is not backed up
and not visible from other nodes. It regenerates in about 40 min from this
spec and commit if lost.

**Independence: all checks pass.**

| pair | NN distance, other → ref (median) | within ref (median) | below ref 1% quantile | max start-state \|corr\| |
|---|---|---|---|---|
| train vs val | 0.418 | 0.422 | 0.6% | 0.73 |
| train vs test | 0.421 | 0.422 | 0.6% | 0.78 |
| val vs test | 0.554 | 0.564 | 0.8% | 0.69 |

**Diversity:**
- calm fraction 20.0% in both train and val;
- design discrepancy 7.4×10⁻⁴ (train);
- val against train: KS p ≥ 0.86 for every factor, the rms curl and the
  KE.

**Regime occupancy confirms the synchronization finding at scale.** All 16
codes occur in train, but codes 0 + 15 hold 52% of windows (54% in val).
The alternating codes 5 and 10 occur in 2 and 4 of 5000 train windows
(0.04% and 0.08%) and in none of the 500 val windows.
