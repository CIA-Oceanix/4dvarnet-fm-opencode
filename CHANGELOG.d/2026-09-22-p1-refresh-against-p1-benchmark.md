## 2026-09-22: Refresh P1 §5 against the dedicated P1 benchmark — and downgrade C4

**Summary:** #235 added `reports/l96/outputs/p1_l96_benchmark.md`, a dedicated P1
benchmark at a P1-specific protocol covering exactly this paper's scheme set.
Every §5 table is refreshed against it. Two claims changed direction, and one is
downgraded from a claim to an open question.

**Numbers, all superseded:**

| | was | now |
|---|---|---|
| Strong-4DVar S0 / S1 | 0.8116 / 1.4617 (1.801x) | **0.7383 / 1.4369 (1.946x)** |
| ETKF | 0.8883 / 1.4998 | 0.8662 / 1.4748 |
| EnKF | 0.9131 / 1.5381 | 0.8943 / 1.5123 |
| best learned | DirectUNet+SDA3 0.4204 | **VanillaCFM-M 0.3445** |

Strong-4DVar's degradation ratio *rose* to 1.95x, which strengthens C1.

**Change 1 — the conditioning ladder inverted.** Old: SDA1 0.5532, SDA2 0.5588,
SDA3 0.5365, i.e. noisy conditioning 3.0% *better* than none. New: SDA1 0.5063,
SDA2 0.5129, SDA3 0.5228 -- noisy conditioning 3.3% *worse*. The **inertness
conclusion is unchanged and if anything stronger** (both conditioned variants now
lose to the unconditional prior), but the draft's sub-claim that noisy
conditioning reliably helps is gone. A `\caveat` now says the sign-flip is itself
the finding: a ~3% effect is smaller than the 15-21% checkpoint-selection noise
the benchmark measures for this family, so no directional claim is warranted.

**Change 2 — C4 is not supported as stated, and is downgraded.** The new
benchmark supplies the ensemble-Kalman calibration column that was missing, and
it points the other way:

| scheme | spread/RMSE (target 0.967) |
|---|---|
| ETKF | **1.127** |
| EnKF | **1.089** |
| VanillaCFM-M | 0.535 |
| SDA1-M | 0.416 |
| PredictStateCFM-M | 0.400 |

The class that zeroes `Psi_NG` is the **best**-calibrated; every scheme with a
full non-Gaussian branch is under-dispersed by 2x or more. §5.5 is rewritten
around this: the accuracy/dispersion trade is real and stark, but runs opposite
to what the operator partition suggests.

The reconciliation offered is that **spread/RMSE is a second-moment diagnostic
and `Psi_NG` is not a second-moment object** -- a Gaussian update can match the
spread exactly and still be wrong about the posterior's shape. That is coherent
but untested, because rank histograms do not exist in this codebase. So C4 is
downgraded to an open question rather than defended on the construction argument
alone, and S-b now rests on the trade being unresolved by *every* class rather
than on a deficit specific to one.

**Blending is demoted to a within-family result.** The new benchmark has no
DirectUNet+SDA rows, so the blend numbers (from the sampler study's own
configuration) are no longer comparable with the main table and are reported as
such.

**Verification:** `pdflatex` x3 -- 16 pages, 0 undefined references; every `\ref`
checked against a matching `\label`, none dangling; swept for the 15 superseded
values, none remain outside the explicit caveat.
