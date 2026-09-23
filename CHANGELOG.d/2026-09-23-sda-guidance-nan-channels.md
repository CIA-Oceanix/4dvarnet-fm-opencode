## 2026-09-23: SDA guidance cost no longer zero-imputes missing channels

**Summary:** `evaluation/sda_sampler.py::guided_obs_cost` masked the guidance cost only by obs TIME: NaN channels inside an observed row were `nan_to_num`'d to 0 and scored as real observations of 0. It now also excludes every NaN entry of `y`.
**Files modified:** `evaluation/sda_sampler.py` — NaN mask in `guided_obs_cost`; `tests/test_sda_sampler.py` — regression test (fails on the old cost).
**Rationale:** Invisible on the regular P1 grid (every observed row has all 24 channels), so P1 SDA scores are unchanged; the obs-density study passed `obs_channel_mask` explicitly and was unaffected. But any partially observed test set (the random observing system: 8 of 16 fast channels per obs time) scored SDA against fake zero obs -- SDA1-M S0 RMSE 1.029 vs 0.947 on a 4-window check.
**Verification:** `pytest tests/test_sda_sampler.py tests/test_eval_sda_l96.py tests/test_sda.py -m "not slow"` — 32 passed; new test fails without the fix; 4-window SDA1-M guided eval on the random 30-obs / 8-fast test set runs end to end.
