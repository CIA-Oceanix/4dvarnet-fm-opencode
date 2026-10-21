import importlib.util
import json
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "qg_specwind_da_report", os.path.join(os.path.dirname(__file__), "..", "reports", "qg",
                                          "generate_qg_specwind_da_report.py"))
rep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rep)


def _write(root, name, meta, scores):
    d = root / rep.NEURAL_SPEC / "test" / name
    d.mkdir(parents=True)
    per_window = [{"index": i, "crps": 1e-6, **{f"ev_da_{f}": s for f in rep.FIELDS}}
                  for i, s in enumerate(scores)]
    (d / "shard_0_of_1_meta.json").write_text(json.dumps({**meta, "per_window": per_window}))


def _meta(cols=3, noise=0.05, s1=None):
    return {"spec": rep.NEURAL_SPEC, "method": "neural_direct_unet_3d", "cols_per_day": cols,
            "obs_noise_frac": noise, "s1_variant": s1}


def test_neural_rows_are_found_by_config_and_skipped_by_the_da_loader(tmp_path):
    cfg = {c[0]: c for c in rep.OBS_CONFIGS}
    scheme = rep.NEURAL_SCHEMES[2]
    _write(tmp_path, "neural_G1L3D_b16_w15_kpow_s0_n15_c3_cubic_N1", _meta(noise=0.15), [0.5] * 100)
    _write(tmp_path, "neural_G1L3D_b16_w15_kpow_s0_c3_cubic_N1_k1_rbase", _meta(s1="base"), [0.4] * 100)
    neural = rep.load_neural(str(tmp_path))
    assert rep._neural_run(neural, scheme, cfg["n15"])["per_window"][0]["ev_da_q1"] == 0.5
    assert rep._neural_run(neural, scheme, cfg["s1"])["per_window"][0]["ev_da_q1"] == 0.4
    assert rep._neural_run(neural, scheme, cfg["c3"]) is None
    assert rep.load_runs(str(tmp_path)) == []


def test_a_row_whose_meta_contradicts_its_name_is_rejected(tmp_path):
    _write(tmp_path, "neural_G1L3D_b16_w15_kpow_s0_n02_c3_cubic_N1", _meta(noise=0.05), [0.5] * 100)
    cfg = {c[0]: c for c in rep.OBS_CONFIGS}["n02"]
    with pytest.raises(ValueError, match="does not match"):
        rep._neural_run(rep.load_neural(str(tmp_path)), rep.NEURAL_SCHEMES[2], cfg)


def test_paired_difference_uses_common_windows():
    a = {"per_window": [{"index": i, **{f"ev_da_{f}": 0.9 for f in rep.FIELDS}} for i in range(10)]}
    b = {"per_window": [{"index": i, **{f"ev_da_{f}": 0.8 for f in rep.FIELDS}} for i in range(5, 15)]}
    m, lo, hi = rep._paired(a, b)
    assert m == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_rank_marks_bold_best_and_italic_second():
    assert rep._mark([0.8, 0.9, None, 0.85]) == ["0.800", "**0.900**", "—", "*0.850*"]
    assert rep._mark([3.0, 2.0, 4.0], lower_better=True) == ["*3.000*", "**2.000**", "4.000"]
