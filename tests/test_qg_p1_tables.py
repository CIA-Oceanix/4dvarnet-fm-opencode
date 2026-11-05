import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gen", os.path.join(ROOT, "reports/qg/generate_qg_p1_tables.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

CLIM = {"crps_clim": {"psi1": 2.0, "psi2": 4.0}, "mean_std": {"psi1": {"mean": 0.0, "std": 2.0},
                                                              "psi2": {"mean": 0.0, "std": 4.0}}}


def gaussian(e2: float, v: float, tau: float) -> float:
    a = gen.alpha(tau)
    return (e2 + a * v ** 2) / (1 + a * v) ** 2


def window(i: int, e2: float, v: float) -> dict:
    w = {"index": i}
    for f in gen.PSI:
        sd = CLIM["mean_std"][f]["std"]
        w[f"rmse_da_{f}"] = np.sqrt(e2) * sd
        w[f"ev_da_{f}"] = 0.5
        w[f"crps_{f}"] = 0.5 * CLIM["crps_clim"][f]
        w[f"ss_{f}_var"] = 10 * v
        w[f"ss_{f}_err2"] = 10 * e2
        w[f"fm_{f}_n"] = 10
        for t in gen.TAUS:
            w[f"fm_{f}_gauss_{t}"] = gaussian(e2, v, t)
            w[f"fm_{f}_gc_{t}"] = [gaussian(e2, c * v, t) for c in gen.CS]
            w[f"fm_{f}_kde_{t}"] = gaussian(e2, v, t)
            w[f"fm_{f}_kdevar_{t}"] = gaussian(e2, v, t)
    return w


def test_table_row_uses_climatological_ev_and_pooled_ss():
    ws = [window(0, 0.1, 0.025), window(1, 0.3, 0.075)]
    r = gen.table_row(ws, "ensemble", CLIM)
    assert r["ev_psi"] == pytest.approx(0.8)
    assert r["ev_window_psi"] == pytest.approx(0.5)
    assert r["crpss_psi"] == pytest.approx(0.5)
    assert r["ss_psi"] == pytest.approx(0.5)


def test_deterministic_fmss_is_ev_at_every_tau_scale():
    ws = [window(0, 0.2, 0.0), window(1, 0.2, 0.0)]
    r = gen.table_row(ws, "deterministic", CLIM)
    s = gen.series_row(ws, "deterministic", CLIM)
    assert s["fmss"][0] == pytest.approx(r["ev_psi"])
    assert s["ss"] is None


def test_series_finds_the_calibrating_inflation():
    ws = [window(0, 0.2, 0.05), window(1, 0.2, 0.05)]
    s = gen.series_row(ws, "ensemble", CLIM)
    assert s["fmss"][0] == pytest.approx(0.8)
    for t, ss, edge, loss in zip(gen.TAUS[1:], s["ss"][1:], s["ss_edge"][1:], s["loss"][1:]):
        assert not edge and loss > 0
        assert ss == pytest.approx(0.5, rel=0.1)
    assert s["g"][1:] == pytest.approx([0.0] * 3)


def test_rows_cover_both_cases(tmp_path):
    assert set(gen.ROWS) == {"s0", "s1"}
    committed = os.path.join(ROOT, "reports/qg/outputs/qg_p1/qg_p1_rows.json")
    if os.path.exists(committed):
        rows = json.load(open(committed))["rows"]
        for case, schemes in gen.ROWS.items():
            assert set(rows[case]) == set(schemes)
