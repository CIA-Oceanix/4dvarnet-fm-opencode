"""Rendering of the P1 report's benchmark-framework sections (reports/l96/p1_benchmark_sections.py)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "l96"))
import p1_benchmark_sections as p1s  # noqa: E402


def _row(label: str, level: float, flag: str = "") -> dict:
    cell = {c: {"rmse": np.full(4, level) + np.arange(4) * 0.01, "crps": level / 2, "sp": 0.6,
                "seed_means": [level, level + 0.002, level - 0.002]} for c in p1s.CASES}
    return {"label": label, "params": "p", "flag": flag, "seeds": "3/3", "reg": cell, "can": cell, "vr": None}


def test_best_unflagged_row_is_bold_and_flagged_rows_are_italic():
    lines = p1s.table("fm", [_row("A", 0.40), _row("B", 0.35), _row("C", 0.30, flag="sensitivity")])
    body = {ln.split(" | ")[0].strip("| "): ln for ln in lines if ln.startswith("| ") and "---" not in ln}
    assert "**" in body["B"] and "**" not in body["A"]
    assert body["*C*"].endswith("| sensitivity |") and "**" not in body["*C*"]


def test_pending_rows_render_without_metrics():
    r = _row("A", 0.4)
    r["can"] = None
    assert any("pending" in ln for ln in p1s.table("det", [r]))


def test_tau0_rows_average_seeds_per_family(tmp_path, monkeypatch):
    monkeypatch.setattr(p1s, "TAU0", tmp_path)
    for _, name in p1s.TAU0_SETS:
        for c in p1s.CASES:
            rows = [{"label": f"PredictStateCFM M seed{s}", "rmse": {"mean": 0.4 + s / 100, "std": 0.1},
                     "crps_mae": {"mean": 0.3}, "draw_dispersion": 0.05} for s in (1, 2, 3)]
            (tmp_path / f"tau0_{name}_{c}.json").write_text(json.dumps({"rows": rows}))
    out = p1s.tau0_rows()
    assert [r["label"] for r in out] == ["PredictStateCFM-M (tau=0)"]
    assert np.isclose(np.mean(out[0]["rmse"][("reg", "s0")]), 0.42)


def test_tau0_rows_pending_when_a_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(p1s, "TAU0", tmp_path)
    assert p1s.tau0_rows() is None
