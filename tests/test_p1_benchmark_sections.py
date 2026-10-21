"""Rendering of the P1 report's benchmark-framework sections (reports/l96/p1_benchmark_sections.py)."""
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


def test_tau0_table_is_pending_for_the_2d_flows():
    lines = p1s.tau0_table(None, [])
    assert any("pending" in ln for ln in lines)


def test_note_column_carries_the_seed_note():
    r = _row("A", 0.4)
    r["seeds"], r["note"] = "1/3", p1s.SEED1
    line = next(ln for ln in p1s.table("fm", [r]) if ln.startswith("| A "))
    assert line.endswith(f"| {p1s.SEED1} |")


def test_reading_skips_pending_rows_and_names_seed_counts():
    one = _row("VanillaCFM-2D-M", 0.30)
    one["seeds"] = "1/3"
    pending = {**_row("Hybrid pending", 0.2, flag="ref"), "reg": None, "can": None}
    rows = {"da": [_row("ETKS", 0.50), _row("ETKS, 100 members", 0.45, flag="ensemble size")],
            "det": [_row("DirectUNet-2D-M", 0.35)], "fm": [one],
            "sda": [_row("SDA1-2D-M", 0.40), pending, _row("Hybrid DU -> SDA", 0.25, flag="ref")]}
    text = "\n".join(p1s.reading(rows))
    assert "VanillaCFM-2D-M" in text and "(1 seed)" in text and "(3 seeds)" in text
    assert "Hybrid pending" not in text and "Hybrid DU -> SDA" in text
