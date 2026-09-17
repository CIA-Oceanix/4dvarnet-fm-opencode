"""Regression tests for the L96 FM-sampler report generator.

The conclusions section once carried a hardcoded "beats Strong-4DVar by 31%"
that reconciled with no pairing in the report's own tables (the true figures are
34.8% on `rmse_repo` and 29.9% on `rmse_pooled`). These tests pin the margin to
something computed from the data.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "reports/l96/generate_l96_fm_sampler_report.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("l96_fm_sampler_report", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def full():
    """The §5.1 full-test-set row, as the driver writes it."""
    return {"windows_done": 200,
            "results": {"mean-only": {"rmse_repo": 0.4714},
                        "cold(lam0=0)": {"rmse_repo": 0.5295}}}


def test_margin_is_computed_from_the_data(gen, full):
    c = gen.cold_vs_baseline(full)
    assert c["baseline"] == "Strong-4DVar"
    assert c["ref"] == pytest.approx(0.8116)
    assert c["cold"] == pytest.approx(0.5295)
    assert c["margin"] == pytest.approx(34.76, abs=0.01)


def test_margin_tracks_the_data(gen, full):
    """Change the cold RMSE and the margin must move with it."""
    full["results"]["cold(lam0=0)"]["rmse_repo"] = 0.8116
    assert gen.cold_vs_baseline(full)["margin"] == pytest.approx(0.0)


def test_conclusions_state_the_margin_and_its_operands(gen, full):
    text = gen.build_conclusions(full)
    assert "35%" in text
    assert "0.5295" in text and "0.8116" in text
    assert "200 windows" in text
    assert "31%" not in text


def test_conclusions_degrade_without_data(gen):
    text = gen.build_conclusions(None)
    assert "margin not computed" in text
    assert "%" not in text.split("* It is")[0]


def test_checked_in_report_matches_the_generator(gen, full):
    """Generator and its committed output must not drift apart."""
    bullet = gen.build_conclusions(full).split("* It is")[0].strip()
    report = (ROOT / "reports/l96/outputs/l96_fm_sampler_benchmark.md").read_text()
    body = bullet.split("\n", 1)[1].strip()
    assert body in report
