import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_quick_quality_pack(tmp_path):
    script = os.path.join(ROOT, "reports", "diagnose_sw_quality.py")
    outdir = str(tmp_path / "sw_quality")
    cmd = [sys.executable, script, "--quick", "--outdir", outdir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Stability finite=True" in res.stdout

    expected = [
        "snapshots.png", "vorticity.png", "spectra.png", "hovmoeller.png",
        "stability.png", "animation_h1.gif", "animation_zeta1.gif",
        "animation_h2.gif", "sw_quality_report.md",
    ]
    for f in expected:
        path = os.path.join(outdir, f)
        assert os.path.isfile(path), f"{f} missing"
        assert os.path.getsize(path) > 0, f"{f} is empty"

    with open(os.path.join(outdir, "sw_quality_report.md")) as fh:
        md = fh.read()
    assert "Rd1" in md and "Rd2" in md
    assert "checklist" in md
