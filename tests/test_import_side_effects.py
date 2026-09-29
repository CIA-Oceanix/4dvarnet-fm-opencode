"""Importing project modules must not change process-wide torch numerics.

``train.py`` used to call ``torch.set_float32_matmul_precision('medium')`` at
import time. Every test module or script importing it (e.g. for
``model_factory``) then ran float32 matmuls in bfloat16 wherever the CPU has a
fast bf16 path. That made the exact linear-Gaussian ETKS check intermittently
fail on CI runners with such CPUs (max error 1.8e-3 against 2e-4).
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", ["train", "evaluation.neural_inference"])
def test_import_keeps_default_float32_matmul_precision(module):
    code = (f"import torch; before = torch.get_float32_matmul_precision(); import {module}; "
            "after = torch.get_float32_matmul_precision(); "
            "assert after == before, (before, after); print(after)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
