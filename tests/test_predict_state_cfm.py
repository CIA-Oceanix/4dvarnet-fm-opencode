"""Tests for V3 PredictStateCFM (CFM parameterized by E[x1|xt,y] instead of E[x1-x0|xt,y])."""
import torch
import torch.nn.functional as F


def test_predict_state_cfm_differs_from_vanilla():
    """Verify this is a fundamentally different formulation."""
    # Both should produce valid outputs
    # But the network predictions should differ for the same input
    print("Implementation placeholder — will be added when V3 is implemented")
