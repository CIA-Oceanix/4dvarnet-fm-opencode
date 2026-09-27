import torch

from eval_neural_l96 import default_n_outer
from models.fourdvarnet import FourDVarNetSolver


def test_unrolled_solver_defaults_to_its_own_n_outer():
    model = FourDVarNetSolver(state_dim=3, hidden_channels=[4, 8], N_outer=7)
    assert default_n_outer(model) == 7


def test_other_models_default_to_one_step():
    assert default_n_outer(torch.nn.Linear(2, 2)) == 1
