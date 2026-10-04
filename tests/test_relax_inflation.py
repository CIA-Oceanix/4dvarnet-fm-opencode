import pytest
import torch

from evaluation.baselines import EnKF, EnKS, ETKF, _relax_anomalies


def _pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ens_f = torch.randn(40, 12, generator=g) * 2.0
    ens_a = 0.3 * ens_f + torch.randn(40, 12, generator=g) * 0.2 + 1.0
    return ens_f - ens_f.mean(0), ens_a


@pytest.mark.parametrize("mode", ["rtpp", "rtps"])
def test_relaxation_keeps_the_mean_and_is_identity_at_alpha_zero(mode):
    A_f, ens_a = _pair()
    assert torch.equal(_relax_anomalies(ens_a, A_f, mode, 0.0), ens_a)
    out = _relax_anomalies(ens_a, A_f, mode, 0.6)
    assert torch.allclose(out.mean(0), ens_a.mean(0), atol=1e-5)


def test_rtpp_at_alpha_one_restores_the_forecast_anomalies():
    A_f, ens_a = _pair()
    out = _relax_anomalies(ens_a, A_f, "rtpp", 1.0)
    assert torch.allclose(out - out.mean(0), A_f, atol=1e-5)


def test_rtps_relaxes_each_variable_spread_toward_the_forecast_spread():
    A_f, ens_a = _pair()
    s_a, s_f = ens_a.std(0), A_f.std(0)
    for alpha in (0.5, 1.0):
        out = _relax_anomalies(ens_a, A_f, "rtps", alpha)
        assert torch.allclose(out.std(0), s_a + alpha * (s_f - s_a), rtol=1e-4)


def test_rtps_leaves_a_collapsed_variable_unchanged():
    A_f, ens_a = _pair()
    ens_a[:, 3] = 5.0
    out = _relax_anomalies(ens_a, A_f, "rtps", 0.5)
    assert torch.all(out[:, 3] == 5.0)


def test_relaxation_mode_is_validated_and_unsupported_smoothers_refuse_it():
    with pytest.raises(ValueError):
        ETKF(N_ensemble=10, relax="rtpx", relax_alpha=0.5)
    with pytest.raises(ValueError):
        EnKF(N_ensemble=10, relax="bogus", relax_alpha=0.5)
    with pytest.raises(NotImplementedError):
        EnKS(N_ensemble=10, relax="rtps", relax_alpha=0.5)
