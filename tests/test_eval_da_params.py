import torch

from evaluation.neural_inference import _window_da_param_vector, collate_joint_eval

TRUE = {"F": 8.5, "c1": 1.2, "hx": 0.95, "eps": 0.105, "fast_weights": [1.07, 0.85, 0.09, 0.09]}
DA = {"F_da": 9.35, "c1_da": 1.32, "hx_da": 1.045, "eps_da": 0.1155, "fast_weights_da": [1.18, 0.94, 0.099, 0.099]}


def _window(biased: bool) -> dict:
    w = {"true_state": torch.zeros(5, 40), "obs": torch.zeros(5, 24), "obs_mask": torch.ones(5),
         "forcing_true": torch.zeros(5), "forcing_corrupted": torch.ones(5), "h": 1.0}
    w.update(TRUE)
    w.update({f"true_{k}": v for k, v in TRUE.items()})
    if biased:
        w.update(DA)
    return w


def test_s1_window_gives_biased_da_params() -> None:
    expected = [DA[f"{n}_da"] for n in ("F", "c1", "hx", "eps")] + DA["fast_weights_da"]
    assert _window_da_param_vector(_window(True)) == expected


def test_s0_window_gives_plain_params() -> None:
    expected = [TRUE[n] for n in ("F", "c1", "hx", "eps")] + TRUE["fast_weights"]
    assert _window_da_param_vector(_window(False)) == expected


def test_flattened_fast_weights_do_not_mask_the_biased_list() -> None:
    w = _window(True)
    w.update({f"w{j}": TRUE["fast_weights"][j - 1] for j in range(1, 5)})
    assert _window_da_param_vector(w)[4:] == DA["fast_weights_da"]


def test_collate_joint_eval_params_are_da_and_true_params_are_truth() -> None:
    batch = collate_joint_eval([_window(True), _window(False)])
    assert torch.allclose(batch["params"][0], torch.tensor(_window_da_param_vector(_window(True))))
    assert torch.allclose(batch["params"][1], batch["true_params"][1])
    assert not torch.allclose(batch["params"][0], batch["true_params"][0])
    assert torch.equal(batch["forcing"][0], torch.ones(5))
