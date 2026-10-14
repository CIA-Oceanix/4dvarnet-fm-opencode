"""L96 test sets that withhold the window parameters and/or forcing from the schemes.

Each output cache keeps the S0 test windows as ``test_s0`` and stores, as ``test_s1``, the
same windows (same truth, same observations) in which the information given to the schemes
is degraded. Both the DA baselines and the parameter-conditioned SDA priors read the window
parameters from the ``*_da`` keys and the forcing from ``forcing_corrupted`` in the S1 case,
so evaluating ``test_s1`` measures how much each scheme uses that information:

    no_theta    nominal parameters (F 8, c1 = h = hx = 1, eps 0.1, w (1, 1, 0.1, 0.1)), true forcing
    no_forcing  true parameters, zero forcing
    neither     nominal parameters, zero forcing

usage: make_l96_conditioning_variants.py SRC_CACHE OUT_DIR
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOMINAL = {"F": 8.0, "c1": 1.0, "h": 1.0, "hx": 1.0, "eps": 0.1, "fast_weights": [1.0, 1.0, 0.1, 0.1]}
VARIANTS = {"no_theta": (True, False), "no_forcing": (False, True), "neither": (True, True)}


def variant_windows(windows: list[dict], nominal_params: bool, zero_forcing: bool) -> list[dict]:
    out = []
    for w in windows:
        v = dict(w)
        for k, nominal in NOMINAL.items():
            v[f"{k}_da"] = list(nominal) if (nominal_params and k == "fast_weights") else (
                nominal if nominal_params else w[f"true_{k}"])
        v["forcing_corrupted"] = torch.zeros_like(w["forcing_true"]) if zero_forcing else w["forcing_true"].clone()
        v["param_bias"] = 0.0
        out.append(v)
    return out


def main(src: str, out_dir: str) -> None:
    d = torch.load(src, weights_only=False, map_location="cpu")
    os.makedirs(out_dir, exist_ok=True)
    for name, (nominal_params, zero_forcing) in VARIANTS.items():
        s1 = copy.copy(d["test_s1"])
        s1.windows = variant_windows(d["test_s0"].windows, nominal_params, zero_forcing)
        path = os.path.join(out_dir, f"l96_conditioning_{name}.pt")
        torch.save({"test_s0": d["test_s0"], "test_s1": s1}, path)
        print(path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
