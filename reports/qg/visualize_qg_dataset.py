"""Export a generated QG spectral-wind dataset for the dataset overview page.

Writes `data.json` (spec, generation summary, independence checks, train/val
distributions, regime occupancy, KS tests) and sample-window animations for
train and val. Following the dataset protocol, the test split contributes only
its independence checks and counts: its distributions and samples are not
exported until a model is frozen.
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data.qg_datasets import SPECS, QGGeneratedSplit, split_dir  # noqa: E402

VARIABLES = {
    "rd": ("deformation radius rd", "km", 1e-3),
    "U1": ("upper-layer mean flow U1", "cm/s", 100.0),
    "rek": ("bottom drag rek", "10⁻⁷ s⁻¹", 1e7),
    "level": ("wind level", "10⁻¹¹ s⁻²", 1e11),
    "cx": ("zonal drift cx", "m/s", 1.0),
    "cy": ("meridional drift cy", "m/s", 1.0),
    "time_unit_days": ("gyrostat time unit", "days per unit", 1.0),
    "rms_curl": ("rms wind curl over the window", "10⁻¹² s⁻²", 1e12),
    "ke_upper": ("upper-layer kinetic energy", "10⁻³ m² s⁻²", 1e3),
}


def _values(split: QGGeneratedSplit, var: str) -> np.ndarray:
    if var in ("rms_curl", "ke_upper"):
        return np.array([w[var] for w in split.windows])
    return np.array([w["factors"][var] for w in split.windows])


def _hist(splits: dict, var: str, bins: int = 12) -> dict:
    scale = VARIABLES[var][2]
    allv = np.concatenate([_values(s, var) for s in splits.values()]) * scale
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    out = {"edges": edges.tolist(), "label": VARIABLES[var][0], "unit": VARIABLES[var][1],
           "series": {}}
    for name, s in splits.items():
        counts, _ = np.histogram(_values(s, var) * scale, bins=edges)
        out["series"][name] = (counts / max(1, counts.sum())).tolist()
    return out


def _pick_samples(split: QGGeneratedSplit, n: int) -> list[int]:
    level = _values(split, "level")
    order = np.argsort(level)
    nonzero = [i for i in order if level[i] > 0]
    picks = []
    if n >= 1 and (level == 0).any():
        picks.append(int(order[0]))
    for q in np.linspace(0.35, 1.0, max(1, n - len(picks))):
        picks.append(int(nonzero[min(len(nonzero) - 1, int(q * (len(nonzero) - 1)))]))
    return sorted(set(picks))[:n]


def _animate(split: QGGeneratedSplit, i: int, path: str, scale: float = 0.55,
             stride: int = 2) -> dict:
    f0 = split.manifest["window_start_frame"]
    keep = split.manifest["keep_every"]
    item = split[i]
    psi = split.streamfunction(i)[f0:].numpy()
    q = item["true_state"][f0:].numpy()
    curl_all = split.curl_field(i).numpy()
    lead_steps = split.spec.n_lead
    curl = curl_all[lead_steps::keep][: q.shape[0]]
    if curl.shape[0] < q.shape[0]:
        curl = np.concatenate([curl, curl[-1:]], 0)
    psi, q, curl = psi[::stride], q[::stride], curl[::stride]
    vm = [float(np.percentile(np.abs(a), 99.5)) for a in (psi[:, 0], q[:, 0], curl)]
    calm = vm[2] == 0.0
    vm = [v if v > 0 else 1.0 for v in vm]
    days = np.arange(q.shape[0]) * stride * keep / split.spec.steps_per_day
    frames = []
    for t in range(q.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
        curl_title = "wind-stress curl (calm)" if calm else "wind-stress curl"
        for ax, (fld, ttl, v) in zip(axes, ((psi[t, 0], "upper ψ₁", vm[0]), (q[t, 0], "upper q₁", vm[1]),
                                            (curl[t], curl_title, vm[2]))):
            ax.imshow(fld, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
            ax.set_title(f"{ttl}, day {days[t]:.1f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout(pad=1.2, rect=(0, 0, 1, 0.97))
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        frames.append(img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=220, loop=0, optimize=True)
    w = split.windows[i]
    return {"file": os.path.basename(path), "index": w["index"], "regime": w["regime"],
            "factors": w["factors"], "rms_curl": w["rms_curl"], "ke_upper": w["ke_upper"],
            "width": frames[0].width, "height": frames[0].height, "frames": len(frames)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", default="qg_specwind_gyrostat_demo", choices=sorted(SPECS))
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-samples", type=int, default=3)
    p.add_argument("--val-samples", type=int, default=2)
    args = p.parse_args()
    spec = SPECS[args.spec]
    os.makedirs(os.path.join(args.out_dir, "gifs"), exist_ok=True)
    tr = QGGeneratedSplit(split_dir(args.root, spec, "train"), purpose="train")
    va = QGGeneratedSplit(split_dir(args.root, spec, "val"), purpose="eval")
    with open(os.path.join(split_dir(args.root, spec, "test"), "manifest.json")) as fh:
        test_manifest = json.load(fh)
    reports = os.path.join(args.root, spec.name, "reports")
    with open(os.path.join(reports, "independence.json")) as fh:
        independence = json.load(fh)
    with open(os.path.join(reports, "diversity.json")) as fh:
        diversity = json.load(fh)
    sizes = {}
    for s in ("train", "val", "test"):
        out = subprocess.run(["du", "-sb", split_dir(args.root, spec, s)], capture_output=True, text=True)
        sizes[s] = int(out.stdout.split()[0]) if out.returncode == 0 else None
    splits = {"train": tr, "val": va}
    regimes = sorted({w["regime"] for s in splits.values() for w in s.windows})
    data = {
        "spec": spec.to_json(),
        "counts": {"train": len(tr), "val": len(va), "test": test_manifest["n"]},
        "generation_seconds": {"train": tr.manifest["generation_seconds"],
                               "val": va.manifest["generation_seconds"],
                               "test": test_manifest["generation_seconds"]},
        "bytes": sizes,
        "split_hashes": {"train": tr.manifest["split_hash"], "val": va.manifest["split_hash"],
                         "test": test_manifest["split_hash"]},
        "independence": independence,
        "histograms": {v: _hist(splits, v) for v in VARIABLES},
        "regimes": {"labels": regimes,
                    "series": {n: [sum(w["regime"] == r for w in s.windows) / len(s) for r in regimes]
                               for n, s in splits.items()}},
        "scatter": {n: [[w["factors"]["time_unit_days"], w["factors"]["level"] * 1e11, w["index"]]
                        for w in s.windows] for n, s in splits.items()},
        "ks_val_vs_train": diversity["val"].get("ks_pvalue_vs_train", {}),
        "discrepancy": {n: diversity[n]["unit_design_discrepancy"] for n in splits},
        "calm_fraction": {n: diversity[n]["calm_fraction"] for n in splits},
        "samples": {},
    }
    for name, s, n in (("train", tr, args.train_samples), ("val", va, args.val_samples)):
        data["samples"][name] = [
            _animate(s, i, os.path.join(args.out_dir, "gifs", f"{name}_{s.windows[i]['index']:04d}.gif"))
            for i in _pick_samples(s, n)]
    with open(os.path.join(args.out_dir, "data.json"), "w") as fh:
        json.dump(data, fh)
    print(json.dumps({"out": args.out_dir, "samples": {k: len(v) for k, v in data["samples"].items()},
                      "independence_passed": independence["passed"]}))


if __name__ == "__main__":
    main()
