"""L96 FDV2 x flow-matching posterior: evaluation harness and baselines B1-B5.

Plan: ``docs/plans/analysis/l96_fdv2_fm_posterior.md`` (sections 0.3 and 3).
Every row is scored window by window in physical units with
``evaluation.posterior_scores``; the output JSON keeps the per-window scores,
the run configuration and the aggregate (never member arrays).

Methods are registered by name in ``METHODS`` (``@register("NAME", **defaults)``);
a method maps one window (a batch of size 1, normalised obs) to an ensemble
``(M, T, D)`` in normalised units and its NFE per member. Method arguments are
passed as ``--args key=value,key=value``.

Common random numbers: each window draws from seeds that depend only on
(``--seed``, window index, role), never on the method or on the other windows
evaluated, so the FDV2 mean (role ``mean``) is the same for B1/B3/B5 and the SDA
start noise (role ``members``) is the same for B3/B4/B5 at a given
``--member-chunk``.

Validation sets (100 windows, seeds 1057/1067, disjoint from train/val/test)::

  python scripts/make_l96_validation_sets.py --n-windows 100

writes ``experiments/l96_valset_regular_w100.pt`` and
``experiments/l96_valset_rlayout_n10-100_k4-16_w100.pt``.

Examples::

  python eval_fdv_fm_posterior_l96.py --method B3 --args tau0=0.5,gw=2 \\
      --split val --layout regular --case s0 --fdv-seed 2 --out row.json
  python eval_fdv_fm_posterior_l96.py --method B5 --args tau0=0.1,gw=2,inflation_from=val_b3.json \\
      --split test --out b5.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset

from data.normalization import denormalize, load_norm_stats
from evaluation.neural_inference import BatchDict, load_checkpoint, load_model, prepare_dataset
from evaluation.posterior_scores import aggregate, inflate, window_scores
from evaluation.sda_sampler import sda_guided_sample

SHARED = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments"
FDV_MASK = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-fdv-mask/experiments"
DATASETS = {
    ("test", "regular"): f"{SHARED}/l96_datasets_obsj2_int100_nwin200.pt",
    ("test", "random"): f"{SHARED}/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt",
    ("val", "regular"): f"{SHARED}/l96_valset_regular_w100.pt",
    ("val", "random"): f"{SHARED}/l96_valset_rlayout_n10-100_k4-16_w100.pt",
}
FDV_RUNS = {
    1: f"{FDV_MASK}/L96B_fdvsubgradmask_monaiM_aux01det_deepsup_ft_coupled_s1philr_seed1",
    2: f"{FDV_MASK}/L96B_fdvsubgradmask_monaiM_aux01det_deepsup_ft_coupled_seed2",
}
SDA_RUN = f"{SHARED}/l96/B4_sda1_monaiM_ep1200_l96_seed1"
NORM_STATS = f"{SHARED}/l96_norm_stats_obsj2.pt"
R_VAR = 0.5


@dataclass
class Context:
    device: torch.device
    norm_stats: dict
    fdv_run: str
    sda_run: str
    n_members: int
    member_chunk: int
    seed: int
    _models: dict = field(default_factory=dict)

    @property
    def obs_std_norm(self) -> torch.Tensor:
        return (R_VAR ** 0.5) / self.norm_stats["std"].to(self.device)

    def model(self, name: str) -> torch.nn.Module:
        if name not in self._models:
            run = {"fdv": self.fdv_run, "sda": self.sda_run}[name]
            m, _ = load_model(f"{run}/checkpoints/stage1_best.ckpt", f"{run}/resolved_config.yaml",
                              device=self.device)
            self._models[name] = m.eval()
        return self._models[name]


Method = Callable[[Context, dict, int, dict], tuple[torch.Tensor, int]]


@dataclass
class MethodSpec:
    fn: Method
    defaults: dict
    tune_inflation: bool = False


METHODS: dict[str, MethodSpec] = {}


def register(name: str, tune_inflation: bool = False, **defaults) -> Callable[[Method], Method]:
    def deco(fn: Method) -> Method:
        METHODS[name] = MethodSpec(fn, defaults, tune_inflation)
        return fn
    return deco


def window_seed(base: int, window: int, role: str) -> int:
    h = hashlib.sha256(f"{base}/{window}/{role}".encode()).hexdigest()
    return int(h[:15], 16)


def replicate(batch: dict, n: int) -> dict:
    return {k: (v.expand(n, *v.shape[1:]).contiguous() if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def fdv_mean(ctx: Context, batch: dict, window: int, n_iter: int) -> torch.Tensor:
    torch.manual_seed(window_seed(ctx.seed, window, "mean"))
    return ctx.model("fdv").sample(BatchDict(batch), N_outer=n_iter)


def chunks(ctx: Context) -> list[tuple[int, int]]:
    step = ctx.member_chunk or ctx.n_members
    return [(s, min(s + step, ctx.n_members)) for s in range(0, ctx.n_members, step)]


def sda_members(ctx: Context, batch: dict, window: int, tau0: float, gw: float, n_steps: int,
                mean: torch.Tensor | None) -> tuple[torch.Tensor, int]:
    out, nfe = [], 0
    for start, stop in chunks(ctx):
        n = stop - start
        b = BatchDict(replicate(batch, n))
        m = mean.expand(n, *mean.shape[1:]).contiguous() if mean is not None else None
        torch.manual_seed(window_seed(ctx.seed, window, f"members/{start}"))
        x, nfe = sda_guided_sample(ctx.model("sda"), b, R_var=R_VAR, N_outer=n_steps, guidance_weight=gw,
                                   n_members=1, mean_estimate=m, tau0=tau0)
        out.append(x.detach())
    return torch.cat(out, 0), nfe


@register("B1", n_iter=10)
def b1_fdv_point(ctx: Context, batch: dict, window: int, args: dict) -> tuple[torch.Tensor, int]:
    return fdv_mean(ctx, batch, window, int(args["n_iter"])), int(args["n_iter"])


@register("B2", n_iter=10)
@torch.no_grad()
def b2_perturbed_obs(ctx: Context, batch: dict, window: int, args: dict) -> tuple[torch.Tensor, int]:
    out = []
    for start, stop in chunks(ctx):
        b = replicate(batch, stop - start)
        torch.manual_seed(window_seed(ctx.seed, window, f"members/{start}"))
        finite = torch.isfinite(b["obs"])
        b["obs"] = torch.where(finite, b["obs"] + torch.randn_like(b["obs"]) * ctx.obs_std_norm, b["obs"])
        out.append(ctx.model("fdv").sample(BatchDict(b), N_outer=int(args["n_iter"])))
    return torch.cat(out, 0), int(args["n_iter"])


@register("B3", tau0=0.1, gw=2.0, n_steps=10, n_iter=10)
def b3_hybrid(ctx: Context, batch: dict, window: int, args: dict) -> tuple[torch.Tensor, int]:
    mean = fdv_mean(ctx, batch, window, int(args["n_iter"]))
    x, nfe = sda_members(ctx, batch, window, float(args["tau0"]), float(args["gw"]), int(args["n_steps"]), mean)
    return x, nfe + int(args["n_iter"])


@register("B4", gw=20.0, n_steps=10)
def b4_sda(ctx: Context, batch: dict, window: int, args: dict) -> tuple[torch.Tensor, int]:
    return sda_members(ctx, batch, window, 0.0, float(args["gw"]), int(args["n_steps"]), None)


@register("B5", tune_inflation=True, tau0=0.1, gw=2.0, n_steps=10, n_iter=10, inflation=None, inflation_from=None)
def b5_hybrid_inflated(ctx: Context, batch: dict, window: int, args: dict) -> tuple[torch.Tensor, int]:
    return b3_hybrid(ctx, batch, window, args)


def parse_args_kv(text: str | None, defaults: dict) -> dict:
    out = dict(defaults)
    for item in filter(None, (text or "").split(",")):
        k, v = item.split("=", 1)
        if k not in defaults:
            raise ValueError(f"unknown method argument {k!r}; expected one of {sorted(defaults)}")
        out[k] = v
    for k, v in out.items():
        if isinstance(v, str) and k != "inflation_from":
            try:
                out[k] = float(v) if any(c in v for c in ".e") else int(v)
            except ValueError:
                pass
    return out


def parse_windows(text: str | None, n: int) -> list[int]:
    if not text:
        return list(range(n))
    idx: list[int] = []
    for part in text.split(","):
        if ":" in part:
            a, b = part.split(":")
            idx += list(range(int(a or 0), int(b or n)))
        else:
            idx.append(int(part))
    if any(i < 0 or i >= n for i in idx):
        raise ValueError(f"window index out of range [0, {n})")
    return idx


def resolve_inflation(args: dict, split: str) -> float | None:
    if args.get("inflation") not in (None, "None"):
        return float(args["inflation"])
    if args.get("inflation_from"):
        with open(args["inflation_from"]) as f:
            src = json.load(f)
        if src["config"]["split"] != "val":
            raise ValueError("inflation_from must be a validation-split run")
        return 1.0 / src["aggregate"]["ssr"]
    if split != "val":
        raise ValueError("B5 on the test split needs inflation=<f> or inflation_from=<val json>")
    return None


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       text=True, stderr=subprocess.DEVNULL,
                                       env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull}).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(opts: argparse.Namespace) -> dict:
    spec = METHODS[opts.method]
    margs = parse_args_kv(opts.args, spec.defaults)
    device = torch.device(opts.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ns = load_norm_stats(opts.norm_stats)
    ctx = Context(device, ns, opts.fdv_run or FDV_RUNS[opts.fdv_seed], opts.sda_run, opts.n_members,
                  opts.member_chunk, opts.seed)
    dataset_path = opts.dataset or DATASETS[(opts.split, opts.layout)]
    _, fdv_cfg = load_checkpoint(f"{ctx.fdv_run}/checkpoints/stage1_best.ckpt", f"{ctx.fdv_run}/resolved_config.yaml")
    _, dls, ovi = prepare_dataset(fdv_cfg, dataset_path, 200, 100, obs_j=2, norm_stats=ns, batch_size=1)
    full = dls[opts.case]
    windows = parse_windows(opts.windows, len(full.dataset))
    loader = DataLoader(Subset(full.dataset, windows), batch_size=1, shuffle=False, collate_fn=full.collate_fn)

    inflation = resolve_inflation(margs, opts.split) if spec.tune_inflation else None
    keep = spec.tune_inflation and inflation is None
    kept: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_window, nfe, t0 = [], 0, time.time()
    for w, batch in zip(windows, loader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        truth = batch["true_state"][0][..., list(ovi)]
        x, nfe = spec.fn(ctx, batch, w, margs)
        members = denormalize(x, ns).float()
        if inflation is not None:
            members = inflate(members, inflation)
        if keep:
            kept.append((members.cpu(), truth.cpu()))
        else:
            per_window.append(window_scores(members, truth))
        if opts.verbose:
            print(f"window {w}: {time.time() - t0:.1f}s", flush=True)
    if keep:
        raw = aggregate([window_scores(m, t) for m, t in kept])
        inflation = 1.0 / raw["ssr"]
        per_window = [window_scores(inflate(m, inflation), t) for m, t in kept]
    wall = time.time() - t0

    config = {"method": opts.method, "method_args": margs, "split": opts.split, "layout": opts.layout,
              "case": opts.case, "fdv_seed": opts.fdv_seed, "fdv_run": ctx.fdv_run, "sda_run": ctx.sda_run,
              "dataset": dataset_path, "norm_stats": opts.norm_stats, "n_members": opts.n_members,
              "member_chunk": opts.member_chunk, "seed": opts.seed, "R_var": R_VAR,
              "nfe_per_member": nfe, "wall_clock_s": wall, "git_sha": git_sha(), "device": str(device)}
    if spec.tune_inflation:
        config["inflation"] = inflation
    return {"config": config, "windows": windows, "per_window": per_window, "aggregate": aggregate(per_window)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--args", default=None, help="method arguments key=value,key=value")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--layout", choices=("regular", "random"), default="regular")
    p.add_argument("--case", choices=("s0", "s1"), default="s0")
    p.add_argument("--fdv-seed", type=int, choices=sorted(FDV_RUNS), default=2)
    p.add_argument("--fdv-run", default=None, help="override the FDV2 run directory")
    p.add_argument("--sda-run", default=SDA_RUN)
    p.add_argument("--dataset", default=None, help="override the split/layout dataset path")
    p.add_argument("--norm-stats", default=NORM_STATS)
    p.add_argument("--n-members", type=int, default=30)
    p.add_argument("--member-chunk", type=int, default=0, help="members per forward batch (0 = all)")
    p.add_argument("--windows", default=None, help="subset, e.g. '0:20' or '0,3,7'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    opts = build_parser().parse_args()
    res = run(opts)
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    with open(opts.out, "w") as f:
        json.dump(res, f)
    a = res["aggregate"]
    print(f"{opts.method} {res['config']['method_args']} {opts.split}/{opts.layout}/{opts.case} "
          f"fdv_seed={opts.fdv_seed} W={a['n_windows']} M={a['n_members']}: rmse {a['rmse']:.4f} "
          f"crps {a['crps']:.4f} ssr {a['ssr']:.3f} rel {a['reliability']:.3f} cov90 {a['cov90']:.3f} "
          f"es {a['energy_score']:.4f} ({res['config']['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
