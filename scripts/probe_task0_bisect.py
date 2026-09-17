"""Fast single-task bisection of the C-MAML vs Res-ER gap.

The advantage is already present on TASK 0 (diag 84.2 vs 82.7), before any
continual-learning dynamics exist, so it is a single-task optimisation
difference. Task 0 trains in ~2 min, which makes bisection cheap compared with
45-minute ten-task runs.

Trains each requested variant on task 0 only, using each model's real observe()
loop and the same evaluation path as main.py, and reports diagonal F1.

Usage:
    la-maml_env/bin/python scripts/probe_task0_bisect.py --variants m4,e0 --seeds 0,39
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types

import numpy as np
import torch

NO_AMP = False
REPO = "/home/lunet/wsmr11/repos/evidential-cl"
os.chdir(REPO)
sys.path.insert(0, REPO)

from utils import misc_utils  # noqa: E402
from utils.training_forward import model_forward_for_metric_loop  # noqa: E402
from utils.training_metrics import macro_f1_including_noise  # noqa: E402

CPT = [6, 7, 6, 7, 7, 6, 6, 6, 7, 6]

# variant -> (params json glob, module, overrides)
VARIANTS = {
    "e0": ("logs/eralg4/e0_resER_joint_splitfix_se_til-*/0/training_parameters.json",
           "model.eralg4", {}),
    "e0_k3": ("logs/eralg4/e0_resER_joint_splitfix_se_til-*/0/training_parameters.json",
              "model.eralg4", {"eralg4_grad_avg": 3}),
    "e0_noclip": ("logs/eralg4/e0_resER_joint_splitfix_se_til-*/0/training_parameters.json",
                  "model.eralg4", {"grad_clip_norm": 0}),
    "m0": ("logs/ablations/til/cmaml/M0/*/0/training_parameters.json",
           "model.lamaml_cifar", {}),
    "m4": ("logs/cmaml_alpha0/m4_alpha0_joint_splitfix_se_til-*/0/training_parameters.json",
           "model.lamaml_cifar", {}),
    "m4_mb1": ("logs/cmaml_alpha0/m4_alpha0_joint_splitfix_se_til-*/0/training_parameters.json",
               "model.lamaml_cifar", {"meta_batches": 1}),
    "m4_noclip": ("logs/cmaml_alpha0/m4_alpha0_joint_splitfix_se_til-*/0/training_parameters.json",
                  "model.lamaml_cifar", {"grad_clip_norm": 0}),
}


def build_args(pattern, overrides, seed):
    import glob

    a = types.SimpleNamespace(**json.load(open(glob.glob(pattern)[0])))
    a.cuda = torch.cuda.is_available()
    a.seed = seed
    a.classes_per_task = CPT
    a.nc_per_task_list = ""
    a.n_epochs = 1
    if NO_AMP:
        a.amp = False
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def evaluate(model, test_loader, args, task):
    model.eval()
    dev = torch.device("cuda" if args.cuda else "cpu")
    f1s = []
    with torch.no_grad():
        for batch in test_loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(dev)
            logits = model_forward_for_metric_loop(model, xb, task, args)
            pred = torch.argmax(logits, dim=1).cpu()
            f1s.append(macro_f1_including_noise(pred, torch.as_tensor(yb).long().view(-1)))
    return 100.0 * sum(f1s) / len(f1s)


def run(variant, seed):
    pattern, module, overrides = VARIANTS[variant]
    args = build_args(pattern, overrides, seed)
    misc_utils.init_seed(seed)
    Loader = importlib.import_module("dataloaders." + args.loader)
    loader = Loader.IncrementalLoader(args, seed=seed)
    n_in, n_out, n_tasks = loader.get_dataset_info()
    args.classes_per_task = loader.classes_per_task or CPT
    info, train_loader, _val, test_loader = loader.new_task()
    t = info["task"]

    misc_utils.init_seed(seed)
    net = importlib.import_module(module).Net(n_in, n_out, n_tasks, args)
    dev = torch.device("cuda" if args.cuda else "cpu")

    # main.py wraps observe() in torch.autocast(bfloat16) when amp is on; the
    # comparison is meaningless without reproducing that.
    from contextlib import nullcontext

    use_amp = bool(getattr(args, "amp", False)) and args.cuda
    t0 = time.time()
    for batch in train_loader:
        xb, yb = batch[0].to(dev), batch[1]
        yb = torch.as_tensor(yb).to(dev)
        net.train()
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else nullcontext()
        )
        with ctx:
            net.observe(xb, yb, t)
    train_s = time.time() - t0
    f1 = evaluate(net, test_loader, args, t)
    print(f"  {variant:11s} seed {seed:<3d}  task0 diag F1 = {f1:6.2f}   ({train_s:.0f}s)")
    return f1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="e0,m4")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--no-amp", action="store_true")
    p = ap.parse_args()
    NO_AMP = p.no_amp
    globals()["NO_AMP"] = p.no_amp
    seeds = [int(s) for s in p.seeds.split(",")]
    print("Task-0-only bisection (no continual-learning dynamics involved)\n")
    out = {}
    for v in p.variants.split(","):
        out[v] = [run(v, s) for s in seeds]
    print()
    for v, vals in out.items():
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {v:11s} mean {np.mean(vals):6.2f} ± {sd:4.2f}  n={len(vals)}")
