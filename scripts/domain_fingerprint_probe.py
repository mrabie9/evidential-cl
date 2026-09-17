#!/usr/bin/env python
"""Is the conflict share a property of the data, or of the head trained on it?

The task-end conflict share separates the three data sources cleanly (`rcn`
0.665, `deeprad` 0.745, `uclresm` 0.916, within-domain sd 0.023) while barely
ranking tasks *inside* a source. That makes it a candidate **domain / drift
detector** rather than a difficulty measure -- but only if the fingerprint
survives being read off a head that was never trained on the domain in question.
As measured during training it is confounded: each task's share is computed over
that task's *own* columns, so "domain" and "the head fitted to that domain" are
never separated.

This scores every task's test data through a **fixed foreign column set** and
reports the conflict share per (data task, scoring head):

* ``--head-task k`` scores every task's data through task k's columns.
* ``--head prior`` scores task t's data through the columns of tasks < t, which
  is exactly the evidence available at the moment task t arrives and is the
  ante-hoc configuration an allocation rule would actually run in.

A fingerprint that holds under a foreign head is usable before training; one that
disappears is an artefact of the fitted readout and cannot drive allocation.

Note the model here has already trained on every task, so this is a *necessary*
condition, not a sufficient one: a negative result rules the idea out, a positive
one still needs the in-training probe.

Usage:
    python scripts/domain_fingerprint_probe.py --run logs/woe_si_lc/caut_sum_lam240000_s0-*/0
    python scripts/domain_fingerprint_probe.py --run <dir> --head prior --limit-batches 8
"""

from __future__ import annotations

import argparse
import glob
import importlib
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.detection_replay import noise_label_from_args  # noqa: E402
from model.woe_si import (  # noqa: E402
    compute_weights_of_evidence,
    per_class_total_evidence,
)
from scripts.confusion_from_run import (  # noqa: E402
    _pick_device,
    _task_feature_mean,
    build_model,
    load_run,
)
from utils import misc_utils  # noqa: E402


def columns_for(args, task_index: int, n_outputs: int, device) -> torch.Tensor:
    return misc_utils.current_task_class_indices(
        task_index,
        args.classes_per_task,
        n_outputs,
        global_noise_label=noise_label_from_args(args),
        device=device,
    )


@torch.no_grad()
def conflict_share(
    model, loader, args, columns: torch.Tensor, limit: Optional[int], bn_training: bool
) -> Optional[float]:
    """Sample-averaged ``2 sum_k w+ w- / I_2`` over one loader, fixed columns.

    Labels are never used: the statistic is a function of the features and the
    readout alone, which is the whole point of proposing it as a signal.
    """
    device = next(model.net.parameters()).device
    centering = str(getattr(args, "woe_centering_mode", "centered_uniform"))
    mu = _task_feature_mean(model, loader, device, limit, bn_training)
    readout = model.net.model.fc
    conflict_total = 0.0
    i2_total = 0.0
    for batch_index, batch in enumerate(loader):
        if limit is not None and batch_index >= limit:
            break
        xb = batch[0].to(device)
        feats = model.net.forward_features(xb, bn_training=bn_training)
        weights = compute_weights_of_evidence(
            feats,
            readout.weight[columns],
            readout.bias[columns],
            mu,
            centering_mode=centering,
        )
        w_plus, w_minus = per_class_total_evidence(weights)
        conflict = 2.0 * (w_plus * w_minus).sum(dim=1)
        i2 = (w_plus.pow(2) + w_minus.pow(2)).sum(dim=1)
        conflict_total += float(conflict.sum().item())
        i2_total += float(i2.sum().item())
    if i2_total <= 0.0:
        return None
    return conflict_total / i2_total


def domain_of(name: str) -> str:
    return name.split("-", 1)[1].replace(".npz", "").replace("-25noise", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--head", default="each", choices=["each", "prior", "own"])
    ap.add_argument("--head-task", type=int, default=None)
    ap.add_argument("--limit-batches", type=int, default=8)
    ap.add_argument("--bn-training", action="store_true")
    args_cli = ap.parse_args()

    run_dirs: List[Path] = []
    for pattern in args_cli.run:
        run_dirs += [Path(p) for p in sorted(glob.glob(pattern))]

    for run_dir in run_dirs:
        state_dict, args = load_run(run_dir)
        args.cuda = bool(getattr(args, "cuda", False)) and torch.cuda.is_available()
        loader_module = importlib.import_module("dataloaders." + str(args.loader))
        incremental = loader_module.IncrementalLoader(
            args, seed=int(getattr(args, "seed", 0))
        )
        n_inputs, _, _ = incremental.get_dataset_info()
        test_loaders = incremental.get_tasks("test")
        model, n_outputs = build_model(args, state_dict, n_inputs)
        model = model.to(_pick_device(args))
        device = next(model.net.parameters()).device
        n_tasks = len(test_loaders)

        heads: List[int]
        if args_cli.head_task is not None:
            heads = [args_cli.head_task]
        elif args_cli.head == "each":
            heads = list(range(n_tasks))
        else:
            heads = []

        print(f"[run] {run_dir}")
        if heads:
            print("conflict share, rows = data task, cols = scoring head")
            header = "     " + "".join(f"{h:>8}" for h in heads) + f"{'own':>8}"
            print(header)
            table = np.full((n_tasks, len(heads) + 1), np.nan)
            for t, loader in enumerate(test_loaders):
                for j, h in enumerate(heads):
                    cols = columns_for(args, h, n_outputs, device)
                    if cols.numel():
                        v = conflict_share(
                            model, loader, args, cols, args_cli.limit_batches,
                            args_cli.bn_training,
                        )
                        table[t, j] = np.nan if v is None else v
                own = columns_for(args, t, n_outputs, device)
                v = conflict_share(
                    model, loader, args, own, args_cli.limit_batches,
                    args_cli.bn_training,
                )
                table[t, -1] = np.nan if v is None else v
                print(f"{t:>5}" + "".join(f"{x:>8.3f}" for x in table[t]))
        else:
            print("ante-hoc: task t's data scored through the columns of tasks < t")
            print(f"{'task':>5}{'prior share':>13}{'own share':>11}")
            for t, loader in enumerate(test_loaders):
                if t == 0:
                    print(f"{t:>5}{'-':>13}", end="")
                    prior_v = None
                else:
                    cols = torch.cat(
                        [columns_for(args, k, n_outputs, device) for k in range(t)]
                    )
                    prior_v = conflict_share(
                        model, loader, args, cols, args_cli.limit_batches,
                        args_cli.bn_training,
                    )
                    print(f"{t:>5}{prior_v:>13.4f}", end="")
                own = columns_for(args, t, n_outputs, device)
                own_v = conflict_share(
                    model, loader, args, own, args_cli.limit_batches,
                    args_cli.bn_training,
                )
                print(f"{own_v:>11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
