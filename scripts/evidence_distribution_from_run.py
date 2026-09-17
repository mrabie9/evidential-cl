#!/usr/bin/env python
"""Per-class shape of the evidence distribution (``I_2`` and conflict) in a run.

Every trace this project prints collapses the class axis: ``information_content``
sums over ``dim=1``, so ``I_2`` and its conflict half are only ever reported as
per-sample totals. That answers "how committed is the readout" and cannot answer
"how is that commitment distributed over classes" -- whether a couple of columns
carry it, whether the conflict share is uniform, or whether the split is
label-conditional in the way an evidential objective needs it to be.

This reconstructs a finished run from ``results.pt`` (which carries the full state
dict even under ``--no-save_checkpoints``) and reports, per task, over that task's
own columns and its own test data:

* ``i2_k = w+_k^2 + w-_k^2`` per class, as a share of the task's total, with the
  concentration statistics (top share, max/min ratio, Gini).
* ``conflict_k = 2 w+_k w-_k`` and the per-class conflict *share*
  ``conflict_k / i2_k``, which is the per-class version of the 84.6% figure.
* The **label-conditional** split of ``b_k = w+_k / (w+_k + w-_k)``: its mean on
  samples of class ``k`` against its mean on every other sample. This is the
  quantity an evidential objective actually targets, and the aggregate conflict
  share cannot see it -- one-sidedness conditioned on the label is invisible to a
  statistic that averages over all samples.
* Whether any of it tracks class frequency, since the priors here run ~5% to ~48%.

Centring follows the run's own ``woe_centering_mode``, with ``mu`` recomputed per
task from that task's test features (the saved ``woe_feature_mean`` is whatever the
last task left behind), matching ``scripts/confusion_from_run.py``.

Usage:
    python scripts/evidence_distribution_from_run.py --run logs/woe_si_evidential/evobj_noanchor_s0-*/0
    python scripts/evidence_distribution_from_run.py --run <dir_a> <dir_b> --limit-batches 4
"""

from __future__ import annotations

import argparse
import glob
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.detection_replay import (  # noqa: E402
    noise_label_from_args,
    unpack_y_to_class_labels,
)
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


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = uniform, ->1 = one class)."""
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(np.clip(values, 0.0, None))
    total = sorted_values.sum()
    if total <= 0.0:
        return 0.0
    index = np.arange(1, sorted_values.size + 1)
    return float(
        (2.0 * index - sorted_values.size - 1.0).dot(sorted_values)
        / (sorted_values.size * total)
    )


@torch.no_grad()
def task_evidence_shape(
    model,
    loader,
    args: object,
    task_index: int,
    n_outputs: int,
    limit: Optional[int],
    bn_training: bool,
) -> Optional[Dict]:
    """Accumulate per-class evidence statistics over one task's test set."""
    device = next(model.net.parameters()).device
    columns = misc_utils.current_task_class_indices(
        task_index,
        args.classes_per_task,
        n_outputs,
        global_noise_label=noise_label_from_args(args),
        device=device,
    )
    if columns.numel() == 0:
        return None
    class_count = int(columns.numel())
    centering = str(getattr(args, "woe_centering_mode", "centered_uniform"))
    mu = _task_feature_mean(model, loader, device, limit, bn_training)
    readout = model.net.model.fc

    # Sums over samples: per-class I_2 / conflict, and b split by membership.
    i2_sum = np.zeros(class_count)
    conflict_sum = np.zeros(class_count)
    b_member_sum = np.zeros(class_count)
    b_other_sum = np.zeros(class_count)
    member_counts = np.zeros(class_count)
    sample_count = 0
    # Evidence mass along the *feature* axis j, summed over samples and classes.
    # Every other statistic here collapses j (w_plus is already a sum over it),
    # so this is the only view that can answer how many features actually carry
    # the evidence -- the question the I_p exponent controls, since I_1 is the L1
    # norm of w_jk and sparsifies it where I_2 spreads it.
    feature_mass_sum: Optional[np.ndarray] = None

    lookup = torch.full((n_outputs,), -1, dtype=torch.long, device=device)
    lookup[columns] = torch.arange(class_count, device=device)

    for batch_index, batch in enumerate(loader):
        if limit is not None and batch_index >= limit:
            break
        xb = batch[0].to(device)
        yb = unpack_y_to_class_labels(batch[1]).long().view(-1).to(device)
        local = lookup[yb.clamp(0, n_outputs - 1)]
        keep = (yb >= 0) & (yb < n_outputs) & (local >= 0)
        if not bool(keep.any()):
            continue
        xb, local = xb[keep], local[keep]

        feats = model.net.forward_features(xb, bn_training=bn_training)
        weights = compute_weights_of_evidence(
            feats,
            readout.weight[columns],
            readout.bias[columns],
            mu,
            centering_mode=centering,
        )
        w_plus, w_minus = per_class_total_evidence(weights)
        i2 = w_plus.pow(2) + w_minus.pow(2)
        conflict = 2.0 * w_plus * w_minus
        balance = w_plus / (w_plus + w_minus).clamp_min(1e-12)

        membership = torch.zeros_like(balance, dtype=torch.bool)
        membership[torch.arange(local.numel(), device=device), local] = True

        i2_sum += i2.sum(dim=0).cpu().numpy()
        conflict_sum += conflict.sum(dim=0).cpu().numpy()
        b_member_sum += (balance * membership).sum(dim=0).cpu().numpy()
        b_other_sum += (balance * ~membership).sum(dim=0).cpu().numpy()
        member_counts += membership.sum(dim=0).cpu().numpy()
        sample_count += int(local.numel())
        batch_feature_mass = weights.abs().sum(dim=(0, 1)).cpu().numpy()
        feature_mass_sum = (
            batch_feature_mass
            if feature_mass_sum is None
            else feature_mass_sum + batch_feature_mass
        )

    if sample_count == 0:
        return None
    other_counts = np.maximum(sample_count - member_counts, 1.0)
    i2_mean = i2_sum / sample_count
    feature_mass = (
        feature_mass_sum / sample_count if feature_mass_sum is not None else np.zeros(1)
    )
    feature_share = feature_mass / max(feature_mass.sum(), 1e-12)
    # Participation ratio: 1 when every feature carries equal evidence, 1/J when
    # one carries it all. The interpretable companion is "effective features",
    # participation * J, i.e. how many features the evidence is really spread
    # over -- which is the number the regularisation/architectural bridge is
    # about, since a mask is the limit of driving it down.
    participation = float(
        1.0 / max(np.square(feature_share).sum() * feature_share.size, 1e-12)
    )
    return {
        "task": task_index,
        "columns": columns.cpu().numpy(),
        "samples": sample_count,
        "i2_mean": i2_mean,
        "i2_share": i2_mean / max(i2_mean.sum(), 1e-12),
        "conflict_share_per_class": conflict_sum / np.maximum(i2_sum, 1e-12),
        "b_member": b_member_sum / np.maximum(member_counts, 1.0),
        "b_other": b_other_sum / other_counts,
        "prior": member_counts / sample_count,
        "feature_count": int(feature_share.size),
        "feature_participation": participation,
        "feature_effective": participation * feature_share.size,
        "feature_gini": _gini(feature_mass),
        "feature_vacuous_frac": float(
            (feature_mass < 0.05 * max(feature_mass.max(), 1e-12)).mean()
        ),
    }


def report(run_dir: Path, rows: List[Dict]) -> None:
    """Print the per-task shape plus the run-level aggregates."""
    print(f"\n=== {run_dir.parent.name} ===")
    print(
        f"{'task':>4} {'K':>3} {'i2_top':>7} {'i2_max/min':>11} {'gini':>6} "
        f"{'cshare_min':>11} {'cshare_max':>11} {'b_mem':>7} {'b_oth':>7} {'gap':>7}"
    )
    gaps, tops, ginis = [], [], []
    for row in rows:
        share = row["i2_share"]
        conflict_share = row["conflict_share_per_class"]
        gap = float(np.mean(row["b_member"] - row["b_other"]))
        gaps.append(gap)
        tops.append(float(share.max()))
        ginis.append(_gini(row["i2_mean"]))
        print(
            f"{row['task']:>4} {len(share):>3} {share.max():>7.3f} "
            f"{share.max() / max(share.min(), 1e-12):>11.1f} {_gini(row['i2_mean']):>6.3f} "
            f"{conflict_share.min():>11.3f} {conflict_share.max():>11.3f} "
            f"{float(np.mean(row['b_member'])):>7.4f} "
            f"{float(np.mean(row['b_other'])):>7.4f} {gap:>+7.4f}"
        )

    # Does any of it track the class prior? Pooled over every (task, class) cell.
    priors = np.concatenate([row["prior"] for row in rows])
    shares = np.concatenate([row["i2_share"] for row in rows])
    conflicts = np.concatenate([row["conflict_share_per_class"] for row in rows])
    print(
        f"\nmean i2_top_share={np.mean(tops):.3f}  mean gini={np.mean(ginis):.3f}  "
        f"mean b gap (member - other)={np.mean(gaps):+.4f}"
    )

    # --- the feature axis ------------------------------------------------
    # Every statistic above collapses j, so none of them can say how many
    # features carry the evidence. That is the quantity the I_p exponent
    # controls: I_1 is the L1 norm of w_jk and sparsifies it, I_2 spreads it.
    # `effective` (participation x J) is the readable form -- how many of the J
    # features the evidence is really spread over. A masking method like PackNet
    # or HAT is the limit of driving this down by construction rather than by
    # penalty, which is what makes p a bridge between the two families.
    print(
        f"\n{'task':>4} {'J':>5} {'partic':>8} {'effective':>10} {'gini_j':>7} "
        f"{'vacuous':>8}"
    )
    for row in rows:
        print(
            f"{row['task']:>4} {row['feature_count']:>5} "
            f"{row['feature_participation']:>8.4f} "
            f"{row['feature_effective']:>10.1f} {row['feature_gini']:>7.3f} "
            f"{row['feature_vacuous_frac']:>8.3f}"
        )
    print(
        f"mean effective features="
        f"{np.mean([row['feature_effective'] for row in rows]):.1f} "
        f"of {rows[0]['feature_count']}   "
        f"mean vacuous fraction="
        f"{np.mean([row['feature_vacuous_frac'] for row in rows]):.3f}"
    )
    if priors.size > 2:
        print(
            f"corr(prior, i2_share)={np.corrcoef(priors, shares)[0, 1]:+.3f}  "
            f"corr(prior, conflict_share)={np.corrcoef(priors, conflicts)[0, 1]:+.3f}"
        )
        # The decisive one: if commitment concentrates on the *conflicted*
        # columns, then per-class one-sidedness can improve while the
        # I_2-weighted aggregate share gets worse -- the two are not in conflict.
        print(
            f"corr(i2_share, conflict_share)="
            f"{np.corrcoef(shares, conflicts)[0, 1]:+.3f}   "
            f"i2-weighted conflict share={float(np.sum(shares * conflicts) / max(np.sum(shares), 1e-12)):.3f}   "
            f"unweighted mean={float(np.mean(conflicts)):.3f}"
        )


def process(run_dir: Path, limit: Optional[int], bn_training: bool) -> None:
    """Rebuild one run and report the shape of its evidence distribution."""
    print(f"[run] {run_dir}", flush=True)
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
    print(
        f"  centering={getattr(args, 'woe_centering_mode', '?')} "
        f"evidential={getattr(args, 'woe_evidential_mode', 'off')}",
        flush=True,
    )

    rows = []
    for task_index, loader in enumerate(test_loaders):
        row = task_evidence_shape(
            model, loader, args, task_index, n_outputs, limit, bn_training
        )
        if row is not None:
            rows.append(row)
    if rows:
        report(run_dir, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=str,
        nargs="+",
        required=True,
        help="Per-seed run directories (globs allowed) holding results.pt.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Cap batches per task; the default scores the whole test set.",
    )
    parser.add_argument(
        "--eval-dropout",
        action="store_true",
        help="Score with dropout active (bn_training=True), which is what the "
        "training harness's own metric loop does. Default scores with it off.",
    )
    args = parser.parse_args()

    directories: List[Path] = []
    for pattern in args.run:
        matches = sorted(glob.glob(pattern))
        directories.extend(Path(match) for match in (matches or [pattern]))
    for directory in directories:
        process(directory, args.limit_batches, args.eval_dropout)


if __name__ == "__main__":
    main()
