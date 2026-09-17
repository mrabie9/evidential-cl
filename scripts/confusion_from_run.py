#!/usr/bin/env python
"""Rebuild a finished run's model offline and dump its confusion structure.

Runs in this project store aggregates only -- ``metrics/task*.npz`` holds recall
and F1 per task, ``results.txt`` the task matrix -- so "which classes are being
confused" cannot be answered from the artefacts directly. It can be answered
*without retraining*, though: ``results.pt`` carries the full model state dict
(weights and BatchNorm buffers, not just the WoE buffers) even for runs launched
with ``--no-save_checkpoints``. This script reloads that, replays the test sets,
and emits three matrices.

**Masked confusion** -- argmax within each sample's own task columns, i.e. what
the reported metrics score. Block-diagonal *by construction* under TIL: a task-3
sample cannot be predicted as a task-7 class however badly the model has
forgotten, so this shows within-task confusion and nothing else.

**Unmasked confusion** -- argmax over the whole 64-wide head. This is where
forgetting becomes visible: it shows which classes an old task's samples get
pulled toward once later tasks have overwritten the representation. Nothing in
the training harness computes it.

**Evidence support** -- mean Dempster-Shafer ``w_plus`` per (true class, class)
pair. The DS-native analogue, and the one that names *pairs*: the conflict term
``2 * w+_k * w-_k`` the objective charges is a per-class scalar, so it can say
"class k is internally conflicted" but never "class k is confused with class j".
This matrix can. Per-class conflict is dumped alongside as a sidebar so the two
readings can be cross-referenced.

Centring: the DS weights of evidence need a feature mean ``mu``. A run's saved
``woe_feature_mean`` is whatever the *last* task left behind, which is not a
sensible reference for earlier tasks, and ``eralg4`` has none at all. Each task's
``mu`` is therefore recomputed here from that task's own test features, in a
first pass, which is well defined for every model and comparable across runs.

Usage:
    python scripts/confusion_from_run.py --run logs/woe_si_lc/lcsplit-anchoronly-*/0
    python scripts/confusion_from_run.py --run <dir> --limit-batches 2   # smoke
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.replay_utils import unpack_y_to_class_labels  # noqa: E402
from model.woe_si import (  # noqa: E402
    compute_weights_of_evidence,
    per_class_total_evidence,
)
from utils import misc_utils  # noqa: E402

# Sequential single-hue ramp, light -> dark: magnitude on a grid takes one hue,
# never a rainbow. Lightest step is allowed to recede toward the chart surface
# because "near zero" is genuinely nothing here.
BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
# Diverging pair for *signed* quantities: blue <-> red across a neutral gray
# midpoint, so "no difference" reads as nothing rather than as a colour. Never
# use the sequential ramp for a contrast -- it puts zero at one end. The palette
# ships a full ramp for blue only, so the warm arm is stepped from its red slot
# (#e34948) at matched lightness.
NEUTRAL = "#f0efec"
DIVERGING_RAMP = [
    "#0d366b",
    "#256abf",
    "#3987e5",
    "#86b6ef",
    "#cde2fb",
    NEUTRAL,
    "#fbd9d8",
    "#f0a3a2",
    "#e34948",
    "#b32e2d",
    "#6b1615",
]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_INK = "#c3c2b7"


def _pick_device(args: object) -> torch.device:
    """Device to score on, chosen here rather than left to each learner.

    ``eralg4.Net`` moves itself to CUDA in its constructor; ``woe_si.Net`` does
    not, because ``main.py`` moves it afterwards. Relying on that difference put
    two runs of the same campaign on different devices, so the choice is made
    explicitly and applied uniformly.
    """
    use_cuda = bool(getattr(args, "cuda", False)) and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


# ======================================================================
# Loading
# ======================================================================
def load_run(run_dir: Path) -> Tuple[Dict[str, torch.Tensor], object]:
    """Pull the state dict and the run's own ``args`` out of ``results.pt``.

    Args:
        run_dir: Directory holding ``results.pt`` (the per-seed leaf).

    Returns:
        ``(state_dict, args)``.

    Raises:
        FileNotFoundError: If the directory has no ``results.pt``.
    """
    path = run_dir / "results.pt"
    if not path.exists():
        raise FileNotFoundError(f"no results.pt under {run_dir}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload[2], payload[5]


def build_model(args: object, state_dict: Dict[str, torch.Tensor], n_inputs: int):
    """Reconstruct the learner and load its trained weights.

    Every model in this campaign wraps a ``ResNet1D`` as ``self.net``, which is
    the only interface this script needs -- so ``woe_si``, ``woe_si_replay`` and
    ``eralg4`` all work through one path.
    """
    module = importlib.import_module("model." + str(args.model))
    n_outputs = int(state_dict["net.model.fc.weight"].shape[0])
    n_tasks = len(args.classes_per_task)
    model = module.Net(n_inputs, n_outputs, n_tasks, args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing key(s), e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected key(s), e.g. {unexpected[:3]}")
    model.eval()
    if not hasattr(model, "net"):
        raise TypeError(f"model {args.model} exposes no .net (ResNet1D) attribute")
    return model, n_outputs


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def _task_feature_mean(
    model, loader, device, limit: Optional[int], bn_training: bool
) -> torch.Tensor:
    """Mean penultimate feature over a task's test set (the DS centring point)."""
    total: Optional[torch.Tensor] = None
    count = 0
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        xb = batch[0].to(device)
        feats = model.net.forward_features(xb, bn_training=bn_training)
        summed = feats.sum(dim=0)
        total = summed if total is None else total + summed
        count += feats.shape[0]
    if total is None or count == 0:
        return torch.zeros(int(model.net.feature_dim), device=device)
    return total / float(count)


@torch.no_grad()
def evaluate(
    model,
    loaders: List[object],
    args: object,
    n_outputs: int,
    limit,
    bn_training: bool = True,
) -> Dict:
    """Replay every task's test set and accumulate the three matrices.

    Returns:
        Dict of ``numpy`` arrays: ``masked`` / ``unmasked`` counts
        ``(n_outputs, n_outputs)``, ``support`` and ``conflict`` sums over the
        same grid, and ``row_counts`` giving each true class's sample count.
    """
    device = next(model.net.parameters()).device
    classes_per_task = args.classes_per_task
    loader_name = getattr(args, "loader", None)

    shape = (n_outputs, n_outputs)
    masked = np.zeros(shape, dtype=np.int64)
    unmasked = np.zeros(shape, dtype=np.int64)
    support = np.zeros(shape, dtype=np.float64)
    counter = np.zeros(shape, dtype=np.float64)
    conflict = np.zeros(shape, dtype=np.float64)
    row_counts = np.zeros(n_outputs, dtype=np.int64)

    readout = model.net.model.fc
    all_columns = torch.arange(n_outputs, device=device)

    for task_index, loader in enumerate(loaders):
        scored_before = int(row_counts.sum())
        mu = _task_feature_mean(model, loader, device, limit, bn_training)
        for batch_index, batch in enumerate(loader):
            if limit is not None and batch_index >= limit:
                break
            xb = batch[0].to(device)
            yb = unpack_y_to_class_labels(batch[1]).long().view(-1).cpu().numpy()

            feats = model.net.forward_features(xb, bn_training=bn_training)
            logits = model.net.forward_classifier(feats, bn_training=bn_training)
            masked_logits = misc_utils.apply_task_incremental_logit_mask(
                logits,
                task_index,
                classes_per_task,
                n_outputs,
                cil_all_seen_upto_task=task_index,
                loader=loader_name,
            )

            pred_masked = masked_logits.argmax(dim=1).cpu().numpy()
            pred_unmasked = logits.argmax(dim=1).cpu().numpy()

            weights = compute_weights_of_evidence(
                feats,
                readout.weight[all_columns],
                readout.bias[all_columns],
                mu,
                centering_mode=str(
                    getattr(args, "woe_centering_mode", "centered_uniform")
                ),
            )
            w_plus, w_minus = per_class_total_evidence(weights)
            w_plus_np = w_plus.cpu().numpy()
            w_minus_np = w_minus.cpu().numpy()
            pair_conflict = (2.0 * w_plus * w_minus).cpu().numpy()

            valid = (yb >= 0) & (yb < n_outputs)
            np.add.at(masked, (yb[valid], pred_masked[valid]), 1)
            np.add.at(unmasked, (yb[valid], pred_unmasked[valid]), 1)
            np.add.at(support, yb[valid], w_plus_np[valid])
            np.add.at(counter, yb[valid], w_minus_np[valid])
            np.add.at(conflict, yb[valid], pair_conflict[valid])
            np.add.at(row_counts, yb[valid], 1)

        print(
            f"  task {task_index}: "
            f"{int(row_counts.sum()) - scored_before} samples scored",
            flush=True,
        )

    return {
        "masked": masked,
        "unmasked": unmasked,
        "support": support,
        "counter": counter,
        "conflict": conflict,
        "row_counts": row_counts,
    }


# ======================================================================
# Reporting
# ======================================================================
def class_to_task(classes_per_task: List[int]) -> np.ndarray:
    """Map each global class column to the task that owns it."""
    owner = []
    for task_index, count in enumerate(classes_per_task):
        owner.extend([task_index] * int(count))
    return np.array(owner, dtype=np.int64)


def row_normalise(matrix: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Divide each row by its sample count, leaving empty rows at zero.

    Row-normalising is what makes classes with very different support
    comparable; raw counts would just redraw the class-frequency histogram.
    """
    safe = np.where(counts > 0, counts, 1).astype(np.float64)
    normalised = matrix.astype(np.float64) / safe[:, None]
    normalised[counts == 0] = 0.0
    return normalised


def column_contrast(rates: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Strip additive row and column effects so only *specific* attraction remains.

    Raw mean ``w_plus`` is not comparable across columns: ``w+_k`` scales with
    ``||beta_k||``, and measured on the anchor-only run a column's mean support
    correlates **0.88** with its readout row norm (both span ~2.2x). A bright
    column therefore mostly means "this class has a large weight vector", not
    "this class attracts evidence" -- which is why the raw matrix has a diagonal
    exceeding its own column mean for only 30 of 64 classes.

    Subtracting a column baseline cancels that offset, leaving "how much *more*
    support does class ``k`` get on class ``c``'s inputs than it normally gets".
    The baseline is taken **within the true class's own task block**, not
    globally: ``mu`` is recomputed per task, so rows from different tasks are
    centred differently and a global baseline would mix reference points. Within
    a block every row shares one ``mu``, so the subtraction is exact.

    Reading: zero means "typical for this task", positive means class ``k`` is
    specifically excited by class ``c``'s inputs -- and off-diagonal positives
    are the confusions. Each task block's rows sum to zero per column by
    construction.

    Args:
        rates: Mean ``w_plus`` per (true class, class), shape ``(K, K)``.
        owner: Task index owning each class column.

    Returns:
        Signed contrast, same shape.
    """
    contrast = np.zeros_like(rates)
    for task in np.unique(owner):
        rows = np.flatnonzero(owner == task)
        block = rates[rows]
        present = block.any(axis=1)
        if not present.any():
            continue
        live = block[present]
        # Two-way additive model: subtract the column effect AND the row effect,
        # adding back the grand mean so the residual is the interaction alone.
        # Removing only the column effect leaves whole rows uniformly hot or
        # cold -- classes whose inputs are simply further from mu raise every
        # w+_k at once (visible as solid rows before this was added, the noise
        # class of each task being uniformly cold). That is a property of the
        # sample, not a confusion with any particular class.
        row_effect = live.mean(axis=1, keepdims=True)
        column_effect = live.mean(axis=0, keepdims=True)
        grand = live.mean()
        residual = np.zeros_like(block)
        residual[present] = live - row_effect - column_effect + grand
        contrast[rows] = residual
    return contrast


def top_confusions(
    matrix: np.ndarray, counts: np.ndarray, owner: np.ndarray, top: int = 25
) -> List[str]:
    """Rank off-diagonal mass, tagging whether the pair crosses a task boundary."""
    rates = row_normalise(matrix, counts)
    np.fill_diagonal(rates, 0.0)
    order = np.argsort(rates, axis=None)[::-1][:top]
    lines = []
    for flat in order:
        true_class, predicted = np.unravel_index(flat, rates.shape)
        rate = rates[true_class, predicted]
        if rate <= 0.0:
            break
        kind = "within" if owner[true_class] == owner[predicted] else "CROSS"
        lines.append(
            f"  {rate:6.3f}  class {true_class:2d} (T{owner[true_class]}) "
            f"-> {predicted:2d} (T{owner[predicted]})  [{kind}-task]"
        )
    return lines


def plot_matrix(
    matrix: np.ndarray,
    owner: np.ndarray,
    title: str,
    path: Path,
    diverging: bool = False,
    cbar_label: str = "row-normalised rate",
) -> None:
    """Render one grid as a heatmap with task blocks.

    Unsigned magnitude takes the sequential single-hue ramp on ``[0, 1]``. A
    signed contrast takes the diverging pair with a neutral midpoint pinned at
    zero and symmetric limits, so "no difference" reads as nothing and the two
    directions are visually equal. Limits for the diverging case come from the
    99th percentile of ``|matrix|`` rather than its max, so one extreme cell
    cannot flatten the rest of the plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if diverging:
        cmap = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING_RAMP)
        span = float(np.percentile(np.abs(matrix), 99)) or 1.0
        vmin, vmax = -span, span
    else:
        cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
        vmin, vmax = 0.0, 1.0

    figure, axes = plt.subplots(figsize=(11, 9.5), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    image = axes.imshow(
        matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )

    # Task-block separators: recessive, so the data reads first.
    boundaries = np.flatnonzero(np.diff(owner)) + 0.5
    for edge in boundaries:
        axes.axhline(edge, color=GRID_INK, linewidth=0.8)
        axes.axvline(edge, color=GRID_INK, linewidth=0.8)

    centres = [
        float(np.mean(np.flatnonzero(owner == task))) for task in np.unique(owner)
    ]
    labels = [f"T{task}" for task in np.unique(owner)]
    axes.set_xticks(centres)
    axes.set_yticks(centres)
    axes.set_xticklabels(labels, color=INK_SECONDARY, fontsize=9)
    axes.set_yticklabels(labels, color=INK_SECONDARY, fontsize=9)
    axes.set_xlabel(
        "predicted class (grouped by task)", color=INK_SECONDARY, fontsize=10
    )
    axes.set_ylabel("true class (grouped by task)", color=INK_SECONDARY, fontsize=10)
    axes.set_title(title, color=INK_PRIMARY, fontsize=12, pad=12)
    for spine in axes.spines.values():
        spine.set_color(GRID_INK)
    axes.tick_params(colors=GRID_INK, length=3)

    bar = figure.colorbar(image, ax=axes, fraction=0.045, pad=0.02)
    bar.set_label(cbar_label, color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    bar.outline.set_edgecolor(GRID_INK)

    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path}")


# ======================================================================
def process(
    run_dir: Path, out_root: Path, limit: Optional[int], bn_training: bool
) -> None:
    """Rebuild one run, evaluate it, and write matrices, plots and a summary."""
    print(f"[run] {run_dir}")
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
    results = evaluate(model, test_loaders, args, n_outputs, limit, bn_training)

    owner = class_to_task(list(args.classes_per_task))
    counts = results["row_counts"]
    out_dir = out_root / run_dir.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "matrices.npz", owner=owner, **results)

    plot_matrix(
        row_normalise(results["masked"], counts),
        owner,
        "Masked confusion — argmax within each sample's own task",
        out_dir / "confusion_masked.png",
    )
    plot_matrix(
        row_normalise(results["unmasked"], counts),
        owner,
        "Unmasked confusion — argmax over the full head (forgetting structure)",
        out_dir / "confusion_unmasked.png",
    )
    plot_matrix(
        column_contrast(row_normalise(results["support"], counts), owner),
        owner,
        "Class-specific SUPPORT — excess w_plus vs expected (blue = less support)",
        out_dir / "evidence_support_contrast.png",
        diverging=True,
        cbar_label="more (+) / less (-) w_plus than expected",
    )
    if "counter" in results:
        plot_matrix(
            column_contrast(row_normalise(results["counter"], counts), owner),
            owner,
            "Class-specific COUNTER-EVIDENCE — excess w_minus vs expected "
            "(red = more evidence AGAINST)",
            out_dir / "evidence_counter_contrast.png",
            diverging=True,
            cbar_label="more (+) / less (-) w_minus than expected",
        )
        plot_matrix(
            column_contrast(row_normalise(results["conflict"], counts), owner),
            owner,
            "Class-specific CONFLICT — excess 2*w_plus*w_minus vs expected "
            "(red = both channels large at once)",
            out_dir / "evidence_conflict_contrast.png",
            diverging=True,
            cbar_label="more (+) / less (-) conflict than expected",
        )

    per_class_conflict = row_normalise(results["conflict"], counts)
    # Accuracy floors. 1/n_outputs is the uniform-guesser floor and is invariant
    # to the class priors by construction, which makes it useless here: the test
    # pool is heavily skewed (task sizes span 21x, and a ~25% noise class sits in
    # every task), so the honest comparisons are against a majority-class
    # predictor and a prior-matched random one, both of which sit well above
    # 1/n_outputs under skew. Reported together so an accuracy is never read
    # against the weakest available baseline.
    total_samples = max(1, int(counts.sum()))
    priors = counts.astype(np.float64) / float(total_samples)
    baselines = {
        "uniform_random": 1.0 / float(n_outputs),
        "prior_matched_random": float(np.sum(priors**2)),
        "majority_class": float(priors.max()),
        "majority_class_id": int(priors.argmax()),
    }
    summary = {
        "run": str(run_dir),
        "model": str(args.model),
        "n_outputs": int(n_outputs),
        "classes_per_task": [int(c) for c in args.classes_per_task],
        "masked_accuracy": float(
            np.trace(results["masked"]) / max(1, int(counts.sum()))
        ),
        "unmasked_accuracy": float(
            np.trace(results["unmasked"]) / max(1, int(counts.sum()))
        ),
        "cross_task_leak": float(
            results["unmasked"][owner[:, None] != owner[None, :]].sum()
            / max(1, int(counts.sum()))
        ),
        "mean_self_conflict": float(
            np.mean([per_class_conflict[k, k] for k in range(n_outputs)])
        ),
        "baselines": baselines,
        "unmasked_vs_majority": float(
            np.trace(results["unmasked"]) / float(total_samples)
            - baselines["majority_class"]
        ),
        "balanced_masked_accuracy": float(
            np.mean(
                [
                    results["masked"][k, k] / counts[k]
                    for k in range(n_outputs)
                    if counts[k] > 0
                ]
            )
        ),
        "balanced_unmasked_accuracy": float(
            np.mean(
                [
                    results["unmasked"][k, k] / counts[k]
                    for k in range(n_outputs)
                    if counts[k] > 0
                ]
            )
        ),
        "total_samples": total_samples,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["MASKED (within-task only, by construction)"]
    lines += top_confusions(results["masked"], counts, owner)
    lines += ["", "UNMASKED (full head — where forgetting sends old classes)"]
    lines += top_confusions(results["unmasked"], counts, owner)
    (out_dir / "top_confusions.txt").write_text("\n".join(lines))
    print("\n".join(lines[:14]))
    print(f"  summary: {json.dumps(summary)}")


def replot(out_dir: Path) -> None:
    """Regenerate plots from a stored ``matrices.npz``, with no re-evaluation.

    The raw sums are what get saved, so a change to normalisation or colour
    encoding is a plotting change -- re-scoring 446k samples to redraw a figure
    would be wasted work.
    """
    payload = np.load(out_dir / "matrices.npz")
    owner, counts = payload["owner"], payload["row_counts"]
    plot_matrix(
        row_normalise(payload["masked"], counts),
        owner,
        "Masked confusion — argmax within each sample's own task",
        out_dir / "confusion_masked.png",
    )
    plot_matrix(
        row_normalise(payload["unmasked"], counts),
        owner,
        "Unmasked confusion — argmax over the full head (forgetting structure)",
        out_dir / "confusion_unmasked.png",
    )
    plot_matrix(
        column_contrast(row_normalise(payload["support"], counts), owner),
        owner,
        "Class-specific SUPPORT — excess w_plus vs expected (blue = less support)",
        out_dir / "evidence_support_contrast.png",
        diverging=True,
        cbar_label="more (+) / less (-) w_plus than expected",
    )
    if "counter" in payload:
        plot_matrix(
            column_contrast(row_normalise(payload["counter"], counts), owner),
            owner,
            "Class-specific COUNTER-EVIDENCE — excess w_minus vs expected "
            "(red = more evidence AGAINST)",
            out_dir / "evidence_counter_contrast.png",
            diverging=True,
            cbar_label="more (+) / less (-) w_minus than expected",
        )
        plot_matrix(
            column_contrast(row_normalise(payload["conflict"], counts), owner),
            owner,
            "Class-specific CONFLICT — excess 2*w_plus*w_minus vs expected "
            "(red = both channels large at once)",
            out_dir / "evidence_conflict_contrast.png",
            diverging=True,
            cbar_label="more (+) / less (-) conflict than expected",
        )

    for name in ("evidence_support.png", "evidence_contrast.png"):
        stale = out_dir / name
        if stale.exists():
            stale.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run directory holding results.pt (globs allowed; repeatable).",
    )
    parser.add_argument("--out", type=str, default="analysis/confusion")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Redraw figures from existing matrices.npz files under --out; "
        "no model rebuild, no re-scoring. --run is ignored.",
    )
    parser.add_argument(
        "--eval-bn",
        action="store_true",
        help=(
            "Score with frozen BatchNorm running statistics and dropout off "
            "(bn_training=False). NOT what the training harness does: "
            "model_forward_for_metric_loop reaches ResNet1D.forward with its "
            "default bn_training=True, so every reported number in this project "
            "uses *batch* statistics. Consecutive tasks are different radar "
            "datasets, so the two differ a lot -- frozen statistics score all ten "
            "tasks with whatever the last task left in the buffers. Default "
            "(off) reproduces the harness."
        ),
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Score only this many batches per task (0 = all). For smoke tests.",
    )
    args = parser.parse_args()

    if args.replot:
        for npz in sorted((REPO_ROOT / args.out).glob("*/matrices.npz")):
            print(f"[replot] {npz.parent}")
            replot(npz.parent)
        return

    limit = args.limit_batches if args.limit_batches > 0 else None
    out_root = REPO_ROOT / args.out
    if args.eval_bn:
        out_root = out_root.with_name(out_root.name + "_evalbn")
    targets: List[Path] = []
    for pattern in args.run:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            print(f"[skip] no match for {pattern}")
        targets.extend(Path(p) for p in expanded if os.path.isdir(p))

    if not targets:
        sys.exit("no run directories matched")
    for run_dir in targets:
        process(run_dir, out_root, limit, bn_training=not args.eval_bn)


if __name__ == "__main__":
    main()
