"""Read and plot metrics from a run's metrics directory.

Metrics are saved by main.py under args.log_dir/metrics/ as task0.npz, task1.npz, ...
Each task*.npz contains:
  - losses: per-step/epoch loss for that task
  - cls_tr_rec: per-step/epoch training recall
  - val_macro_rec: validation macro recall per task (length = task_index + 1)
    (older runs spell this ``val_acc``)

Usage:
    python scripts/plot_metrics.py logs/ctn/eclresm_test-2026-03-05_14-58-57-4091/0/metrics
    python scripts/plot_metrics.py logs/ctn/eclresm_test-2026-03-05_14-58-57-4091/0/metrics -o plots/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

# Use non-interactive backend only when saving to file (-o); otherwise plt.show() can open windows
_pre_parser = argparse.ArgumentParser()
_pre_parser.add_argument("-o", "--output-dir", type=Path, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.output_dir is not None:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metric_keys import extract_metric, first_present_key  # noqa: E402


def load_metrics(metrics_dir: Path) -> list[dict[str, Any]]:
    """Load all task*.npz files from metrics_dir into a list of dicts.

    Args:
        metrics_dir: Path to the metrics directory (contains task0.npz, task1.npz, ...).

    Returns:
        List of dicts, one per task, each holding that task's stored metric
        arrays (``losses``, ``tr_macro_rec``, ``val_macro_rec``, ...). Legacy
        key spellings from older runs are preserved as-is; read them through
        :func:`utils.metric_keys.extract_metric`.
    """
    task_files = sorted(
        metrics_dir.glob("task*.npz"),
        key=lambda p: _task_index(p.name),
    )
    if not task_files:
        raise FileNotFoundError(f"No task*.npz files in {metrics_dir}")

    tasks = []
    for path in task_files:
        data = np.load(path, allow_pickle=True)
        task_data = {key: np.asarray(data[key]) for key in data.files}
        task_idx = _task_index(path.name)
        # Metrics like val_acc / val_f1 might contain intermediate epoch
        # validations flattened. The correct length for the final validation
        # after this task is task_idx + 1.
        num_tasks_seen = task_idx + 1
        for key in ("val_macro_rec", "val_acc", "val_macro_f1", "val_f1"):
            if key in task_data and len(task_data[key]) > num_tasks_seen:
                task_data[key] = task_data[key][-num_tasks_seen:]
        tasks.append(task_data)
    return tasks


def _task_index(filename: str) -> int:
    """Extract task number from filename like task0.npz or task12.npz."""
    match = re.match(r"task(\d+)\.npz", filename, re.IGNORECASE)
    if not match:
        return -1
    return int(match.group(1))


def load_n_epochs(metrics_dir: Path) -> int | None:
    """Read n_epochs from training_parameters.json in the run directory (parent of metrics_dir)."""
    params_file = metrics_dir.parent / "training_parameters.json"
    if not params_file.is_file():
        return None
    try:
        with open(params_file) as f:
            params = json.load(f)
        return params.get("n_epochs")
    except (json.JSONDecodeError, TypeError):
        return None


def _aggregate_per_epoch(values: np.ndarray, n_epochs: int) -> np.ndarray:
    """Average per-step values into per-epoch (one value per epoch)."""
    n = len(values)
    steps_per_epoch = n // n_epochs
    if steps_per_epoch == 0:
        return np.array([np.mean(values)]) if n > 0 else np.array([])
    n_use = steps_per_epoch * n_epochs
    return np.mean(values[:n_use].reshape(n_epochs, steps_per_epoch), axis=1)


def load_task_names(metrics_dir: Path) -> list[str] | None:
    """Load task names from a ``task_order.txt`` file if present.

    The file is expected to live inside ``metrics_dir`` and contain one task
    name per line in task index order.

    Args:
        metrics_dir: Path to the metrics directory.

    Returns:
        A list of task names, or ``None`` if the file does not exist or is
        empty.
    """
    order_file = metrics_dir / "task_order.txt"
    if not order_file.is_file():
        return None

    names: list[str] = []
    with order_file.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                names.append(stripped)
    return names or None


def _group_for_task_name(task_name: str) -> str:
    """Return a semantic group label for a task name."""
    name = task_name.lower()
    if "rcn" in name:
        return "rcn"
    if "deeprad" in name:
        return "deeprad"
    if "uclresm" in name:
        return "uclresm"
    return "other"


def get_task_color(task_idx: int, task_names: Sequence[str] | None = None):
    """Assign a colour for a task, grouping by task name when available.

    This matches the grouping logic used in ``scripts/plot_multi_algorithms.py``
    so that colours are consistent between single-run and multi-algorithm
    plots.
    """
    if task_names is None or task_idx >= len(task_names):
        group1 = [3, 4, 6, 9]
        group2 = [0, 1, 8]
        group3 = [2, 5, 7]

        if task_idx in group1:
            cmap = plt.get_cmap("Blues")
            idx = group1.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group1)))
        if task_idx in group2:
            cmap = plt.get_cmap("Oranges")
            idx = group2.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group2)))
        if task_idx in group3:
            cmap = plt.get_cmap("Greens")
            idx = group3.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group3)))
        return f"C{task_idx % 10}"

    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(task_names):
        group = _group_for_task_name(name)
        groups.setdefault(group, []).append(idx)

    task_name = task_names[task_idx]
    group = _group_for_task_name(task_name)
    group_indices = groups.get(group, [task_idx])
    position_in_group = group_indices.index(task_idx)
    group_size = len(group_indices)

    def _shade(index: int, size: int) -> float:
        if size <= 1:
            return 0.7
        return 0.4 + 0.6 * (index / float(size))

    cmap_name = {
        "rcn": "Oranges",
        "deeprad": "Blues",
        "uclresm": "Greens",
        "other": "Greys",
    }[group]
    cmap = plt.get_cmap(cmap_name)
    return cmap(_shade(position_in_group, group_size))


def plot_per_task_curves(
    tasks: list[dict[str, Any]],
    output_dir: Path | None,
    task_names: Sequence[str] | None = None,
) -> None:
    """Plot loss and training recall per task (steps within task)."""
    n_tasks = len(tasks)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for task_idx, task in enumerate(tasks):
        steps = np.arange(len(task["losses"]))
        acc_values = extract_metric(task, "tr_macro_rec")
        if acc_values is None:
            acc_values = task.get("tr_acc")
        if acc_values is None:
            raise KeyError(
                "Neither 'tr_macro_rec' (or legacy 'cls_tr_rec') nor 'tr_acc' "
                "found in task metrics."
            )
        c = get_task_color(task_idx, task_names)
        axes[0].plot(
            steps, task["losses"], label=f"Task {task_idx}", color=c, alpha=0.8
        )
        axes[1].plot(
            steps,
            np.asarray(acc_values, dtype=float),
            label=f"Task {task_idx}",
            color=c,
            alpha=0.8,
        )

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per task (per step)")
    axes[0].legend(ncol=min(n_tasks, 5), fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Train recall")
    axes[1].set_xlabel("Step / epoch index")
    axes[1].set_title("Training cls recall per task")
    axes[1].legend(ncol=min(n_tasks, 5), fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        fig.savefig(
            output_dir / "per_task_loss_and_acc.png", dpi=150, bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


def plot_per_epoch_curves(
    tasks: list[dict[str, Any]],
    n_epochs: int,
    output_dir: Path | None,
    task_names: Sequence[str] | None = None,
) -> None:
    """Plot mean loss and mean training recall per epoch (one point per epoch per task)."""
    n_tasks = len(tasks)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for task_idx, task in enumerate(tasks):
        loss_ep = _aggregate_per_epoch(task["losses"], n_epochs)
        acc_values = extract_metric(task, "tr_macro_rec")
        if acc_values is None:
            acc_values = task.get("tr_acc")
        if acc_values is None:
            raise KeyError(
                "Neither 'tr_macro_rec' (or legacy 'cls_tr_rec') nor 'tr_acc' "
                "found in task metrics."
            )
        acc_ep = _aggregate_per_epoch(np.asarray(acc_values, dtype=float), n_epochs)
        epochs = np.arange(len(loss_ep))
        c = get_task_color(task_idx, task_names)
        axes[0].plot(
            epochs, loss_ep, "o-", label=f"Task {task_idx}", color=c, alpha=0.8
        )
        axes[1].plot(epochs, acc_ep, "o-", label=f"Task {task_idx}", color=c, alpha=0.8)

    axes[0].set_ylabel("Loss (mean)")
    axes[0].set_title("Loss per epoch (mean over steps)")
    axes[0].legend(ncol=min(n_tasks, 5), fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Train recall (mean)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Training cls recall per epoch (mean over steps)")
    axes[1].legend(ncol=min(n_tasks, 5), fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        fig.savefig(
            output_dir / "per_epoch_loss_and_acc.png", dpi=150, bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


def plot_final_validation(
    tasks: list[dict[str, Any]],
    output_dir: Path | None,
    task_names: Sequence[str] | None = None,
) -> None:
    """Plot per-task final validation macro recall from the last task.

    Args:
        tasks: Loaded per-task metric dicts, in task order.
        output_dir: Directory to write ``final_validation.png`` into, or ``None``
            to leave the figure open for an interactive ``plt.show()``.
        task_names: Optional task names used for consistent per-task colouring.

    Returns:
        None.

    Usage:
        plot_final_validation(tasks, Path("plots"), task_names)
    """
    if not tasks:
        return

    val_recall = extract_metric(tasks[-1], "val_macro_rec")
    if val_recall is None:
        return

    recall_values = np.asarray(val_recall, dtype=float)
    n_tasks = recall_values.size
    if n_tasks == 0:
        return
    task_indices = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [get_task_color(i, task_names) for i in range(n_tasks)]
    ax.bar(task_indices, recall_values, width=0.6, label="Cls Recall", color=colors)

    ax.set_xlabel("Task")
    ax.set_ylabel("Metric value")
    ax.set_title("Final validation metrics (after all tasks)")
    ax.set_xticks(task_indices)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "final_validation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # else: leave figure open for plt.show() at end of main


def plot_validation_over_time(
    tasks: list[dict[str, Any]],
    output_dir: Path | None,
    task_names: Sequence[str] | None = None,
    val_metric_key: str = "val_macro_rec",
    val_metric_label: str = "Macro Recall",
) -> None:
    """Plot a validation metric per task as more tasks are trained (metric matrix).

    Includes an aggregate curve where each point k is the mean metric over
    tasks 0..k at checkpoint k.
    """
    n_tasks = len(tasks)
    # After training task k, we have val_acc of length k+1 (tasks 0..k)
    x = np.arange(1, n_tasks + 1)  # "after task 0", "after task 1", ...

    fig, ax = plt.subplots(figsize=(9, 5))
    for task_idx in range(n_tasks):
        # For each task t, get its val acc from task files task_idx, task_idx+1, ..., n_tasks-1
        ys = []
        for k in range(n_tasks):
            metrics_k = tasks[k]
            metric_vals = metrics_k.get(val_metric_key)
            if metric_vals is None or task_idx >= len(metric_vals):
                ys.append(np.nan)
                continue
            ys.append(float(metric_vals[task_idx]))
        ax.plot(
            x,
            ys,
            "o-",
            label=f"Task {task_idx}",
            color=get_task_color(task_idx, task_names),
            alpha=0.8,
        )

    # Aggregate curve: at checkpoint k, mean over tasks 0..k. Skip missing/NaN.
    agg_vals: list[float] = []
    for k in range(n_tasks):
        metrics_k = tasks[k]
        metric_vals_k = metrics_k.get(val_metric_key)
        if metric_vals_k is None:
            agg_vals.append(float("nan"))
            continue
        window: list[float] = []
        for t in range(k + 1):
            if t >= len(metric_vals_k):
                continue
            v = float(metric_vals_k[t])
            if np.isnan(v):
                continue
            window.append(v)
        if window:
            agg_vals.append(float(np.mean(window)))
        else:
            agg_vals.append(float("nan"))

    ax.plot(
        x,
        np.asarray(agg_vals, dtype=float),
        "k^-",
        linewidth=2.0,
        markersize=4.0,
        label=f"Mean {val_metric_label}",
    )

    ax.set_xlabel("After training up to task")
    ax.set_ylabel(f"Validation ({val_metric_label})")
    ax.set_title(f"Validation {val_metric_label} per task over continual learning")
    ax.legend(ncol=min(n_tasks, 5), fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "val_acc_over_tasks.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # else: leave figure open for plt.show() at end of main


def compute_average_forgetting(
    tasks: list[dict[str, Any]],
    val_metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute average forgetting over all previous tasks at each checkpoint.

    This uses a single validation metric key (e.g. ``val_f1`` or ``val_acc``)
    consistently for peak and current values.

    For each task t, peak metric = metric[t] after training task t
    (i.e. tasks[t][val_metric_key][t]). After training
    task K > t, the metric on task t is tasks[K][val_metric_key][t]. Forgetting for task t at K is
    max(0, peak[t] - current_metric).
    Average forgetting at K = mean over t in 0..K-1 (no previous tasks when K=0).

    Returns:
        x: checkpoint indices (0, 1, ..., n_tasks-1), "after training task K"
        avg_forgetting: average forgetting at each checkpoint (length n_tasks)
    """
    n_tasks = len(tasks)
    # Peak metric after learning each task t.
    peak_vals: list[float] = []
    for t in range(n_tasks):
        metrics_t = tasks[t]
        metric_vals = metrics_t.get(val_metric_key)
        if metric_vals is None or len(metric_vals) <= t:
            continue
        peak_vals.append(float(metric_vals[t]))
    peak_vals_arr = np.asarray(peak_vals, dtype=float)
    avg_forgetting = np.zeros(n_tasks)
    for k in range(n_tasks):
        if k == 0:
            avg_forgetting[k] = 0.0
            continue
        forgets: list[float] = []
        for t in range(k):
            metrics_k = tasks[k]
            metric_vals_k = metrics_k.get(val_metric_key)
            if metric_vals_k is None or len(metric_vals_k) <= t:
                # If we do not have a value, treat as no additional forgetting signal.
                continue
            current_metric = float(metric_vals_k[t])
            forgets.append(max(0.0, float(peak_vals_arr[t] - current_metric)))
        avg_forgetting[k] = float(np.mean(forgets)) if forgets else 0.0
    return np.arange(n_tasks), avg_forgetting


def plot_average_forgetting(
    tasks: list[dict[str, Any]],
    output_dir: Path | None,
    val_metric_key: str,
    val_metric_label: str,
) -> None:
    """Plot average forgetting over previous tasks vs checkpoint (after training task K).

    This uses a single validation metric (e.g. Total F1 or classification
    recall) resolved in ``main`` and passed via ``val_metric_key``.
    """
    x, avg_forgetting = compute_average_forgetting(tasks, val_metric_key)
    y_label = f"Average forgetting ({val_metric_label})"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, avg_forgetting, "o-", color="C0", linewidth=2, markersize=6)
    ax.set_xlabel("After training up to task")
    ax.set_ylabel(y_label)
    ax.set_title(f"Average forgetting on all previous tasks ({val_metric_label})")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "average_forgetting.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # else: leave figure open for plt.show() at end of main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read and plot metrics from a run's metrics directory.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("logs/ctn/eclresm_test-2026-03-05_14-58-57-4091/0/metrics"),
        nargs="?",
        help="Path to the metrics directory (contains task0.npz, task1.npz, ...).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plot images. If not set, only display.",
    )
    parser.add_argument(
        "--val-metric",
        type=str,
        choices=("total_f1", "cls_recall"),
        default="total_f1",
        help=(
            "Validation metric to use for 'val over tasks' and average forgetting "
            "plots. Defaults to total_f1; falls back to cls_recall if F1 is not "
            "available in the metrics."
        ),
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir.resolve()
    if not metrics_dir.is_dir():
        raise SystemExit(f"Not a directory: {metrics_dir}")

    import matplotlib

    if args.output_dir is None and matplotlib.get_backend().lower() == "agg":
        print(
            f"Non-interactive backend detected. Defaulting to saving plots in {metrics_dir}"
        )
        args.output_dir = metrics_dir

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_metrics(metrics_dir)
    print(f"Loaded {len(tasks)} task(s) from {metrics_dir}")

    # Resolve which validation metric to use for plots that depend on a single
    # validation signal (validation over tasks and average forgetting).
    recall_key = first_present_key(tasks[0], ["val_macro_rec"]) or "val_macro_rec"
    if args.val_metric == "cls_recall":
        val_metric_key = recall_key
        val_metric_label = "Macro Recall"
    else:
        # Prefer macro F1 when available; otherwise fall back to macro recall.
        f1_key = next(
            (k for t in tasks if (k := first_present_key(t, ["val_macro_f1"]))), None
        )
        if f1_key is not None:
            val_metric_key = f1_key
            val_metric_label = "Macro F1"
        else:
            val_metric_key = recall_key
            val_metric_label = "Macro Recall"
            print(
                "[WARN] Requested val-metric=total_f1 but no macro-F1 key found in "
                "metrics; falling back to macro recall."
            )

    task_names = load_task_names(metrics_dir)

    plot_per_task_curves(tasks, args.output_dir, task_names=task_names)

    n_epochs = load_n_epochs(metrics_dir)
    if n_epochs is not None:
        plot_per_epoch_curves(tasks, n_epochs, args.output_dir, task_names=task_names)
    else:
        print("Skipping per-epoch plot (no n_epochs in training_parameters.json)")

    plot_final_validation(tasks, args.output_dir, task_names=task_names)
    plot_validation_over_time(
        tasks,
        args.output_dir,
        task_names=task_names,
        val_metric_key=val_metric_key,
        val_metric_label=val_metric_label,
    )
    plot_average_forgetting(
        tasks,
        args.output_dir,
        val_metric_key=val_metric_key,
        val_metric_label=val_metric_label,
    )

    if args.output_dir:
        print(f"Plots saved under {args.output_dir}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
