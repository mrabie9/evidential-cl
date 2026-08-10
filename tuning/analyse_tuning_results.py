"""Utility for inspecting La-MAML tuning summary files.

This script reads the JSON output produced by tuning runs and provides
basic analytics together with an optional heatmap visualisation of the
validation scores across the explored hyper-parameter grid.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")  # Allows running in headless environments.
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - matplotlib not always installed.
    plt = None
    _HAS_MPL = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse and visualise tuning summary files produced by the tuning scripts",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Path to a tuning summary JSON file (e.g. logs/tuning/<model>/.../summary.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top trials to display in the textual report (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the generated visualisation (PNG). Defaults to <summary_dir>/tuning_heatmap.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots with matplotlib after saving (has no effect if matplotlib is unavailable)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation and only print the textual analytics",
    )
    parser.add_argument(
        "--plot-params",
        nargs="+",
        default=None,
        help="Override plot axes with 2 or 3 param names (e.g. --plot-params lr beta [memory_strength])",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        params = item.get("params", {})
        raw_scores = item.get("val_per_task") or []
        val_scores = [
            float(score) for score in raw_scores if isinstance(score, (int, float))
        ]
        val_min = min(val_scores) if val_scores else float("nan")
        val_max = max(val_scores) if val_scores else float("nan")
        if len(val_scores) > 1:
            val_std = statistics.pstdev(val_scores)
        elif val_scores:
            val_std = 0.0
        else:
            val_std = float("nan")
        rows.append(
            {
                "trial": item.get("trial"),
                "status": item.get("status"),
                "stage": item.get("stage"),
                "seed": item.get("seed"),
                "params": params,
                "score": item.get("score", float("nan")),
                "val_mean": item.get("val_mean", float("nan")),
                "val_det_mean": item.get("val_det_mean", float("nan")),
                "val_pfa_mean": item.get("val_pfa_mean", float("nan")),
                "val_min": val_min,
                "val_max": val_max,
                "val_std": val_std,
                "test_det_mean": item.get("test_det_mean", float("nan")),
                "test_pfa_mean": item.get("test_pfa_mean", float("nan")),
                "duration_sec": item.get("duration_sec"),
                "log_dir": item.get("log_dir"),
            }
        )
    return rows


def fmt_float(value: Any, precision: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if isinstance(value, (int, float)):
        magnitude = abs(value)
        if magnitude and (
            magnitude < 10**-precision or magnitude >= 10 ** (precision + 1)
        ):
            return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def fmt_value(value: Any, precision: int = 4) -> str:
    if isinstance(value, (int, float)):
        return fmt_float(value, precision)
    return str(value)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _reconstruct_hierarchical_params(
    summary: Dict[str, Any], param_names: Sequence[str]
) -> Dict[str, Any]:
    """Rebuild final hierarchical hyperparameters from per-stage trial winners."""
    search_space = summary.get("search_space")
    if not isinstance(search_space, dict):
        return {}

    stage_winners: Dict[str, Dict[str, Any]] = {}
    for result in summary.get("results", []):
        if result.get("status") != "ok":
            continue
        stage = result.get("stage")
        if stage not in search_space:
            continue
        current = stage_winners.get(stage)
        score = result.get("score")
        if not is_finite_number(score):
            continue
        if current is None or float(score) > float(current.get("score", float("-inf"))):
            stage_winners[stage] = result

    combined: Dict[str, Any] = {}
    for stage in search_space:
        winner = stage_winners.get(stage)
        if winner is None:
            continue
        value = (winner.get("trial_params") or {}).get(stage)
        if value is None:
            value = (winner.get("params") or {}).get(stage)
        if value is not None:
            combined[stage] = value
    if not combined:
        return {}
    return {name: combined.get(name) for name in param_names}


def resolve_best_display_params(
    summary: Dict[str, Any], best: Dict[str, Any], param_names: Sequence[str]
) -> Dict[str, Any]:
    """Return the hyperparameters that should be shown for the best trial.

    Hierarchical sweeps store per-stage ``params`` on each trial, so the global
    ``best`` entry may only contain the parameter tuned in that stage. Prefer
    ``hierarchical_final_params`` when present, otherwise merge fixed and trial
    overrides into the reported ``params`` mapping.

    Args:
        summary: Parsed tuning summary JSON.
        best: Best-trial record from the summary.
        param_names: Hyperparameter names to include in the output mapping.

    Returns:
        Mapping from hyperparameter name to value for display/writeback hints.
    """
    hierarchical_params = summary.get("hierarchical_final_params")
    if isinstance(hierarchical_params, dict) and hierarchical_params:
        return {name: hierarchical_params.get(name) for name in param_names}

    if summary.get("hierarchical") and param_names:
        reconstructed = _reconstruct_hierarchical_params(summary, param_names)
        if reconstructed:
            return reconstructed

    merged: Dict[str, Any] = {}
    merged.update(best.get("fixed_params") or {})
    merged.update(best.get("trial_params") or {})
    merged.update(best.get("params") or {})
    return {name: merged.get(name) for name in param_names}


def infer_param_names(
    summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]
) -> List[str]:
    search_space = summary.get("search_space")
    if isinstance(search_space, dict) and search_space:
        return list(search_space.keys())
    names: List[str] = []
    seen = set()
    for row in rows:
        params = row.get("params") or {}
        for key in params.keys():
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def infer_primary_metric(rows: Sequence[Dict[str, Any]]) -> str:
    if any(is_finite_number(row.get("score")) for row in rows):
        return "score"
    return "val_mean"


def print_header(
    summary: Dict[str, Any], rows: Sequence[Dict[str, Any]], param_names: Sequence[str]
) -> None:
    print("\n=== Tuning summary ===")
    print(f"Config file      : {summary.get('config')}")
    print(f"Experiment name  : {summary.get('base_expt_name')}")
    print(f"Session directory: {summary.get('session_dir')}")
    print(f"Trials completed : {len(rows)} / {summary.get('num_trials')}\n")

    best = summary.get("best") or {}
    best_params = resolve_best_display_params(summary, best, param_names)
    print("Best trial (from summary)")
    if best.get("stage") == "stage2":
        print(
            "  note: winner selected by stage-2 MEAN over seeds "
            f"{summary.get('stage2_seeds')} (std {fmt_float(best.get('stage2_std'))}); "
            "single-seed rows in 'Top trials' may rank differently."
        )
    elif summary.get("hierarchical") and best.get("stage") != list(param_names)[-1]:
        print(
            "  note: hierarchical sweep; displayed params are the final combined "
            f"values (best trial #{best.get('trial')} is from stage '{best.get('stage')}')."
        )
    best_score = best.get("score")
    if is_finite_number(best_score):
        print(
            "  trial #{} | score={} | val_mean={} | det_mean={} | pfa_mean={}".format(
                best.get("trial"),
                fmt_float(best_score),
                fmt_float(best.get("val_mean")),
                fmt_float(best.get("val_det_mean")),
                fmt_float(best.get("val_pfa_mean")),
            )
        )
    else:
        print(
            f"  trial #{best.get('trial')} | val_mean={fmt_float(best.get('val_mean'))}"
        )
    if param_names:
        param_text = ", ".join(
            f"{name}={fmt_value(best_params.get(name), 5)}" for name in param_names
        )
        print(f"  params: {param_text}")
    else:
        print("  params: (none)")
    print(f"  duration: {fmt_float(best.get('duration_sec'), 2)} sec")


def print_top_trials(
    rows: Sequence[Dict[str, Any]],
    k: int,
    param_names: Sequence[str],
    metric_key: str,
) -> None:
    def sort_key(item: Dict[str, Any]) -> tuple[float, int, int]:
        value = item.get(metric_key)
        score_value = float(value) if is_finite_number(value) else float("-inf")
        params = item.get("params") or {}
        trial_index = item.get("trial")
        trial_number = int(trial_index) if isinstance(trial_index, int) else -1
        return (score_value, len(params), trial_number)

    valid_rows = sorted(rows, key=sort_key, reverse=True)
    print(f"\nTop trials by {metric_key}")
    if any(row.get("stage") == "stage2-seed" for row in rows):
        print(
        "  note: rows are SINGLE-SEED runs (stage-2 re-runs included, seed column"
        " set); the sweep's winner is chosen by the stage-2 mean, not this table."
        )
    param_cols = [(name, max(len(name), 8)) for name in param_names]
    param_header = " ".join(f"{name:>{width}}" for name, width in param_cols)
    header = (
        f"{'rank':>4} {'trial':>5} {'seed':>5} {metric_key:>10} {'val_mean':>10} "
        f"{'det_mean':>10} {'pfa_mean':>10} {'val_std':>10} {'duration_s':>11}"
    )
    if param_header:
        header = f"{header} {param_header}"
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(valid_rows[:k], start=1):
        params = row.get("params") or {}
        param_values = " ".join(
            f"{fmt_value(params.get(name), 5):>{width}}" for name, width in param_cols
        )
        seed = row.get("seed")
        print(
            f"{idx:>4} {row.get('trial', ''):>5} {('' if seed is None else seed):>5} "
            f"{fmt_float(row.get(metric_key)):>10} "
            f"{fmt_float(row.get('val_mean')):>10} {fmt_float(row.get('val_det_mean')):>10} "
            f"{fmt_float(row.get('val_pfa_mean')):>10} {fmt_float(row.get('val_std')):>10} "
            f"{fmt_float(row.get('duration_sec'), 2):>11}"
            f"{' ' + param_values if param_values else ''}"
        )


def print_stage2_ranking(summary: Dict[str, Any], param_names: Sequence[str]) -> None:
    """Print the multi-seed stage-2 ranking; this is what selects the winner."""
    aggregates = summary.get("stage2_aggregates") or []
    if not aggregates:
        return
    seeds = summary.get("stage2_seeds") or []
    print(f"\nStage-2 multi-seed ranking (seeds {seeds}) — selection basis")
    header = f"{'rank':>4} {'mean':>10} {'std':>10} {'per-seed scores':>32}  params"
    print(header)
    print("-" * len(header))
    ordered = sorted(
        aggregates,
        key=lambda a: a.get("score") if is_finite_number(a.get("score")) else float("-inf"),
        reverse=True,
    )
    for idx, agg in enumerate(ordered, start=1):
        by_seed = agg.get("stage2_scores_by_seed") or {}
        per_seed = ", ".join(f"{k}:{fmt_float(v)}" for k, v in sorted(by_seed.items()))
        param_text = ", ".join(
            f"{k}={fmt_value(v, 5)}" for k, v in (agg.get("trial_params") or {}).items()
        )
        print(
            f"{idx:>4} {fmt_float(agg.get('score')):>10} {fmt_float(agg.get('stage2_std')):>10} "
            f"{per_seed:>32}  {param_text}"
        )


def summarise_by_param(
    rows: Iterable[Dict[str, Any]],
    param: str,
    metric_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[float]] = defaultdict(list)
    for row in rows:
        params = row.get("params") or {}
        value = params.get(param)
        metric_value = row.get(metric_key)
        if value is None or not is_finite_number(metric_value):
            continue
        grouped[value].append(float(metric_value))

    summary_rows: List[Dict[str, Any]] = []
    for value, scores in grouped.items():
        arr = [float(score) for score in scores]
        summary_rows.append(
            {
                param: value,
                "count": len(arr),
                "mean": statistics.mean(arr),
                "std": statistics.pstdev(arr) if len(arr) > 1 else float("nan"),
                "best": max(arr),
            }
        )
    summary_rows.sort(key=lambda item: item["mean"], reverse=True)
    return summary_rows


def print_param_summaries(
    rows: Sequence[Dict[str, Any]],
    param_names: Sequence[str],
    metric_key: str,
) -> None:
    for param in param_names:
        summary_rows = summarise_by_param(rows, param, metric_key)
        if not summary_rows:
            continue
        print(f"\nAverages grouped by {param} (metric={metric_key})")
        header = f"{param:>12} {'count':>7} {'mean':>10} {'std':>10} {'best':>10}"
        print(header)
        print("-" * len(header))
        for item in summary_rows:
            print(
                f"{fmt_float(item[param], 5):>12} {item['count']:>7} {fmt_float(item['mean']):>10} "
                f"{fmt_float(item['std']):>10} {fmt_float(item['best']):>10}"
            )


def collect_param_values(rows: Sequence[Dict[str, Any]], param: str) -> List[float]:
    values = {
        (row.get("params") or {}).get(param)
        for row in rows
        if is_finite_number((row.get("params") or {}).get(param))
    }
    return sorted(values)


def build_score_matrix(
    rows: Sequence[Dict[str, Any]],
    x_param: str,
    y_param: str,
    metric_key: str,
):
    x_values = collect_param_values(rows, x_param)
    y_values = collect_param_values(rows, y_param)
    if not x_values or not y_values:
        return None

    matrix: List[List[float]] = [[float("nan") for _ in x_values] for _ in y_values]
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}

    for row in rows:
        params = row.get("params") or {}
        x_val = params.get(x_param)
        y_val = params.get(y_param)
        metric_value = row.get(metric_key)
        if not (
            is_finite_number(x_val)
            and is_finite_number(y_val)
            and is_finite_number(metric_value)
        ):
            continue
        matrix[y_index[y_val]][x_index[x_val]] = float(metric_value)

    return matrix, x_values, y_values


def build_score_tensor(
    rows: Sequence[Dict[str, Any]],
    x_param: str,
    y_param: str,
    slice_param: str,
    metric_key: str,
):
    x_values = collect_param_values(rows, x_param)
    y_values = collect_param_values(rows, y_param)
    slice_values = collect_param_values(rows, slice_param)
    if not x_values or not y_values or not slice_values:
        return None

    tensor: List[List[List[float]]] = []
    for _ in slice_values:
        tensor.append([[float("nan") for _ in x_values] for _ in y_values])

    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    slice_index = {value: idx for idx, value in enumerate(slice_values)}

    for row in rows:
        params = row.get("params") or {}
        x_val = params.get(x_param)
        y_val = params.get(y_param)
        slice_val = params.get(slice_param)
        metric_value = row.get(metric_key)
        if not (
            is_finite_number(x_val)
            and is_finite_number(y_val)
            and is_finite_number(slice_val)
            and is_finite_number(metric_value)
        ):
            continue
        tensor[slice_index[slice_val]][y_index[y_val]][x_index[x_val]] = float(
            metric_value
        )

    return tensor, x_values, y_values, slice_values


def plot_heatmap_2d(
    matrix,
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_label: str,
    y_label: str,
    metric_key: str,
    output_path: Path,
    show: bool,
) -> None:
    if not _HAS_MPL:
        print("Matplotlib is not available; skipping plot generation.")
        return

    finite_scores = [
        value for row in matrix for value in row if is_finite_number(value)
    ]
    if not finite_scores:
        print("No finite validation scores available for plotting.")
        return

    vmin, vmax = min(finite_scores), max(finite_scores)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    fig.suptitle(f"{metric_key} heatmap")

    im = ax.imshow(
        matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis"
    )
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{value:g}" for value in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([f"{value:g}" for value in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for y_idx, _ in enumerate(y_values):
        for x_idx, _ in enumerate(x_values):
            value = matrix[y_idx][x_idx]
            if is_finite_number(value):
                ax.text(
                    x_idx,
                    y_idx,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(output_path, dpi=200)
    print(f"Saved heatmap visualisation to {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_heatmaps_3d(
    score_tensor,
    x_values: Sequence[float],
    y_values: Sequence[float],
    slice_values: Sequence[float],
    x_label: str,
    y_label: str,
    slice_label: str,
    metric_key: str,
    output_path: Path,
    show: bool,
) -> None:
    if not _HAS_MPL:
        print("Matplotlib is not available; skipping plot generation.")
        return

    finite_scores = [
        value
        for matrix in score_tensor
        for row in matrix
        for value in row
        if is_finite_number(value)
    ]
    if not finite_scores:
        print("No finite validation scores available for plotting.")
        return

    vmin, vmax = min(finite_scores), max(finite_scores)

    cols = min(len(slice_values), 3)
    rows = int(math.ceil(len(slice_values) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.8 * cols, 4.0 * rows), squeeze=False
    )
    fig.suptitle(f"{metric_key} heatmaps by {slice_label}")

    for idx, slice_val in enumerate(slice_values):
        ax = axes[idx // cols][idx % cols]
        matrix = score_tensor[idx]
        im = ax.imshow(
            matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis"
        )
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(
            [f"{value:g}" for value in x_values], rotation=45, ha="right"
        )
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([f"{value:g}" for value in y_values])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{slice_label} = {slice_val:g}")

        for y_idx, _ in enumerate(y_values):
            for x_idx, _ in enumerate(x_values):
                value = matrix[y_idx][x_idx]
                if is_finite_number(value):
                    ax.text(
                        x_idx,
                        y_idx,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric_key)

    for idx in range(len(slice_values), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    print(f"Saved heatmap visualisation to {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.summary is None:
        raise SystemExit(
            "Please provide --summary pointing to a tuning summary JSON file."
        )
    summary = load_summary(args.summary)
    rows = flatten_results(summary.get("results", []))
    param_names = infer_param_names(summary, rows)

    metric_key = infer_primary_metric(rows)
    print_header(summary, rows, param_names)
    print_stage2_ranking(summary, param_names)
    print_top_trials(rows, args.top_k, param_names, metric_key)
    print_param_summaries(rows, param_names, metric_key)

    if args.no_plot:
        return

    plot_params = args.plot_params
    if plot_params is None:
        candidates = [
            name for name in param_names if len(collect_param_values(rows, name)) > 1
        ]
        plot_params = candidates[:3]

    if len(plot_params) == 2:
        matrix_info = build_score_matrix(
            rows, plot_params[0], plot_params[1], metric_key
        )
        if matrix_info is None:
            print("Insufficient hyper-parameter coverage to build a heatmap.")
            return
        matrix, x_values, y_values = matrix_info
        output_path = args.output
        if output_path is None:
            output_path = args.summary.parent / "tuning_heatmap.png"
        plot_heatmap_2d(
            matrix,
            x_values,
            y_values,
            plot_params[0],
            plot_params[1],
            metric_key,
            output_path,
            args.show,
        )
        return

    if len(plot_params) >= 3:
        tensor_info = build_score_tensor(
            rows, plot_params[0], plot_params[1], plot_params[2], metric_key
        )
        if tensor_info is None:
            print("Insufficient hyper-parameter coverage to build a heatmap.")
            return
        score_tensor, x_values, y_values, slice_values = tensor_info
        output_path = args.output
        if output_path is None:
            output_path = args.summary.parent / "tuning_heatmap.png"
        plot_heatmaps_3d(
            score_tensor,
            x_values,
            y_values,
            slice_values,
            plot_params[0],
            plot_params[1],
            plot_params[2],
            metric_key,
            output_path,
            args.show,
        )
        return

    print(
        "Not enough varying hyper-parameters to build a heatmap; use --plot-params to override."
    )


if __name__ == "__main__":
    main()
