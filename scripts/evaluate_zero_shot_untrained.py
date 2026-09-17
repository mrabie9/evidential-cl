#!/usr/bin/env python3
"""Evaluate zero-shot validation metrics for an untrained model.

This script initializes a model and dataloader exactly like ``main.py`` but does
not train. It evaluates zero-shot performance per task independently and reports
collector-compatible columns:

- ``f1_cls``: harmonic F1 from ``rec_cls`` and ``prec_cls``
- ``total_f1_zs``: raw evaluator F1 at the current task index

Optionally, it can also export a cumulative per-task matrix at each checkpoint,
mirroring the pre-train evaluation structure used in continual runs.

Usage:
    python scripts/evaluate_zero_shot_untrained.py --config configs/base.yaml --config configs/models/cmaml.yaml

    python scripts/evaluate_zero_shot_untrained.py --config configs/base.yaml --config configs/models/cmaml.yaml --csv plots/zs_untrained.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import parser as file_parser  # noqa: E402
from main import _split_eval_output, eval_class_tasks, eval_tasks  # noqa: E402
from utils import misc_utils  # noqa: E402


def _default_main_config_chain() -> List[str]:
    """Return the default YAML config chain used by ``main.py``.

    Returns:
        Ordered list of YAML paths to apply as defaults.

    Usage:
        >>> isinstance(_default_main_config_chain(), list)
        True
    """
    chain: List[str] = []
    base_cfg = Path("configs/base.yaml")
    if base_cfg.exists():
        chain.append(str(base_cfg))
    legacy = Path("config_all.yaml")
    if legacy.exists():
        chain.append(str(legacy))
    return chain


def _safe_float(value: Any) -> float:
    """Convert a scalar-like value to ``float``.

    Args:
        value: Input scalar/tensor-like value.

    Returns:
        Float value or NaN when conversion is not possible.

    Usage:
        >>> _safe_float(np.float64(0.5))
        0.5
    """
    if value is None:
        return float("nan")
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return float("nan")
        return float(arr[0])
    except (TypeError, ValueError):
        return float("nan")


def _harmonic_mean_f1(precision: float, recall: float) -> float:
    """Compute harmonic-mean F1 from precision and recall.

    Args:
        precision: Precision value.
        recall: Recall value.

    Returns:
        Harmonic F1, or NaN when undefined due to missing inputs.

    Usage:
        >>> _harmonic_mean_f1(0.5, 0.5)
        0.5
    """
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    denominator = precision + recall
    if denominator <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / denominator)


def _nan_to_none(value: Any) -> Any:
    """Convert NaN values recursively for JSON output.

    Args:
        value: Potentially nested data structure.

    Returns:
        JSON-safe structure where NaN floats become ``None``.

    Usage:
        >>> _nan_to_none(float("nan")) is None
        True
    """
    if isinstance(value, dict):
        return {key: _nan_to_none(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    """Write tabular rows to CSV.

    Args:
        rows: Rows to write.
        path: Output CSV path.

    Usage:
        >>> isinstance(_write_csv, object)
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_runtime_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    """Resolve config chain and parser args for zero-shot evaluation.

    Returns:
        Tuple ``(script_args, run_args)`` where ``script_args`` are script-only
        options and ``run_args`` are model/data args compatible with ``main.py``.

    Usage:
        >>> isinstance(_resolve_runtime_args, object)
        True
    """
    front_parser = argparse.ArgumentParser(add_help=False)
    front_parser.add_argument("--config", action="append", default=[])
    front_parser.add_argument("--config-dir", action="append", default=[])
    front_parser.add_argument("--no-config", action="store_true")
    front_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Runtime device override for this script.",
    )
    front_parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output for independent per-task rows.",
    )
    front_parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output for independent per-task rows.",
    )
    front_parser.add_argument(
        "--include-cumulative-matrix",
        action="store_true",
        help="Also export cumulative per-task zero-shot matrix rows.",
    )
    front_parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=None,
        help="Optional CSV output for --include-cumulative-matrix rows.",
    )
    script_args, remaining = front_parser.parse_known_args()

    config_chain: List[str] = []
    if not script_args.no_config:
        config_chain.extend(script_args.config_dir)
        config_chain.extend(script_args.config)
        if not config_chain:
            config_chain = _default_main_config_chain()

    base_args = file_parser.parse_args_from_yaml(config_chain or None)
    run_parser = file_parser.get_parser()
    run_args = run_parser.parse_args(remaining, namespace=base_args)
    run_args.lr = misc_utils.scale_learning_rate_for_batch_size(
        run_args.lr, run_args.batch_size
    )
    return script_args, run_args


def _apply_device_override(
    run_args: argparse.Namespace, device_choice: str
) -> torch.device:
    """Apply device override to runtime args.

    Args:
        run_args: Parsed runtime args.
        device_choice: One of ``auto``, ``cpu``, ``cuda``.

    Returns:
        Torch device selected for evaluation.

    Usage:
        >>> isinstance(_apply_device_override, object)
        True
    """
    if device_choice == "cpu":
        run_args.cuda = False
        return torch.device("cpu")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        run_args.cuda = True
        return torch.device("cuda")
    if getattr(run_args, "cuda", False) and torch.cuda.is_available():
        run_args.cuda = True
        return torch.device("cuda")
    run_args.cuda = False
    return torch.device("cpu")


def _build_model_and_loader(
    run_args: argparse.Namespace,
) -> tuple[torch.nn.Module, Any]:
    """Instantiate loader and untrained model.

    Args:
        run_args: Runtime args compatible with ``main.py``.

    Returns:
        Tuple of ``(model, loader)``.

    Usage:
        >>> isinstance(_build_model_and_loader, object)
        True
    """
    misc_utils.init_seed(run_args.seed)
    loader_module = importlib.import_module(f"dataloaders.{run_args.loader}")
    loader = loader_module.IncrementalLoader(run_args, seed=run_args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    model_module = importlib.import_module(f"model.{run_args.model}")
    model = model_module.Net(n_inputs, n_outputs, n_tasks, run_args)
    if run_args.cuda:
        try:
            model.cuda()
        except RuntimeError:
            run_args.cuda = False
            model.cpu()
    return model, loader


def _extract_metric_at_index(metric: object, index: int) -> float:
    """Extract metric value at index from evaluator output.

    Args:
        metric: Evaluator metric sequence or scalar.
        index: Task index to read.

    Returns:
        Float value or NaN.

    Usage:
        >>> _extract_metric_at_index([0.1, 0.2], 1)
        0.2
    """
    if metric is None:
        return float("nan")
    if isinstance(metric, (list, tuple)):
        if index < 0 or index >= len(metric):
            return float("nan")
        return _safe_float(metric[index])
    return _safe_float(metric)


def _collect_untrained_zero_shot(
    model: torch.nn.Module,
    loader: Any,
    run_args: argparse.Namespace,
    include_cumulative_matrix: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect per-task zero-shot metrics without any training.

    Args:
        model: Untrained model.
        loader: Incremental loader instance.
        run_args: Runtime args.
        include_cumulative_matrix: Whether to produce cumulative matrix rows.

    Returns:
        Tuple ``(independent_rows, cumulative_rows)``.

    Usage:
        >>> isinstance(_collect_untrained_zero_shot, object)
        True
    """
    evaluator = (
        eval_class_tasks
        if run_args.loader == "class_incremental_loader"
        else eval_tasks
    )
    independent_rows: List[Dict[str, Any]] = []
    cumulative_rows: List[Dict[str, Any]] = []
    test_task_loaders: List[Any] = []
    task_infos: List[Dict[str, Any]] = []

    for _ in range(loader.n_tasks):
        task_info, _train_loader, _val_loader, test_loader = loader.new_task()
        task_infos.append(task_info)
        test_task_loaders.append(test_loader)

    # Evaluate all tasks in one pass so evaluator task indices match the
    # original continual task ids (important for split-head models).
    independent_output = evaluator(model, test_task_loaders, run_args)
    ind_rec, ind_prec, ind_f1 = _split_eval_output(independent_output)
    for task_index, task_info in enumerate(task_infos):
        rec_cls = _extract_metric_at_index(ind_rec, task_index)
        prec_cls = _extract_metric_at_index(ind_prec, task_index)
        total_macro_f1_zs = _extract_metric_at_index(ind_f1, task_index)
        macro_f1 = _harmonic_mean_f1(prec_cls, rec_cls)

        independent_rows.append(
            {
                "algo": run_args.model,
                "task": task_index,
                "task_name": task_info.get("task_name", ""),
                "macro_f1": macro_f1,
                "macro_rec": rec_cls,
                "macro_prec": prec_cls,
                "total_macro_f1_zs": total_macro_f1_zs,
            }
        )

    if include_cumulative_matrix:
        for checkpoint_task_index in range(len(test_task_loaders)):
            cumulative_output = evaluator(
                model, test_task_loaders[: checkpoint_task_index + 1], run_args
            )
            cum_rec, cum_prec, cum_f1 = _split_eval_output(cumulative_output)
            for eval_task_idx in range(checkpoint_task_index + 1):
                cumulative_rows.append(
                    {
                        "algo": run_args.model,
                        "checkpoint_task_index": checkpoint_task_index,
                        "evaluated_task_index": eval_task_idx,
                        "macro_rec": _extract_metric_at_index(cum_rec, eval_task_idx),
                        "macro_prec": _extract_metric_at_index(cum_prec, eval_task_idx),
                        "total_macro_f1_zs": _extract_metric_at_index(
                            cum_f1, eval_task_idx
                        ),
                    }
                )

    return independent_rows, cumulative_rows


def _print_rows(rows: Sequence[Dict[str, Any]]) -> None:
    """Print independent per-task rows as a readable table.

    Args:
        rows: Independent per-task rows.

    Usage:
        >>> isinstance(_print_rows, object)
        True
    """
    header = (
        f"{'algo':<12} {'task':>4} {'macro_f1':>10} {'macro_rec':>10} "
        f"{'macro_prec':>10} {'total_macro_f1_zs':>18}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['algo']:<12} {int(row['task']):4d} "
            f"{float(row['macro_f1']):10.6f} {float(row['macro_rec']):10.6f} "
            f"{float(row['macro_prec']):10.6f} "
            f"{float(row['total_macro_f1_zs']):18.6f}"
        )


def main() -> int:
    """Run zero-shot evaluation for an untrained model over all tasks."""
    script_args, run_args = _resolve_runtime_args()
    device = _apply_device_override(run_args, script_args.device)
    model, loader = _build_model_and_loader(run_args)
    model.to(device)
    print(f"Evaluating untrained model '{run_args.model}' on device {device}.")

    independent_rows, cumulative_rows = _collect_untrained_zero_shot(
        model=model,
        loader=loader,
        run_args=run_args,
        include_cumulative_matrix=script_args.include_cumulative_matrix,
    )
    _print_rows(independent_rows)

    if script_args.csv is not None:
        _write_csv(independent_rows, script_args.csv)
        print(f"Saved independent rows CSV to {script_args.csv}")
    if script_args.json is not None:
        script_args.json.parent.mkdir(parents=True, exist_ok=True)
        with script_args.json.open("w", encoding="utf-8") as output_json:
            json.dump(_nan_to_none(list(independent_rows)), output_json, indent=2)
            output_json.write("\n")
        print(f"Saved independent rows JSON to {script_args.json}")
    if script_args.include_cumulative_matrix and script_args.matrix_csv is not None:
        _write_csv(cumulative_rows, script_args.matrix_csv)
        print(f"Saved cumulative matrix CSV to {script_args.matrix_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
