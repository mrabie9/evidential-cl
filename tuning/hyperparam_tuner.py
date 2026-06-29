#!/usr/bin/env python3
"""Reusable hyperparameter tuning harness utilities."""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import random
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch
import sys
import yaml

sys.path.append("/home/lunet/wsmr11/repos/La-MAML")  # to import from parent directory
import parser as file_parser
from main import life_experience
from utils import misc_utils

Grid = Dict[str, List[Any]]
TypeHints = Dict[str, type]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _dedupe_config_sources(sources: Sequence[str]) -> List[str]:
    """Drop duplicate config paths while preserving order."""
    seen: set[str] = set()
    unique: List[str] = []
    for source in sources:
        resolved = str(Path(source).expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(source)
    return unique


def _default_config_chain(model_name: str, preset_default: str | None) -> List[str]:
    """Return the default stack of config fragments for a tuning run.

    Tuning intentionally omits ``configs/base.yaml`` (full-experiment task order
    and data paths). Use ``configs/tuning_defaults.yaml`` plus the model fragment.
    """

    chain: List[str] = []
    tuning_defaults = REPO_ROOT / "configs/tuning_defaults.yaml"
    if tuning_defaults.exists():
        chain.append(str(tuning_defaults))
    model_cfg_candidates = (
        REPO_ROOT / "configs/models" / f"{model_name}.yaml",
        REPO_ROOT / "configs/models/til" / f"{model_name}.yaml",
        REPO_ROOT / "configs/models/cil" / f"{model_name}.yaml",
    )
    model_cfg_found = False
    for model_cfg in model_cfg_candidates:
        if model_cfg.exists():
            chain.append(str(model_cfg))
            model_cfg_found = True
            break
    if not model_cfg_found and preset_default:
        preset_path = Path(preset_default)
        if not preset_path.is_absolute():
            preset_path = REPO_ROOT / preset_path
        chain.append(str(preset_path))
    return chain


@dataclass
class TuningPreset:
    """Configuration wrapper for building a tuning entrypoint."""

    model_name: str
    description: str | None = None
    default_config: str = "config_all.yaml"
    default_output_root: str | None = None
    default_grid: Grid | None = None
    type_hints: TypeHints = field(default_factory=dict)
    grid_factory: Callable[[argparse.Namespace], Grid] | None = None

    def resolve_description(self) -> str:
        if self.description:
            return self.description
        return (
            f"Run grid or random search over {self.model_name.upper()} hyperparameters."
        )

    def resolve_output_root(self) -> str:
        if self.default_output_root:
            return self.default_output_root
        return f"~/repos/La-MAML/logs/tuning/{self.model_name}"


def build_cli(preset: TuningPreset) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=preset.resolve_description(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="YAML config file(s) applied in order. Defaults to tuning_defaults.yaml"
        " plus the model-specific fragment when --config is omitted.",
    )
    parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of YAML config fragments to apply (alphabetical order).",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="PARAM=V1,V2,...",
        help="Grid specification. May be provided multiple times.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PARAM=VALUE",
        help="Override applied to all trials (e.g. --override n_epochs=10).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Randomly sample this many combinations from the grid.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=0,
        help="Seed for shuffling or sampling trial order.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Evaluate at most this many trials (after sampling/shuffling).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to the base seed for each trial index.",
    )
    parser.add_argument(
        "--vary-seed",
        action="store_true",
        help="If set, add the trial index to the base seed.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=preset.resolve_output_root(),
        help="Directory where aggregated tuning summaries are stored.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned trials without running them.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the full grid even when every combination is evaluated.",
    )
    parser.add_argument(
        "--keep-expt-name",
        action="store_true",
        help="Do not alter the experiment name from the base config.",
    )
    parser.add_argument(
        "--lr-first",
        action="store_true",
        help="Tune the learning rate first, then tune remaining parameters.",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Tune each hyperparameter sequentially, carrying forward the best value.",
    )
    parser.add_argument(
        "--lr-key",
        type=str,
        default="lr",
        help="Comma-separated parameter names treated as learning rates for --lr-first.",
    )
    parser.add_argument(
        "--tune-only",
        action="append",
        default=[],
        metavar="PARAM",
        help="Restrict the grid to specific hyperparameter(s). May be repeated or comma-separated.",
    )
    return parser


def get_reference(
    key: str, base_args: argparse.Namespace, type_hints: TypeHints
) -> Any:
    if hasattr(base_args, key):
        return getattr(base_args, key)
    return type_hints.get(key)


def coerce_value(raw: str, reference: Any) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None

    if isinstance(reference, bool):
        return lowered in {"1", "true", "yes", "on"}
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, str):
        return value
    if isinstance(reference, type):
        if reference is bool:
            return lowered in {"1", "true", "yes", "on"}
        if reference is int:
            return int(value)
        if reference is float:
            return float(value)
        return reference(value)

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_grid_specs(
    specs: Sequence[str],
    base_args: argparse.Namespace,
    type_hints: TypeHints,
) -> Grid:
    space: Grid = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid grid spec '{spec}'. Use PARAM=v1,v2,... format.")
        key, raw_values = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid grid spec '{spec}'.")
        values: List[Any] = []
        for item in raw_values.split(","):
            item = item.strip()
            if not item:
                continue
            reference = get_reference(key, base_args, type_hints)
            values.append(coerce_value(item, reference))
        if not values:
            raise ValueError(f"No values provided for grid parameter '{key}'.")
        space[key] = values
    return space


def parse_override_specs(
    specs: Sequence[str],
    base_args: argparse.Namespace,
    type_hints: TypeHints,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid override '{spec}'. Use PARAM=value format.")
        key, raw_value = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{spec}'.")
        reference = get_reference(key, base_args, type_hints)
        overrides[key] = coerce_value(raw_value, reference)
    return overrides


def expand_trials(
    space: Grid,
    num_samples: int | None,
    search_seed: int,
    max_trials: int | None,
    shuffle: bool,
) -> List[Dict[str, Any]]:
    if not space:
        return [{}]

    keys = sorted(space)
    grid_iter = itertools.product(*(space[key] for key in keys))
    all_combos = [dict(zip(keys, values)) for values in grid_iter]

    rng = random.Random(search_seed)

    if num_samples is not None and num_samples < len(all_combos):
        indices = list(range(len(all_combos)))
        rng.shuffle(indices)
        selected = indices[:num_samples]
        combos = [all_combos[i] for i in selected]
    else:
        combos = all_combos
        if shuffle:
            rng.shuffle(combos)

    if max_trials is not None:
        combos = combos[:max_trials]
    return combos or [{}]


def parse_lr_keys(raw: str) -> List[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    return keys or ["lr"]


def parse_tune_only(specs: Sequence[str]) -> List[str]:
    keys: List[str] = []
    seen: set[str] = set()
    for spec in specs:
        for item in spec.split(","):
            key = item.strip()
            if not key or key in seen:
                continue
            keys.append(key)
            seen.add(key)
    return keys


def format_value_for_slug(value: Any) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        abs_val = abs(value)
        if abs_val >= 1:
            txt = f"{value:.3f}".rstrip("0").rstrip(".")
        else:
            txt = f"{value:.0e}".replace("+", "")
        return txt.replace("-", "m").replace(".", "p")
    return str(value).replace(" ", "")


def slugify_params(params: Dict[str, Any], max_length: int = 80) -> str:
    if not params:
        return "base"
    parts = []
    for key in sorted(params):
        encoded = format_value_for_slug(params[key])
        encoded = encoded.replace("/", "_")
        if len(encoded) > 12:
            encoded = encoded[:12]
        parts.append(f"{key}-{encoded}")
    slug = "_".join(parts)
    if len(slug) > max_length:
        slug = slug[:max_length]
    return slug


def extract_final_scores(tensorlike: Any) -> List[float]:
    if isinstance(tensorlike, torch.Tensor):
        if tensorlike.numel() == 0:
            return []
        last = tensorlike[-1]
        if last.ndim == 0:
            return [float(last.item())]
        return [float(x) for x in last.tolist()]
    if isinstance(tensorlike, list) and tensorlike:
        last = tensorlike[-1]
        if isinstance(last, (list, tuple)):
            return [float(x) for x in last]
        return [float(last)]
    return []


def compute_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _trial_rank_key(trial: Dict[str, Any]) -> tuple[float, int, int]:
    """Sort trials by score, then param completeness, then trial index."""
    score = trial.get("score")
    score_value = float(score) if isinstance(score, (int, float)) else float("-inf")
    params = trial.get("params") or {}
    return (score_value, len(params), int(trial.get("trial", -1)))


def select_best_trial(
    successes: List[Dict[str, Any]],
    search_space: Grid,
    *,
    hierarchical: bool,
) -> Dict[str, Any] | None:
    """Pick the best trial for reporting and YAML writeback.

    For hierarchical sweeps, prefer the final tuning stage so the reported best
    params include every searched hyperparameter. When scores tie, prefer the
    trial with the most complete ``params`` mapping and the highest trial index.

    Args:
        successes: Successful trial result dicts.
        search_space: Explored hyperparameter grid.
        hierarchical: Whether the run used ``--hierarchical``.

    Returns:
        The selected best trial dict, or ``None`` when ``successes`` is empty.

    Usage:
        best = select_best_trial(results, search_space, hierarchical=True)
    """
    if not successes:
        return None
    if not hierarchical:
        return max(successes, key=_trial_rank_key)

    stage_keys = list(search_space.keys())
    final_stage = stage_keys[-1] if stage_keys else None
    final_stage_trials = (
        [trial for trial in successes if trial.get("stage") == final_stage]
        if final_stage
        else []
    )
    candidate_pool = final_stage_trials or successes
    return max(candidate_pool, key=_trial_rank_key)


def extract_total_f1_mean_from_trial_logs(
    log_dir: str | Path, fallback_num_tasks: int
) -> float:
    """Extract final mean total F1 from the latest trial metrics file.

    This reads the latest ``task*.npz`` produced by a training run and resolves
    ``val_f1`` to a single run-level score by taking the last ``n_tasks``
    entries when needed.

    Args:
        log_dir: Trial output directory containing ``task*.npz`` files.
        fallback_num_tasks: Fallback task count when detection vectors are
            unavailable in metrics.

    Returns:
        Final mean total F1 score, or NaN when it cannot be recovered.

    Usage:
        f1_score = extract_total_f1_mean_from_trial_logs("/tmp/run", 3)
    """
    candidate_dir = Path(log_dir)
    metrics_dir = (
        candidate_dir / "metrics"
        if (candidate_dir / "metrics").is_dir()
        else candidate_dir
    )
    task_files = sorted(metrics_dir.glob("task*.npz"))
    if not task_files:
        return float("nan")

    latest_metrics = np.load(task_files[-1], allow_pickle=False)
    if "val_f1" not in latest_metrics:
        return float("nan")

    val_f1_array = np.asarray(latest_metrics["val_f1"], dtype=float).reshape(-1)
    if val_f1_array.size == 0:
        return float("nan")

    inferred_num_tasks = 0
    if "val_det_acc" in latest_metrics:
        inferred_num_tasks = max(
            inferred_num_tasks, int(np.asarray(latest_metrics["val_det_acc"]).size)
        )
    if "val_det_fa" in latest_metrics:
        inferred_num_tasks = max(
            inferred_num_tasks, int(np.asarray(latest_metrics["val_det_fa"]).size)
        )
    if inferred_num_tasks <= 0:
        inferred_num_tasks = max(int(fallback_num_tasks), 1)

    if val_f1_array.size >= inferred_num_tasks:
        final_slice = val_f1_array[-inferred_num_tasks:]
        return float(np.mean(final_slice))

    return float(np.mean(val_f1_array))


def run_single_trial(
    base_args: argparse.Namespace,
    constant_overrides: Dict[str, Any],
    trial_overrides: Dict[str, Any],
    trial_idx: int,
    session_timestamp: str,
    runs_root: Path,
    seed_offset: int,
    vary_seed: bool,
    keep_expt_name: bool,
    model_name: str,
) -> Dict[str, Any]:
    args = deepcopy(base_args)
    merged = dict(constant_overrides)
    merged.update(trial_overrides)
    for key, value in merged.items():
        setattr(args, key, value)

    args.model = model_name

    args.log_dir = str(runs_root)
    seed_base = int(getattr(base_args, "seed", 0) + seed_offset)
    args.seed = seed_base + (trial_idx if vary_seed else 0)

    trial_slug = slugify_params(trial_overrides)
    if not keep_expt_name:
        base_name = getattr(base_args, "expt_name", model_name)
        args.expt_name = f"{base_name}_tune_{trial_idx:03d}_{trial_slug}"[:120]

    trial_timestamp = f"{session_timestamp}-trial{trial_idx:03d}"

    misc_utils.init_seed(args.seed)
    log_dir, tf_dir = misc_utils.log_dir(args, trial_timestamp, model_name)
    args.log_dir = log_dir
    args.tf_dir = tf_dir
    if hasattr(args, "data_path"):
        data_path = Path(args.data_path).expanduser()
        if not data_path.is_absolute() and not data_path.exists():
            candidate = REPO_ROOT / data_path
            if candidate.exists():
                args.data_path = str(candidate)

    loader_mod = importlib.import_module(f"dataloaders.{args.loader}")
    loader = loader_mod.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)

    model_mod = importlib.import_module(f"model.{args.model}")
    model = model_mod.Net(n_inputs, n_outputs, n_tasks, args)

    if getattr(args, "cuda", False) and torch.cuda.is_available():
        model = model.cuda()

    try:
        if args.model == "iid2":
            # IID2 is a non-lifelong (single-round) experiment. We run the
            # single-round training pipeline and map its metrics into the
            # same result-tuple shape the tuner expects.
            from main_single_round import (
                build_single_round_loaders,
                run_single_round_training,
            )

            train_loader, test_loader, _selected_indices = build_single_round_loaders(
                args, loader
            )
            (
                result_val_t,
                result_val_a,
                spent,
                metrics_payload,
            ) = run_single_round_training(model, train_loader, test_loader, args)

            # main_single_round does not compute separate test metrics.
            result_test_t = torch.empty((0,), dtype=torch.long)
            result_test_a = torch.empty((0, 0), dtype=torch.float)
            result_test_det_a = torch.empty((0,), dtype=torch.float)
            result_test_det_fa = torch.empty((0,), dtype=torch.float)

            result_val_det_a = torch.as_tensor(
                metrics_payload.get("val_det_acc", []), dtype=torch.float
            )
            result_val_det_fa = torch.as_tensor(
                metrics_payload.get("val_det_fa", []), dtype=torch.float
            )
        else:
            (
                result_val_t,
                result_val_a,
                result_test_t,
                result_test_a,
                result_val_det_a,
                result_val_det_fa,
                result_test_det_a,
                result_test_det_fa,
                _result_val_f1,
                _result_test_f1,
                spent,
            ) = life_experience(model, loader, args)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_scores = extract_final_scores(result_val_a)
    test_scores = extract_final_scores(result_test_a)
    val_det_scores = extract_final_scores(result_val_det_a)
    val_pfa_scores = extract_final_scores(result_val_det_fa)
    test_det_scores = extract_final_scores(result_test_det_a)
    test_pfa_scores = extract_final_scores(result_test_det_fa)

    val_mean = compute_mean(val_scores)
    val_f1_mean = extract_total_f1_mean_from_trial_logs(log_dir, len(val_scores))
    det_mean = compute_mean(val_det_scores)
    pfa_mean = compute_mean(val_pfa_scores)
    if np.isnan(val_f1_mean):
        print(
            "[WARN] Trial {} has no usable val_f1 in {}. Falling back to val_mean ({:.4f}) for tuning score.".format(
                trial_idx, log_dir, val_mean
            )
        )
        score = val_mean
    else:
        score = val_f1_mean

    return {
        "status": "ok",
        "trial": trial_idx,
        "log_dir": log_dir,
        "tf_dir": tf_dir,
        "params": merged,
        "trial_params": dict(trial_overrides),
        "fixed_params": dict(constant_overrides),
        "val_per_task": val_scores,
        "val_mean": val_mean,
        "val_f1_mean": val_f1_mean,
        "val_det_per_task": val_det_scores,
        "val_det_mean": det_mean,
        "val_pfa_per_task": val_pfa_scores,
        "val_pfa_mean": pfa_mean,
        "test_per_task": test_scores,
        "test_mean": compute_mean(test_scores),
        "test_det_per_task": test_det_scores,
        "test_det_mean": compute_mean(test_det_scores),
        "test_pfa_per_task": test_pfa_scores,
        "test_pfa_mean": compute_mean(test_pfa_scores),
        "score": score,
        "duration_sec": float(spent),
    }


def dump_summary(
    session_dir: Path, summary: Dict[str, Any], successes: List[Dict[str, Any]]
) -> None:
    summary_path = session_dir / "summary.json"
    session_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if not successes:
        return

    field_names = [
        "trial",
        "score",
        "val_mean",
        "val_det_mean",
        "val_pfa_mean",
        "test_mean",
        "test_det_mean",
        "test_pfa_mean",
        "duration_sec",
        "log_dir",
    ]
    if any("stage" in trial for trial in successes):
        field_names.insert(1, "stage")
    param_keys = sorted({key for trial in successes for key in trial["params"].keys()})
    field_names.extend(param_keys)
    csv_path = session_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for trial in successes:
            row = {
                "trial": trial["trial"],
                "score": trial.get("score"),
                "val_mean": trial.get("val_mean"),
                "val_det_mean": trial.get("val_det_mean"),
                "val_pfa_mean": trial.get("val_pfa_mean"),
                "test_mean": trial.get("test_mean"),
                "test_det_mean": trial.get("test_det_mean"),
                "test_pfa_mean": trial.get("test_pfa_mean"),
                "duration_sec": trial["duration_sec"],
                "log_dir": trial["log_dir"],
            }
            if "stage" in field_names:
                row["stage"] = trial.get("stage")
            for key in param_keys:
                row[key] = trial["params"].get(key)
            writer.writerow(row)


def write_best_params_to_yaml(
    yaml_path: Path, best_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Write best hyperparameter values into an existing YAML config.

    Args:
        yaml_path: Path to the YAML file that should be updated in-place.
        best_params: Hyperparameter key/value pairs to persist.

    Returns:
        The dictionary of values that were written to the YAML file.

    Notes:
        This function uses ``yaml.safe_dump`` and rewrites the full file. Existing
        YAML comments and some formatting details are not preserved.

    Usage:
        applied = write_best_params_to_yaml(Path("configs/models/til/hat.yaml"), {"lr": 0.003})
    """
    payload: Dict[str, Any] = {}
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Cannot update YAML file '{yaml_path}': top-level document is not a mapping."
            )
        payload = dict(loaded)
    else:
        raise FileNotFoundError(f"Cannot update missing YAML file: {yaml_path}")

    for key, value in best_params.items():
        payload[key] = value

    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return best_params


def resolve_cli_config_path(config_path: str) -> Path:
    """Resolve a CLI-provided config path using caller working-directory semantics."""
    candidate = Path(config_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def run_tuning(preset: TuningPreset) -> None:
    cli = build_cli(preset).parse_args()

    config_sources: List[str] = []
    # Apply defaults first so explicit model configs override them.
    config_sources.extend(cli.config_dir)
    config_sources.extend(cli.config)
    if not config_sources:
        config_sources = _default_config_chain(preset.model_name, preset.default_config)
    config_sources = _dedupe_config_sources(config_sources)

    base_args = file_parser.parse_args_from_yaml(config_sources)
    if getattr(base_args, "model", preset.model_name) != preset.model_name:
        base_args.model = preset.model_name

    constant_overrides = parse_override_specs(
        cli.override, base_args, preset.type_hints
    )

    if cli.grid:
        search_space = parse_grid_specs(cli.grid, base_args, preset.type_hints)
    elif preset.grid_factory is not None:
        search_space = preset.grid_factory(base_args)
    elif preset.default_grid is not None:
        search_space = {key: values[:] for key, values in preset.default_grid.items()}
    else:
        search_space = {}

    for key in list(search_space.keys()):
        if key in constant_overrides:
            del search_space[key]

    tune_only = parse_tune_only(cli.tune_only)
    if tune_only:
        missing = [key for key in tune_only if key not in search_space]
        if missing:
            raise ValueError(
                "Tune-only parameter(s) not found in the search space: "
                + ", ".join(missing)
            )
        search_space = {key: search_space[key] for key in tune_only}

    if cli.hierarchical and cli.lr_first:
        raise ValueError("Choose either --hierarchical or --lr-first, not both.")

    lr_keys = parse_lr_keys(cli.lr_key)
    lr_first = bool(cli.lr_first)
    lr_space = {key: search_space[key] for key in lr_keys if key in search_space}

    if lr_first and not lr_space:
        print(
            "LR-first tuning requested but no learning-rate keys exist in the search space."
            " Proceeding with the full grid."
        )
        lr_first = False

    full_search_space = {key: values[:] for key, values in search_space.items()}
    trials = expand_trials(
        search_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle
    )

    if cli.dry_run:
        print("Planned trials (dry-run):")
        if cli.hierarchical:
            total_trials = 0
            trial_idx = 0
            stage_overrides: Dict[str, Any] = dict(constant_overrides)
            for key in search_space.keys():
                stage_space = {key: search_space[key]}
                stage_trials = expand_trials(
                    stage_space,
                    cli.num_samples,
                    cli.search_seed,
                    cli.max_trials,
                    cli.shuffle,
                )
                for trial in stage_trials:
                    merged = dict(stage_overrides)
                    merged.update(trial)
                    print(f"  [{key}] #{trial_idx:03d}: {merged}")
                    trial_idx += 1
                total_trials += len(stage_trials)
            print(f"Total: {total_trials} trials")
        elif lr_first:
            lr_trials = expand_trials(
                lr_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle
            )
            rest_space = {k: v for k, v in search_space.items() if k not in lr_space}
            rest_trials = expand_trials(
                rest_space,
                cli.num_samples,
                cli.search_seed,
                cli.max_trials,
                cli.shuffle,
            )
            total_trials = len(lr_trials) + len(rest_trials)
            for idx, tr in enumerate(lr_trials):
                merged = dict(constant_overrides)
                merged.update(tr)
                print(f"  [lr] #{idx:03d}: {merged}")
            offset = len(lr_trials)
            for idx, tr in enumerate(rest_trials):
                merged = dict(constant_overrides)
                merged.update(tr)
                print(f"  [rest] #{idx + offset:03d}: {merged}")
            print(f"Total: {total_trials} trials")
        else:
            for idx, tr in enumerate(trials):
                merged = dict(constant_overrides)
                merged.update(tr)
                print(f"  #{idx:03d}: {merged}")
            print(f"Total: {len(trials)} trials")
        return

    session_timestamp = misc_utils.get_date_time()
    session_dir = Path(cli.output_root) / session_timestamp
    runs_root = session_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    def run_trials(
        trial_list: List[Dict[str, Any]],
        overrides: Dict[str, Any],
        stage: str | None,
        start_idx: int,
    ) -> List[Dict[str, Any]]:
        stage_results: List[Dict[str, Any]] = []
        for offset, trial_params in enumerate(trial_list):
            trial_idx = start_idx + offset
            try:
                outcome = run_single_trial(
                    base_args,
                    overrides,
                    trial_params,
                    trial_idx,
                    session_timestamp,
                    runs_root,
                    cli.seed_offset,
                    cli.vary_seed,
                    cli.keep_expt_name,
                    preset.model_name,
                )
            except Exception as exc:  # pylint: disable=broad-except
                trace = traceback.format_exc()
                outcome = {
                    "status": "failed",
                    "trial": trial_idx,
                    "params": dict(overrides, **trial_params),
                    "error": str(exc),
                    "traceback": trace,
                }
                if stage:
                    outcome["stage"] = stage
                print(f"Trial {trial_idx} failed: {exc}")
            else:
                if stage:
                    outcome["stage"] = stage
                    print(
                        f"[{stage}] Trial {trial_idx} finished | score={outcome['score']:.4f} |"
                        f" params={trial_params}"
                    )
                else:
                    print(
                        f"Trial {trial_idx} finished | score={outcome['score']:.4f} |"
                        f" params={trial_params}"
                    )
            stage_results.append(outcome)
        return stage_results

    lr_first_best: Dict[str, Any] | None = None
    hierarchical_final_params: Dict[str, Any] | None = None
    if cli.hierarchical:
        trial_idx = 0
        stage_overrides: Dict[str, Any] = dict(constant_overrides)
        for key in search_space.keys():
            stage_space = {key: search_space[key]}
            stage_trials = expand_trials(
                stage_space,
                cli.num_samples,
                cli.search_seed,
                cli.max_trials,
                cli.shuffle,
            )
            stage_results = run_trials(stage_trials, stage_overrides, key, trial_idx)
            results.extend(stage_results)
            trial_idx += len(stage_trials)

            stage_successes = [
                r
                for r in stage_results
                if r.get("status") == "ok" and r.get("stage") == key
            ]
            stage_best = (
                max(stage_successes, key=lambda r: r["score"])
                if stage_successes
                else None
            )
            if stage_best is None:
                print(
                    f"Hierarchical stage '{key}' recorded no successful trials; stopping."
                )
                break
            stage_overrides[key] = stage_best["trial_params"].get(key)
        hierarchical_final_params = {
            key: stage_overrides[key] for key in search_space if key in stage_overrides
        }
    elif lr_first:
        lr_trials = expand_trials(
            lr_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle
        )
        results.extend(run_trials(lr_trials, constant_overrides, "lr", 0))
        lr_successes = [
            r for r in results if r.get("status") == "ok" and r.get("stage") == "lr"
        ]
        lr_best = max(lr_successes, key=lambda r: r["score"]) if lr_successes else None
        if lr_best:
            lr_first_best = {key: lr_best["trial_params"].get(key) for key in lr_space}
            stage2_overrides = dict(constant_overrides, **lr_first_best)
            rest_space = {k: v for k, v in search_space.items() if k not in lr_space}
            rest_trials = expand_trials(
                rest_space,
                cli.num_samples,
                cli.search_seed,
                cli.max_trials,
                cli.shuffle,
            )
            results.extend(
                run_trials(rest_trials, stage2_overrides, "rest", len(lr_trials))
            )
        else:
            print(
                "LR-first stage recorded no successful trials; skipping remaining parameters."
            )
    else:
        results.extend(run_trials(trials, constant_overrides, None, 0))

    successes = [r for r in results if r.get("status") == "ok"]
    best = select_best_trial(
        successes,
        full_search_space,
        hierarchical=bool(cli.hierarchical),
    )

    resolved_chain = [str(Path(path).resolve()) for path in config_sources]
    summary = {
        "config": resolved_chain[-1] if resolved_chain else None,
        "config_chain": resolved_chain,
        "base_expt_name": getattr(base_args, "expt_name", preset.model_name),
        "session_dir": str(session_dir.resolve()),
        "timestamp": session_timestamp,
        "fixed_overrides": constant_overrides,
        "search_space": full_search_space,
        "hierarchical": bool(cli.hierarchical),
        "hierarchical_final_params": hierarchical_final_params,
        "lr_first": lr_first,
        "lr_first_keys": lr_keys if lr_first else None,
        "lr_first_best": lr_first_best,
        "num_trials": len(results),
        "results": results,
        "best": best,
    }

    dump_summary(session_dir, summary, successes)

    if best:
        updated_yaml_path: str | None = None
        updated_yaml_values: Dict[str, Any] | None = None
        yaml_update_error: str | None = None
        if cli.config:
            target_yaml = resolve_cli_config_path(cli.config[-1])
            if hierarchical_final_params:
                values_to_write = dict(hierarchical_final_params)
            else:
                values_to_write = dict(
                    best.get("params") or best.get("trial_params") or {}
                )
            if values_to_write:
                try:
                    updated_yaml_values = write_best_params_to_yaml(
                        target_yaml, values_to_write
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    yaml_update_error = str(exc)
                    summary["updated_yaml_error"] = yaml_update_error
                    dump_summary(session_dir, summary, successes)
                else:
                    updated_yaml_path = str(target_yaml)
                    summary["updated_yaml_path"] = updated_yaml_path
                    summary["updated_yaml_values"] = updated_yaml_values
                    dump_summary(session_dir, summary, successes)

        print(
            f"Best trial #{best['trial']} | score={best['score']:.4f} | params={best['trial_params']}"
        )
        print(f"Logs stored in: {best['log_dir']}")
        if updated_yaml_path:
            print(f"Updated YAML config with best params: {updated_yaml_path}")
        if yaml_update_error:
            print(
                f"[WARN] Could not write tuned params back to YAML: {yaml_update_error}"
            )
    else:
        print("No successful trials were recorded.")


def make_main(preset: TuningPreset) -> Callable[[], None]:
    def _main() -> None:
        run_tuning(preset)

    return _main


__all__ = [
    "Grid",
    "TypeHints",
    "TuningPreset",
    "build_cli",
    "run_tuning",
    "make_main",
    "select_best_trial",
]
