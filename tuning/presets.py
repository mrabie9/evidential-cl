"""Preset definitions for hyperparameter tuning entrypoints."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Iterable, List

from tuning.hyperparam_tuner import Grid, TuningPreset

TypeSpec = Dict[str, Any]
GridSpec = Dict[str, TypeSpec]


COMMON_TYPE_HINTS: Dict[str, type] = {
    "batch_size": int,
    "lr": float,
    "memory_strength": float,
    "memories": int,
    "n_memories": int,
    "replay_batch_size": float,
    "beta": float,
    "gamma": float,
    "s": float,
    "batches_per_example": float,
    "meta_lr": float,
    "update_lr": float,
    "opt_lr": float,
    "opt_wt": float,
    "alpha_init": float,
    "mean_eta": float,
    "std_init": float,
    "train_mc_iters": int,
    "bgd_optimizer": str,
    "temperature": float,
    "inner_steps": int,
    "n_meta": int,
    "task_emb": int,
    "lr_rho": float,
    "alpha": float,
    "ratio": float,
    "clipgrad": float,
    "momentum": float,
    "weight_decay": float,
    "lamb": float,
    "eps": float,
    "si_c": float,
    "si_epsilon": float,
    "woe_lambda": float,
    "woe_xi": float,
    "woe_centering_mode": str,
    "woe_mu_momentum": float,
    "woe_importance_stride": int,
    "woe_conflict_weighting": bool,
    "clipgrad_norm": float,
    "optimizer": str,
    "smax": float,
    "reg_lambda": float,
    "probe_loss_weight": float,
    "nu": float,
    "importance_batches": int,
    "grad_clip_norm": float,
    "eucr_depth": int,
    "kl_warmup_epochs": int,
    "proto_factor": int,
}


def _unique(values: Iterable[Any]) -> List[Any]:
    seen: List[Any] = []
    for value in values:
        if isinstance(value, float):
            value = float(f"{value:.6g}")
        if value not in seen:
            seen.append(value)
    return seen


def _scale_float(
    base: float, factors: Iterable[float], minimum: float | None
) -> List[float]:
    if base <= 0 and minimum is not None:
        base = minimum
    values = []
    for factor in factors:
        candidate = base * factor
        if minimum is not None:
            candidate = max(candidate, minimum)
        values.append(float(candidate))
    return _unique(values)


def _scale_int(base: int, factors: Iterable[float], minimum: int | None) -> List[int]:
    if base <= 0 and minimum is not None:
        base = max(minimum, 1)
    values = []
    for factor in factors:
        candidate = int(round(base * factor))
        if minimum is not None:
            candidate = max(candidate, minimum)
        if candidate <= 0:
            continue
        values.append(candidate)
    if not values and minimum is not None:
        values.append(max(minimum, 1))
    if not values:
        values.append(max(base, 1))
    return sorted(set(values))


def make_grid_factory(spec: GridSpec) -> Callable[[argparse.Namespace], Grid]:
    def factory(args: argparse.Namespace) -> Grid:
        grid: Grid = {}
        for key, meta in spec.items():
            if "values" in meta:
                grid[key] = list(meta["values"])
                continue

            base = getattr(args, key, None)
            if base in (None, 0) and "fallback" in meta:
                base = meta["fallback"]

            if base is None:
                continue

            kind = meta.get("kind", "float")
            factors = meta.get("factors", (0.5, 1.0, 2.0))
            minimum = meta.get("min")
            extra = meta.get("extra", [])

            if kind == "int":
                values = _scale_int(int(round(base)), factors, minimum)
                values.extend(int(v) for v in extra)
                grid[key] = sorted(set(values))
            else:
                values = _scale_float(float(base), factors, minimum)
                values.extend(float(v) for v in extra)
                grid[key] = _unique(values)
        out = {k: v for k, v in grid.items() if v}
        return out

    return factory


TUNING_PRESETS: Dict[str, TuningPreset] = {
    "agem": TuningPreset(
        model_name="agem",
        description="Run grid or random search over AGEM hyperparameters.",
        default_output_root="logs/tuning/agem",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "memory_loss_lambda": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.1,
                    "fallback": 1.0,
                    "values": [0.1, 0.5, 1, 5, 10, 50, 100, 500],
                },
            }
        ),
    ),
    "bcl_dual": TuningPreset(
        model_name="bcl_dual",
        description="Run grid or random search over BCL-Dual hyperparameters.",
        default_output_root="logs/tuning/dual",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },  # "values":[0.001]},
                "memory_strength": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.1,
                    "fallback": 1.0,
                    "values": [1, 10, 100, 1000, 10000],
                },
                "beta": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.1,
                    "fallback": 1.0,
                    "values": [0.3, 1.0, 3.0, 10.0, 30.0],
                },
            }
        ),
    ),
    "ctn": TuningPreset(
        model_name="ctn",
        description="Run grid or random search over CTN hyperparameters.",
        default_output_root="logs/tuning/ctn",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.01,
                    "values": [0.03, 0.01, 0.003, 0.001],
                },  # ,0.0003,0.0001,0.00003,0.00001]},#"values":[0.003]},
                "ctx_lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.05,
                    "values": [0.03, 0.01, 0.003, 0.001],
                },  # ,0.0003,0.0001,0.00003,0.00001]},
                "task_emb": {
                    "kind": "int",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 16,
                    "fallback": 64,
                    "values": [16, 32, 64, 128],
                },
                "memory_strength": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1.0,
                    "fallback": 100.0,
                    "values": [0.1, 0.5, 1, 5, 10, 50, 100],
                },
            }
        ),
    ),
    "eralg4": TuningPreset(
        model_name="eralg4",
        description="Run grid or random search over ER-Alg4 hyperparameters.",
        default_output_root="logs/tuning/eralg4",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "memory_loss_lambda": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.1,
                    "fallback": 1.0,
                    "values": [0.1, 0.5, 1, 5, 10, 50, 100, 500],
                },
            }
        ),
    ),
    "lwf": TuningPreset(
        model_name="lwf",
        description="Run grid or random search over LWF hyperparameters.",
        default_output_root="logs/tuning/lwf",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },  # "values":[0.01]},
                "distill_lambda": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 4.0,
                    "fallback": 10.0,
                    "values": [0.1, 1.0, 5.0, 10.0],
                },
            }
        ),
    ),
    "er_ring": TuningPreset(
        model_name="er_ring",
        description="Run grid or random search over ER-Ring hyperparameters.",
        default_output_root="logs/tuning/er_ring",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "memory_loss_lambda": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.1,
                    "fallback": 1.0,
                    "values": [0.1, 0.5, 1, 5, 10, 50, 100, 500],
                },
            }
        ),
    ),
    "ewc": TuningPreset(
        model_name="ewc",
        description="Run grid or random search over EWC hyperparameters.",
        default_output_root="logs/tuning/ewc",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-5,
                    "fallback": 3e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "lamb": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-2,
                    "fallback": 1.0,
                    "values": [1, 10, 100, 1000, 10000],
                },
            }
        ),
    ),
    "gem": TuningPreset(
        model_name="gem",
        description="Run grid or random search over GEM hyperparameters.",
        default_output_root="logs/tuning/gem",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },  # "values":[0.03]},
                "memory_strength": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.1,
                    "fallback": 0.5,
                    "values": [0.1, 0.5, 1, 5, 10, 50, 100, 500],
                },
            }
        ),
    ),
    "hat": TuningPreset(
        model_name="hat",
        description="Run grid or random search over hat hyperparameters.",
        default_output_root="logs/tuning/hat",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "smax": {
                    "kind": "int",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 32,
                    "fallback": 256,
                    "values": [25, 50, 100, 200, 400, 800],
                },
                "gamma": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.1,
                    "fallback": 0.5,
                    "values": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5],
                },
            }
        ),
    ),
    "icarl": TuningPreset(
        model_name="icarl",
        description="Run grid or random search over iCaRL hyperparameters.",
        default_output_root="logs/tuning/icarl",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },  # "values":[0.01]},
                "memory_strength": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.1,
                    "fallback": 0.5,
                    "values": [0.1, 0.5, 1, 5, 10, 50],
                },
            }
        ),
    ),
    "iid2": TuningPreset(
        model_name="iid2",
        description="Run grid or random search over IID2 hyperparameters.",
        default_output_root="logs/tuning/iid2",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
            }
        ),
    ),
    "lamaml": TuningPreset(
        model_name="lamaml_cifar",
        description="Run grid or random search over La-MAML hyperparameters.",
        default_output_root="logs/tuning/lamaml",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "alpha_init": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.1,
                    "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                },
                "opt_lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.1,
                    "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                },
            }
        ),
    ),
    "smaml": TuningPreset(
        model_name="lamaml_cifar",
        description="Run grid or random search over La-MAML hyperparameters.",
        default_output_root="logs/tuning/smaml",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "alpha_init": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.1,
                    "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                },
                "opt_lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.1,
                    "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                },
                "opt_wt": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.1,
                    "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                },
            }
        ),
    ),
    "cmaml": TuningPreset(
        model_name="lamaml_cifar",
        description="Run grid or random search over La-MAML hyperparameters.",
        default_output_root="logs/tuning/cmaml",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "alpha_init": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.1,
                    "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                },
                "opt_wt": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.1,
                    "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                },
            }
        ),
    ),
    "la-er": TuningPreset(
        model_name="eralg4",
        description="Run grid or random search over La-ER hyperparameters.",
        default_output_root="logs/tuning/la-er",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-2,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "learn_lr": {"kind": "bool", "values": [True]},
                "opt_lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-4,
                    "fallback": 0.1,
                    "values": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                },
            }
        ),
    ),
    "packnet": TuningPreset(
        model_name="packnet",
        description="Run grid or random search over PackNet hyperparameters.",
        default_output_root="logs/tuning/packnet",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
            }
        ),
    ),
    "rwalk": TuningPreset(
        model_name="rwalk",
        description="Run grid or random search over RWalk hyperparameters.",
        default_output_root="logs/tuning/rwalk",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "lamb": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-2,
                    "fallback": 1.0,
                    "values": [0.1, 1, 10, 100, 1000],
                },
                "alpha": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 1.5),
                    "min": 0.1,
                    "fallback": 0.9,
                    "values": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
            }
        ),
    ),
    "si": TuningPreset(
        model_name="si",
        description="Run grid or random search over Synaptic Intelligence hyperparameters.",
        default_output_root="logs/tuning/si",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "si_c": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.01,
                    "fallback": 0.1,
                    "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                },
            }
        ),
    ),
    "woe_si": TuningPreset(
        model_name="woe_si",
        description="Run grid or random search over Weight-of-Evidence SI hyperparameters.",
        default_config="configs/models/til/woe_si.yaml",
        default_output_root="logs/tuning/woe_si",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "woe_lambda": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.01,
                    "fallback": 0.1,
                    "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                },
                "woe_xi": {
                    "kind": "float",
                    "factors": (0.1, 1.0, 10.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [1e-4, 1e-3, 1e-2],
                },
            }
        ),
    ),
    "eucr": TuningPreset(
        model_name="eucr",
        description="Run grid or random search over EUCR hyperparameters.",
        default_config="configs/models/til/eucr.yaml",
        default_output_root="logs/tuning/eucr",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],
                },
                "reg_lambda": {
                    "kind": "float",
                    "factors": (0.3, 1.0, 3.0),
                    "min": 1e-2,
                    "fallback": 1000.0,
                    "values": [1, 10, 100, 1000, 10000],
                },
                "probe_loss_weight": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.05,
                    "fallback": 0.5,
                    "values": [0.1, 0.25, 0.5, 1.0],
                },
            }
        ),
    ),
    "ucl_bresnet": TuningPreset(
        model_name="ucl_bresnet",
        description="Run grid or random search over UCL hyperparameters.",
        default_output_root="logs/tuning/ucl",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [
                        0.03,
                        0.01,
                        0.003,
                        0.001,
                        0.0003,
                        0.0001,
                        0.00003,
                        0.00001,
                    ],  #
                },
                "ratio": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [0.1, 0.25, 0.5],  #
                },
                "beta": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-5,
                    "fallback": 2e-4,
                    "values": [
                        0.0001,
                        0.0002,
                        0.001,
                        0.002,
                        0.01,
                        0.02,
                        # 0.05,
                        # 0.1,
                        # 0.2,
                        # 0,
                        # 1,
                        # 5,
                        # 10,
                        # 50,
                        # 100
                    ],
                },
                "lr_rho": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 1e-5,
                    "fallback": 1e-3,
                    "values": [0.005, 0.001, 0.02, 0.05],  #
                },
                "alpha": {
                    "kind": "float",
                    "factors": (0.5, 1.0, 2.0),
                    "min": 0.05,
                    "fallback": 0.3,
                    "values": [1, 2, 5, 10, 20, 50],
                },
            }
        ),
    ),
}


__all__ = ["COMMON_TYPE_HINTS", "TUNING_PRESETS", "make_grid_factory"]
