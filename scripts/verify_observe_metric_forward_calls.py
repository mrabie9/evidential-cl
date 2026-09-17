#!/usr/bin/env python3
"""Verify Tier-1 models skip train-batch metric forward; Tier-2 still call it."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import life_experience


class _SingleTaskLoader:
    n_tasks = 1

    def new_task(self):
        features = torch.randn(32, 2, 64)
        labels = torch.randint(0, 5, (32,))
        train_loader = DataLoader(TensorDataset(features, labels), batch_size=16)
        test_loader = DataLoader(TensorDataset(features, labels), batch_size=16)
        return {"task": 0}, train_loader, None, test_loader


def _minimal_args(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_name,
        arch="resnet1d",
        loader="task_incremental_loader",
        cuda=False,
        amp=False,
        amp_dtype="bfloat16",
        state_logging=False,
        n_epochs=1,
        classes_per_task=[5],
        val_rate=10_000,
        calc_test_accuracy=False,
        log_dir=os.path.join(ROOT, "logs", "verify_metric_forward"),
    )


def _run_and_count_metric_forwards(model_module: str, model_yaml_name: str) -> int:
    import importlib

    from utils import misc_utils

    misc_utils.init_seed(0)
    Model = importlib.import_module(f"model.{model_module}")
    model = Model.Net(
        128,
        10,
        1,
        type(
            "A",
            (),
            {
                "class_incremental": True,
                "classes_per_task": [5],
                "nc_per_task_list": "",
                "nc_per_task": None,
                "class_weighted_ce": False,
                "use_iq_aug_features": False,
                "data_scaling": "none",
                "iq_aug_feature_type": "power",
                "lr": 0.01,
                "optimizer": "adam",
                "inner_steps": 1,
                "loader": "task_incremental_loader",
                "memory_loss_lambda": 1.0,
                "grad_clip_norm": 2.0,
                "memories": 64,
                "n_memories": 64,
                "replay_batch_size": 16,
                "alpha_init": 1e-3,
                "opt_wt": 0.01,
                "opt_lr": 0.1,
                "learn_lr": False,
                "sync_update": True,
                "second_order": False,
                "meta_batches": 2,
                "dataset": "iq",
                "cls_lambda": 1.0,
                "norm_track_stats": True,
                "lamb": 1.0,
                "alpha": 0.9,
                "eps": 0.01,
                "clipgrad": 100.0,
            },
        )(),
    )
    loader = _SingleTaskLoader()
    args = _minimal_args(model_yaml_name)
    metric_forward_calls = 0

    def _counting_forward(*_a, **_kw):
        nonlocal metric_forward_calls
        metric_forward_calls += 1
        return torch.zeros(16, 10)

    def _fake_eval(*_a, **_kw):
        return [0.5], [0.5], [0.5], [0.0], [0.0]

    with (
        patch("main.eval_tasks", side_effect=_fake_eval),
        patch("main._save_task_checkpoint", return_value="/tmp/ckpt.pt"),
        patch("main.model_forward_for_metric_loop", side_effect=_counting_forward),
    ):
        life_experience(model, loader, args)
    return metric_forward_calls


def main() -> None:
    tier1_calls = _run_and_count_metric_forwards("agem", "agem")
    tier2_calls = _run_and_count_metric_forwards("lamaml_cifar", "cmaml")
    print(f"agem (tier1, expects 0 train-batch metric forwards): {tier1_calls}")
    print(f"cmaml (tier2, expects >0 train-batch metric forwards): {tier2_calls}")
    if tier1_calls != 0:
        raise SystemExit(
            f"Tier-1 agem unexpectedly called metric forward {tier1_calls} times"
        )
    if tier2_calls == 0:
        raise SystemExit("Tier-2 cmaml never called metric forward during training")
    print("PASS: metric forward call counts match Tier-1/Tier-2 expectations")


if __name__ == "__main__":
    main()
