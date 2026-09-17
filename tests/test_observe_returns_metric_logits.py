"""Tests for ``observe()`` returning metric logits and ``life_experience`` consumption."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT)

from main import life_experience
from model.rwalk import Net as RWalkNet
from utils.training_forward import unpack_observe_result


def _minimal_rwalk_args(*, n_tasks: int = 2) -> object:
    """Build args for a small RWalk smoke test."""
    args = type("Args", (), {})()
    args.class_incremental = True
    args.classes_per_task = [5, 6][:n_tasks] if n_tasks <= 2 else [5] * n_tasks
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.class_weighted_ce = False
    args.use_iq_aug_features = False
    args.data_scaling = "none"
    args.iq_aug_feature_type = "power"
    args.lr = 0.01
    args.optimizer = "adam"
    args.lamb = 1.0
    args.alpha = 0.9
    args.eps = 0.01
    args.clipgrad = 100.0
    args.cls_lambda = 1.0
    args.norm_track_stats = True
    args.alpha_init = 1e-3
    args.inner_steps = 1
    args.loader = "task_incremental_loader"
    return args


def _minimal_life_experience_args() -> SimpleNamespace:
    """Build minimal args for a one-batch ``life_experience`` smoke test."""
    return SimpleNamespace(
        model="rwalk",
        arch="resnet1d",
        loader="task_incremental_loader",
        cuda=False,
        amp=False,
        amp_dtype="bfloat16",
        state_logging=False,
        n_epochs=1,
        classes_per_task=5,
        class_order="sequential",
    )


class _SingleBatchLoader:
    """One-task incremental loader with a single training batch."""

    n_tasks = 1

    def new_task(self):
        features = torch.randn(4, 2, 64)
        labels = torch.randint(0, 5, (4,))
        train_loader = DataLoader(TensorDataset(features, labels), batch_size=4)
        test_loader = DataLoader(TensorDataset(features, labels), batch_size=4)
        return {"task": 0}, train_loader, None, test_loader


def test_rwalk_observe_returns_detached_metric_logits() -> None:
    """RWalk ``observe`` exposes logits aligned with the training forward."""
    model = RWalkNet(128, 11, 2, _minimal_rwalk_args())
    model.train()
    torch.manual_seed(0)
    batch_x = torch.randn(8, 2, 64)
    batch_y = torch.randint(0, 5, (8,))

    loss, recall, metric_logits = unpack_observe_result(
        model.observe(batch_x, batch_y, t=0)
    )

    assert isinstance(loss, float)
    assert isinstance(recall, float)
    assert metric_logits is not None
    assert metric_logits.shape[0] == 8
    assert metric_logits.shape[1] == model.n_outputs
    assert not metric_logits.requires_grad
    assert torch.isfinite(metric_logits).all()


def test_life_experience_skips_metric_forward_when_logits_returned(
    tmp_path: Path,
) -> None:
    """``life_experience`` must not call the eval metric forward when logits exist."""
    args = _minimal_life_experience_args()
    args.log_dir = str(tmp_path)
    args.calc_test_accuracy = False
    args.val_rate = 10_000
    model = RWalkNet(128, 11, 1, _minimal_rwalk_args(n_tasks=1))
    loader = _SingleBatchLoader()

    def _fake_eval(_model, tasks, _args, **_kwargs):
        task_count = len(tasks)
        per_task = [0.5] * task_count
        # (macro_rec, macro_prec, macro_f1)
        return per_task, per_task, per_task

    with (
        patch("main.eval_tasks", side_effect=_fake_eval),
        patch("main._save_task_checkpoint", return_value=str(tmp_path / "ckpt.pt")),
        patch(
            "main.model_forward_for_metric_loop",
            side_effect=AssertionError("metric forward should not run"),
        ),
    ):
        life_experience(model, loader, args)


def test_unpack_observe_result_accepts_legacy_two_tuple() -> None:
    """Unpack helper remains compatible with legacy 2-tuple returns."""
    loss, recall, metric_logits = unpack_observe_result((1.5, 0.25))
    assert loss == 1.5
    assert recall == 0.25
    assert metric_logits is None
