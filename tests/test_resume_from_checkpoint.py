"""Tests for resuming an interrupted experiment from task checkpoints."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT)

from main import (
    _load_checkpoint_into_model,
    _resolve_resume_plan,
    _save_task_checkpoint,
    life_experience,
)
from model.rwalk import Net as RWalkNet


def _write_checkpoints(experiment_dir: Path, task_ids: list[int]) -> None:
    """Create dummy ``task_<i>.pt`` checkpoints under ``experiment_dir``."""
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        torch.save(
            {"task": task_id, "state_dict": {}},
            checkpoints_dir / "task_{}.pt".format(task_id),
        )


def test_resolve_resume_plan_uses_latest_checkpoint(tmp_path: Path) -> None:
    """Without overrides, resume continues one task past the latest checkpoint."""
    _write_checkpoints(tmp_path, [0, 1, 2])
    args = SimpleNamespace(resume_task=None, cuda=False)

    plan = _resolve_resume_plan(args, str(tmp_path))

    assert plan["experiment_dir"] == str(tmp_path)
    assert plan["resume_from_task"] == 3
    assert plan["checkpoint_path"].endswith("task_2.pt")
    assert plan["tf_dir"] == str(tmp_path / "tfdir")
    assert os.path.isdir(plan["tf_dir"])


def test_resolve_resume_plan_explicit_checkpoint_file(tmp_path: Path) -> None:
    """Pointing --resume at a specific checkpoint resumes after that task."""
    _write_checkpoints(tmp_path, [0, 1, 2])
    args = SimpleNamespace(resume_task=None, cuda=False)

    checkpoint_file = tmp_path / "checkpoints" / "task_1.pt"
    plan = _resolve_resume_plan(args, str(checkpoint_file))

    assert plan["resume_from_task"] == 2
    assert plan["checkpoint_path"].endswith("task_1.pt")
    assert plan["experiment_dir"] == str(tmp_path)


def test_resolve_resume_plan_respects_resume_task_override(tmp_path: Path) -> None:
    """``--resume_task N`` loads ``task_<N-1>.pt`` even if later ones exist."""
    _write_checkpoints(tmp_path, [0, 1, 2, 3])
    args = SimpleNamespace(resume_task=2, cuda=False)

    plan = _resolve_resume_plan(args, str(tmp_path))

    assert plan["resume_from_task"] == 2
    assert plan["checkpoint_path"].endswith("task_1.pt")


def test_resolve_resume_plan_missing_override_checkpoint_errors(
    tmp_path: Path,
) -> None:
    """A resume_task that needs an absent checkpoint fails loudly."""
    _write_checkpoints(tmp_path, [0, 1])
    args = SimpleNamespace(resume_task=5, cuda=False)

    with pytest.raises(SystemExit):
        _resolve_resume_plan(args, str(tmp_path))


def test_resolve_resume_plan_no_checkpoints_errors(tmp_path: Path) -> None:
    """Resuming a directory without checkpoints fails loudly."""
    (tmp_path / "checkpoints").mkdir()
    args = SimpleNamespace(resume_task=None, cuda=False)

    with pytest.raises(SystemExit):
        _resolve_resume_plan(args, str(tmp_path))


def _minimal_rwalk_args(*, n_tasks: int) -> object:
    """Build args for a small RWalk model used in resume tests."""
    args = type("Args", (), {})()
    args.class_incremental = True
    args.classes_per_task = [5] * n_tasks
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.noise_label = None
    args.class_weighted_ce = False
    args.use_detector_arch = False
    args.use_iq_aug_features = False
    args.data_scaling = "none"
    args.iq_aug_feature_type = "power"
    args.lr = 0.01
    args.optimizer = "adam"
    args.lamb = 1.0
    args.alpha = 0.9
    args.eps = 0.01
    args.clipgrad = 100.0
    args.det_lambda = 1.0
    args.cls_lambda = 1.0
    args.det_memories = 0
    args.det_replay_batch = 64
    args.norm_track_stats = True
    args.alpha_init = 1e-3
    args.inner_steps = 1
    args.loader = "task_incremental_loader"
    return args


def test_load_checkpoint_into_model_round_trips_weights(tmp_path: Path) -> None:
    """Weights saved by ``_save_task_checkpoint`` are restored on load."""
    source = RWalkNet(128, 11, 1, _minimal_rwalk_args(n_tasks=1))
    with torch.no_grad():
        for param in source.parameters():
            param.add_(0.123)
    checkpoint_path = _save_task_checkpoint(source, str(tmp_path), task_index=0)

    target = RWalkNet(128, 11, 1, _minimal_rwalk_args(n_tasks=1))
    _load_checkpoint_into_model(target, checkpoint_path, SimpleNamespace(cuda=False))

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert torch.allclose(source_param, target_param)


class _MultiTaskLoader:
    """Minimal incremental loader that yields ``n_tasks`` single-batch tasks."""

    def __init__(self, n_tasks: int) -> None:
        self.n_tasks = n_tasks
        self.new_task_calls = 0
        self._t = 0

    def new_task(self):
        task_id = self._t
        self.new_task_calls += 1
        self._t += 1
        features = torch.randn(4, 2, 64)
        labels = torch.randint(0, 5, (4,))
        train_loader = DataLoader(TensorDataset(features, labels), batch_size=4)
        test_loader = DataLoader(TensorDataset(features, labels), batch_size=4)
        return {"task": task_id}, train_loader, None, test_loader


def test_life_experience_skips_completed_tasks_when_resuming(tmp_path: Path) -> None:
    """Resumed runs rebuild every loader but only train remaining tasks."""
    args = SimpleNamespace(
        model="rwalk",
        arch="resnet1d",
        loader="task_incremental_loader",
        cuda=False,
        amp=False,
        amp_dtype="bfloat16",
        state_logging=False,
        n_epochs=1,
        use_detector_arch=False,
        classes_per_task=5,
        noise_label=None,
        class_order="sequential",
        log_dir=str(tmp_path),
        calc_test_accuracy=False,
        val_rate=10_000,
        resume_from_task=2,
    )
    model = RWalkNet(128, 11, 3, _minimal_rwalk_args(n_tasks=3))
    loader = _MultiTaskLoader(n_tasks=3)

    observed_tasks: list[int] = []
    original_observe = model.observe

    def _counting_observe(x, y, t):
        observed_tasks.append(t)
        return original_observe(x, y, t)

    model.observe = _counting_observe

    def _fake_eval(_model, tasks, _args, **_kwargs):
        zeros = [0.5] * len(tasks)
        return zeros, zeros, zeros, zeros, zeros

    saved_tasks: list[int] = []

    def _record_checkpoint(_model, _log_dir, task_index):
        saved_tasks.append(task_index)
        return str(tmp_path / "ckpt.pt")

    with (
        patch("main.eval_tasks", side_effect=_fake_eval),
        patch("main._save_task_checkpoint", side_effect=_record_checkpoint),
    ):
        life_experience(model, loader, args)

    # Every task advanced the loader so prior-task loaders exist for retention.
    assert loader.new_task_calls == 3
    # Only the resumed task (id 2) trained and checkpointed; 0 and 1 were skipped.
    assert set(observed_tasks) == {2}
    assert saved_tasks == [2]
