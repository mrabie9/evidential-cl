"""Tests for eralg4's LwF distillation term (``--eralg4_lwf_lambda``).

``model.eralg4`` is the fourth host of ``model/lwf_regulariser.py`` and the
first replay-based one, so the cases that matter here are the wiring ones: the
term is off by default, a task transition freezes the teacher, and the
distillation gradient reaches the parameters through *both* training loops --
the default concatenated forward (``ER``) and the two-forward joint loop
(``ER_joint``, the pinned CIL operating point).  The numerics themselves are
covered once, for every host, in ``tests/test_lwf_regulariser.py``.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import random
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import eralg4

JOINT = [False, True]
JOINT_IDS = ["concat", "joint"]


def _make_args(**overrides) -> object:
    """Minimal namespace with the fields ``ResNet1D`` / ``ErAlgConfig`` expect."""
    o = type("Args", (), {})()
    o.classes_per_task = [3, 3]
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.class_weighted_ce = False
    o.use_iq_aug_features = False
    o.data_scaling = "none"
    o.iq_aug_feature_type = "power"
    o.arch = "resnet1d"
    o.dataset = "tinyimagenet"
    o.cuda = False
    o.loader = overrides.get("loader", "class_incremental_loader")
    o.class_incremental = True
    o.lr = 0.01
    o.opt_lr = 0.1
    o.alpha_init = 1e-3
    o.learn_lr = False
    o.second_order = False
    o.meta_batches = 1
    o.inner_steps = 1
    o.memories = 64
    o.replay_batch_size = 4
    o.grad_clip_norm = 2.0
    o.memory_loss_lambda = 1.0
    o.cls_lambda = 1.0
    o.eralg4_masked_loss = True
    o.eralg4_grad_avg = 1
    o.eralg4_joint_er = overrides.get("joint", False)
    o.er_distill = overrides.get("er_distill", False)
    o.memory_strength = 1.0
    o.temperature = 5.0
    o.eralg4_lwf_lambda = overrides.get("lwf_lambda", 0.0)
    o.eralg4_lwf_temperature = overrides.get("lwf_temperature", 5.0)
    return o


def _build(**overrides):
    return eralg4.Net(1, 6, 2, _make_args(**overrides))


def _paired_run(model, x, y_first, y_second) -> None:
    """One step on task 0 and two on task 1, with every RNG the loop reads pinned.

    ResNet1D carries no dropout, so on the first step of task 1 the student is
    still identical to the teacher just frozen from it and the LwF gradient is
    exactly zero. The second task-1 step is the first one it can move.
    """
    for y, task in ((y_first, 0), (y_second, 1), (y_second, 1)):
        torch.manual_seed(99)
        random.seed(99)
        model.observe(x, y, task)


# ----------------------------------------------------------------------
def test_lwf_term_is_off_by_default() -> None:
    model = _build()
    assert model.lwf_lambda == 0.0
    x = torch.randn(4, 2, 1024)
    logits = model.net(x)
    assert float(model._lwf_distillation_loss(logits, x, 1).item()) == 0.0


def test_no_teacher_without_distillation() -> None:
    """With both distillation flags off, a boundary must not pay for a deepcopy."""
    torch.manual_seed(1)
    model = _build()
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    model.observe(x, torch.randint(3, 6, (6,)), 1)
    assert model.teacher is None


@pytest.mark.parametrize("joint", JOINT, ids=JOINT_IDS)
def test_task_boundary_snapshots_a_teacher(joint: bool) -> None:
    torch.manual_seed(2)
    model = _build(lwf_lambda=1.0, joint=joint)
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    assert model.teacher is None
    model.observe(x, torch.randint(3, 6, (6,)), 1)
    assert model.teacher is not None


def test_er_distill_still_gets_a_teacher_when_lwf_is_on() -> None:
    """The two terms share one snapshot; neither may starve the other."""
    torch.manual_seed(3)
    model = _build(lwf_lambda=1.0, er_distill=True)
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    model.observe(x, torch.randint(3, 6, (6,)), 1)
    assert model.use_distill and model.teacher is not None


@pytest.mark.parametrize("joint", JOINT, ids=JOINT_IDS)
def test_lwf_changes_the_update(joint: bool) -> None:
    """The term must reach the parameters, not just the reported loss.

    Two models train from identical weights on identical batches, one with
    distillation on.  The reservoir draws (``random``, which ``getBatch`` /
    ``_sample_replay`` use) are re-seeded per step, so the only difference
    between the runs is the distillation gradient.  Task 0 cannot differ (no
    teacher yet, and no previous classes), so the divergence is checked after
    two steps on task 1.
    ``test_lwf_off_leaves_the_update_identical`` guards this pairing: without
    the ``random`` seeding both arms drift apart on replay sampling alone and
    the assertion below would hold vacuously.
    """
    torch.manual_seed(4)
    plain = _build(lwf_lambda=0.0, joint=joint)
    distilled = _build(lwf_lambda=1.0, joint=joint)
    distilled.net.load_state_dict(plain.net.state_dict())

    x = torch.randn(6, 2, 1024)
    y_first = torch.randint(0, 3, (6,))
    y_second = torch.randint(3, 6, (6,))
    _paired_run(plain, x, y_first, y_second)
    _paired_run(distilled, x, y_first, y_second)

    weight = "model.fc.weight"
    assert not torch.allclose(
        plain.net.state_dict()[weight], distilled.net.state_dict()[weight]
    )


@pytest.mark.parametrize("joint", JOINT, ids=JOINT_IDS)
def test_lwf_off_leaves_the_update_identical(joint: bool) -> None:
    """Negative control for the test above: same seeding, no distillation."""
    torch.manual_seed(4)
    first = _build(lwf_lambda=0.0, joint=joint)
    second = _build(lwf_lambda=0.0, joint=joint)
    second.net.load_state_dict(first.net.state_dict())

    x = torch.randn(6, 2, 1024)
    y_first = torch.randint(0, 3, (6,))
    y_second = torch.randint(3, 6, (6,))
    _paired_run(first, x, y_first, y_second)
    _paired_run(second, x, y_first, y_second)

    weight = "model.fc.weight"
    assert torch.allclose(
        first.net.state_dict()[weight], second.net.state_dict()[weight]
    )
