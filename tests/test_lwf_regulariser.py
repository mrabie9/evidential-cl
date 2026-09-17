"""Tests for the shared LwF distillation mixin (``model/lwf_regulariser.py``).

Every case runs against both hosts -- ``model.si`` and ``model.rwalk`` -- since
the point of the mixin is that the two behave identically here.  Mirrors the
``woe_lwf_lambda`` block of ``tests/test_woe_si.py``: off by default, zero
before a teacher exists, zero against an identical teacher, the teacher stays
frozen, positive once the student moves, and the term reaches the gradient.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import rwalk, si

# (module, flag prefix, the host's own anchor-strength attribute)
HOSTS = [(si, "si"), (rwalk, "rwalk")]
HOST_IDS = ["si", "rwalk"]


def _make_args(prefix: str, **overrides) -> object:
    """Minimal namespace with the fields ``ResNet1D`` / SI / RWalk expect."""
    o = type("Args", (), {})()
    o.classes_per_task = overrides.get("classes_per_task", [3, 3])
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.class_weighted_ce = False
    o.use_iq_aug_features = False
    o.data_scaling = "none"
    o.iq_aug_feature_type = "power"
    o.lr = overrides.get("lr", 0.01)
    o.optimizer = "sgd"
    o.clipgrad = 100.0
    o.cls_lambda = 1.0
    o.alpha_init = 1e-3
    o.loader = overrides.get("loader", "task_incremental_loader")
    o.class_incremental = True
    o.inner_steps = 1
    o.anchor_mode = overrides.get("anchor_mode", "loss")
    # SI knobs.
    o.si_c = overrides.get("si_c", 0.1)
    o.si_epsilon = 0.01
    # RWalk knobs.
    o.lamb = overrides.get("lamb", 1.0)
    o.alpha = 0.9
    o.eps = 0.01
    # The distillation flags under test, on whichever namespace the host reads.
    setattr(o, f"{prefix}_lwf_lambda", overrides.get("lwf_lambda", 0.0))
    setattr(o, f"{prefix}_lwf_temperature", overrides.get("lwf_temperature", 5.0))
    return o


def _build(module, prefix: str, **overrides):
    return module.Net(1, 6, 2, _make_args(prefix, **overrides))


# ----------------------------------------------------------------------
@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_term_is_off_by_default(module, prefix: str) -> None:
    model = _build(module, prefix)
    assert model.lwf_lambda == 0.0
    x = torch.randn(4, 2, 1024)
    logits = model.net(x)
    assert float(model._lwf_distillation_loss(logits, x, 1).item()) == 0.0


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_snapshot_is_a_noop_when_disabled(module, prefix: str) -> None:
    """A consolidation with the term off must not pay for a ``deepcopy``."""
    torch.manual_seed(1)
    model = _build(module, prefix)
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    model._consolidate_current_task()
    assert model.teacher is None


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_zero_before_a_teacher_exists(module, prefix: str) -> None:
    model = _build(module, prefix, lwf_lambda=1.0)
    x = torch.randn(4, 2, 1024)
    logits = model.net(x)
    assert model.teacher is None
    assert float(model._lwf_distillation_loss(logits, x, 1).item()) == 0.0


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_zero_on_the_first_task(module, prefix: str) -> None:
    """No completed tasks means no previous classes, hence nothing to preserve."""
    torch.manual_seed(2)
    model = _build(module, prefix, lwf_lambda=1.0)
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    model._snapshot_lwf_teacher()
    logits = model.net(x)
    assert float(model._lwf_distillation_loss(logits, x, 0).item()) == 0.0


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_zero_against_an_identical_teacher(module, prefix: str) -> None:
    """KL(p || p) = 0, so a self-distilling network is charged nothing.

    Both sides are scored with ``bn_training=True`` (batch statistics), matching
    ``model.lwf``. That also activates the backbone's 4 dropout modules, so the
    teacher forward is *stochastic*; the seed is reset before each forward so the
    dropout masks coincide.
    """
    torch.manual_seed(3)
    model = _build(module, prefix, lwf_lambda=1.0)
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    model._snapshot_lwf_teacher()
    torch.manual_seed(123)
    logits = model.net(x, bn_training=True)
    torch.manual_seed(123)
    assert abs(float(model._lwf_distillation_loss(logits, x, 1).item())) < 1e-5


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_does_not_mutate_the_frozen_teacher(module, prefix: str) -> None:
    """The teacher must stay frozen across steps, statistics included.

    ``num_batches_tracked`` is excluded: it still increments, but it is only read
    when ``momentum is None``, so it cannot affect the teacher's output.
    """
    torch.manual_seed(6)
    model = _build(module, prefix, lwf_lambda=1.0)
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    model._snapshot_lwf_teacher()
    before = {
        k: v.clone()
        for k, v in model.teacher.state_dict().items()
        if not k.endswith("num_batches_tracked")
    }
    for _ in range(3):
        model.observe(x, torch.randint(3, 6, (6,)), 1)
    after = model.teacher.state_dict()
    changed = [
        k for k in before if not torch.equal(before[k].float(), after[k].float())
    ]
    assert changed == [], f"teacher drifted on {len(changed)} buffers: {changed[:3]}"


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_positive_once_the_student_moves(module, prefix: str) -> None:
    torch.manual_seed(4)
    model = _build(module, prefix, lwf_lambda=1.0)
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    model._snapshot_lwf_teacher()
    with torch.no_grad():
        model.net.model.fc.weight.mul_(3.0)
    logits = model.net(x)
    assert float(model._lwf_distillation_loss(logits, x, 1).item()) > 0.0


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_task_boundary_snapshots_a_teacher(module, prefix: str) -> None:
    """The teacher is frozen by the ordinary task transition inside ``observe``."""
    torch.manual_seed(5)
    model = _build(module, prefix, lwf_lambda=1.0)
    x = torch.randn(6, 2, 1024)
    model.observe(x, torch.randint(0, 3, (6,)), 0)
    assert model.teacher is None
    model.observe(x, torch.randint(3, 6, (6,)), 1)
    assert model.teacher is not None


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_changes_the_update(module, prefix: str) -> None:
    """The term must reach the parameters, not just the reported loss.

    Two models are trained from identical initial weights on identical batches,
    one with distillation on. The dropout masks are re-seeded per step so the
    only difference between the runs is the distillation gradient.
    """
    torch.manual_seed(7)
    plain = _build(module, prefix, lwf_lambda=0.0)
    distilled = _build(module, prefix, lwf_lambda=1.0)
    distilled.net.load_state_dict(plain.net.state_dict())

    x = torch.randn(6, 2, 1024)
    y_first = torch.randint(0, 3, (6,))
    y_second = torch.randint(3, 6, (6,))
    for model in (plain, distilled):
        for step, (labels, task) in enumerate(
            [(y_first, 0), (y_second, 1), (y_second, 1)]
        ):
            torch.manual_seed(100 + step)
            model.observe(x, labels, task)

    weight_plain = plain.net.model.fc.weight
    weight_distilled = distilled.net.model.fc.weight
    assert not torch.allclose(weight_plain, weight_distilled, atol=1e-8)


@pytest.mark.parametrize("module,prefix", HOSTS, ids=HOST_IDS)
def test_lwf_rejects_non_positive_temperature(module, prefix: str) -> None:
    with pytest.raises(ValueError):
        _build(module, prefix, lwf_lambda=1.0, lwf_temperature=0.0)
