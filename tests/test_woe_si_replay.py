"""Tests for WoE-SI + reservoir replay (``model/woe_si_replay.py``).

Covers the reservoir buffer's capacity/uniformity bookkeeping, that base WoE-SI
adds no replay contribution (hooks are no-ops), and a short integration check
that the subclass observes across two tasks and actually fills its buffer.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.woe_si import Net as WoeSiNet
from model.woe_si_replay import Net, ReservoirReplayBuffer


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_args(loader: str, **overrides) -> object:
    """Minimal namespace with the fields ResNet1D / WoE-SI-replay expect."""
    o = type("Args", (), {})()
    o.classes_per_task = overrides.get("classes_per_task", [3, 3])
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.noise_label = overrides.get("noise_label", None)
    o.class_weighted_ce = False
    o.use_detector_arch = False
    o.use_iq_aug_features = False
    o.data_scaling = "none"
    o.iq_aug_feature_type = "power"
    o.lr = overrides.get("lr", 0.01)
    o.optimizer = "sgd"
    o.clipgrad = 100.0
    o.cls_lambda = 1.0
    o.det_memories = 0
    o.det_replay_batch = 64
    o.alpha_init = 1e-3
    o.loader = loader
    o.inner_steps = 1
    o.woe_lambda = overrides.get("woe_lambda", 0.5)
    o.woe_xi = overrides.get("woe_xi", 1e-3)
    o.woe_centering_mode = "centered_uniform"
    o.woe_mu_momentum = 0.9
    o.woe_importance_stride = 1
    o.woe_conflict_weighting = False
    o.woe_reg_level = overrides.get("woe_reg_level", "parameter")
    o.woe_replay_memories = overrides.get("woe_replay_memories", 64)
    o.woe_replay_batch_size = overrides.get("woe_replay_batch_size", 8)
    o.woe_replay_lambda = overrides.get("woe_replay_lambda", 1.0)
    return o


# ----------------------------------------------------------------------
# Reservoir buffer
# ----------------------------------------------------------------------
def test_reservoir_buffer_fills_then_caps_at_capacity() -> None:
    """The buffer grows to capacity and never exceeds it."""
    buffer = ReservoirReplayBuffer(capacity=10)
    for _ in range(5):
        buffer.add(torch.randn(4, 2, 8), torch.zeros(4, dtype=torch.long), task_id=0)
    assert len(buffer) == 10
    assert buffer.seen == 20


def test_reservoir_buffer_zero_capacity_is_inert() -> None:
    """Capacity <= 0 stores nothing and samples None."""
    buffer = ReservoirReplayBuffer(capacity=0)
    buffer.add(torch.randn(4, 2, 8), torch.zeros(4, dtype=torch.long), task_id=0)
    assert len(buffer) == 0
    assert buffer.sample(4) is None


def test_reservoir_buffer_sample_shapes_and_bounds() -> None:
    """A draw returns aligned (inputs, labels, tasks) capped at buffer size."""
    buffer = ReservoirReplayBuffer(capacity=100)
    buffer.add(torch.randn(6, 2, 8), torch.arange(6), task_id=2)
    draw = buffer.sample(16)  # request more than stored
    assert draw is not None
    inputs, labels, tasks = draw
    assert inputs.shape == (6, 2, 8)
    assert labels.shape == (6,)
    assert tasks.shape == (6,)
    assert torch.all(tasks == 2)


def test_reservoir_retains_all_items_with_equal_probability() -> None:
    """Over many trials each stream item is kept with frequency ~ capacity/seen."""
    torch.manual_seed(0)
    import random

    random.seed(0)
    capacity, stream = 5, 20
    trials = 4000
    counts = [0] * stream
    for _ in range(trials):
        buffer = ReservoirReplayBuffer(capacity)
        for item in range(stream):
            buffer.add(
                torch.full((1, 1), float(item)),
                torch.tensor([item]),
                task_id=0,
            )
        for label in buffer.labels:
            counts[label] += 1
    expected = trials * capacity / stream
    # Each item should be retained about equally often (loose tolerance).
    for count in counts:
        assert abs(count - expected) < 0.25 * expected


# ----------------------------------------------------------------------
# Hook behaviour
# ----------------------------------------------------------------------
def test_base_woe_si_replay_hooks_are_noops() -> None:
    """Base WoE-SI adds no replay loss and stores nothing."""
    model = WoeSiNet(2, 6, 2, _make_args("task_incremental_loader"))
    loss = model._classification_replay_loss(0)
    assert float(loss.item()) == 0.0
    # No buffer attribute exists on the base learner.
    assert not hasattr(model, "replay_buffer")


def test_replay_loss_zero_before_any_storage() -> None:
    """With an empty buffer the replay term is exactly zero (task 0, step 0)."""
    model = Net(2, 6, 2, _make_args("task_incremental_loader"))
    assert float(model._classification_replay_loss(0).item()) == 0.0


def test_observe_fills_buffer_and_replays_across_tasks() -> None:
    """Two-task run: buffer fills and a positive replay loss appears on task 1."""
    torch.manual_seed(0)
    model = Net(
        2,
        6,
        2,
        _make_args("task_incremental_loader", woe_replay_memories=64),
    )
    x = torch.randn(8, 2, 128)
    y0 = torch.randint(0, 3, (8,))
    y1 = torch.randint(3, 6, (8,))
    for _ in range(4):
        model.observe(x, y0, 0)
    assert len(model.replay_buffer) > 0
    # After task 0 the buffer is non-empty, so the replay term is positive.
    replay_loss = model._classification_replay_loss(1)
    assert float(replay_loss.item()) > 0.0
    # And observing task 1 keeps everything finite.
    loss, _rec, _logits = model.observe(x, y1, 1)
    assert torch.isfinite(torch.tensor(loss))


def test_replay_lambda_zero_disables_replay_term() -> None:
    """woe_replay_lambda=0 yields a zero replay contribution even when filled."""
    torch.manual_seed(0)
    model = Net(
        2,
        6,
        2,
        _make_args("task_incremental_loader", woe_replay_lambda=0.0),
    )
    x = torch.randn(8, 2, 128)
    y0 = torch.randint(0, 3, (8,))
    for _ in range(3):
        model.observe(x, y0, 0)
    assert len(model.replay_buffer) > 0
    assert float(model._classification_replay_loss(1).item()) == 0.0
