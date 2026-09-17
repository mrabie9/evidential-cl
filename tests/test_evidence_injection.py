"""Tests for replay-as-evidence-injection (``woe_replay_mode``, ``woe_replay_store``).

The benchmarking result this project already has is that replay is the only
load-bearing mechanism. The Dempster-Shafer reading of *why* is that replay
re-injects evidence for old classes to counteract the conflict new evidence
introduces -- and if that is really the mechanism, two things follow that are
sharp enough to be wrong:

1. What a buffer should carry is the **weight-of-evidence vector**, not the
   logits. A logit is ``z_k = w+_k - w-_k``, the difference of the two channels,
   so it discards their common magnitude -- the ignorance degree of freedom.
   ``woe_replay_mode='evidence_sym'`` and ``'logit'`` are a matched pair testing
   exactly this: same buffer, same draw, same symmetric squared form against a
   per-item snapshot, differing only in what is distilled.
2. What a buffer should carry is a **sufficient statistic for the evidence**,
   which at the readout is ``phi`` -- not the input that produces it.
   ``woe_replay_store='feature'`` stores 512 floats where the input needs 1024,
   so a matched byte budget buys twice the exemplars.

Covered here: that the pair really is matched (identical form, differing only in
target), that both are zero for an unchanged network and positive after drift,
that a feature buffer stores what it claims and halves the per-item footprint,
that replay through a feature buffer never touches the backbone, and that the
snapshots are taken on the pre-encoding input so both targets come from the same
network at the same moment.
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

from model.woe_si_replay import Net, ReservoirReplayBuffer


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_args(**overrides) -> object:
    """Minimal namespace with the fields ResNet1D / WoE-SI-replay expect."""
    o = type("Args", (), {})()
    o.classes_per_task = overrides.get("classes_per_task", [3, 3])
    o.nc_per_task_list = ""
    o.nc_per_task = None
    o.noise_label = None
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
    o.loader = "task_incremental_loader"
    o.inner_steps = 1
    o.woe_lambda = overrides.get("woe_lambda", 0.0)
    o.woe_xi = 1e-3
    o.woe_centering_mode = "centered_uniform"
    o.woe_mu_momentum = 0.9
    o.woe_importance_stride = 1
    o.woe_conflict_weighting = False
    o.woe_reg_level = "parameter"
    o.woe_anchor_mode = "proximal"
    o.woe_replay_memories = overrides.get("woe_replay_memories", 64)
    o.woe_replay_batch_size = overrides.get("woe_replay_batch_size", 8)
    o.woe_replay_lambda = overrides.get("woe_replay_lambda", 1.0)
    o.woe_replay_mode = overrides.get("woe_replay_mode", "ce")
    o.woe_replay_store = overrides.get("woe_replay_store", "input")
    o.woe_evidence_lambda = overrides.get("woe_evidence_lambda", 1.0)
    o.woe_evidence_readout_only = False
    o.woe_evidence_scale = "weight"
    o.woe_evidence_belief_tau = 1.0
    return o


def _observe(model: Net, x: torch.Tensor, y: torch.Tensor, task: int, steps: int = 3):
    for _ in range(steps):
        model.observe(x, y, task)


# ----------------------------------------------------------------------
# The matched pair: distilling w against distilling z
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["evidence_sym", "logit"])
def test_drift_penalty_is_zero_for_an_unchanged_network(mode: str) -> None:
    """A network scored against its own snapshot pays nothing, in either arm.

    This is the property that makes the pair a fair comparison: both targets are
    recorded from the same network on the same items, so neither arm starts with
    a penalty the other does not have.
    """
    torch.manual_seed(0)
    model = Net(2, 6, 2, _make_args(woe_replay_mode=mode))
    model.eval()
    x = torch.randn(8, 2, 128)
    y = torch.randint(0, 3, (8,))
    model._store_classification_replay(x, y, 0)
    assert len(model.replay_buffer) == 8

    penalty = model._classification_replay_loss(0)
    assert float(penalty) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("mode", ["evidence_sym", "logit"])
def test_drift_penalty_is_positive_after_the_readout_moves(mode: str) -> None:
    """Perturbing the readout makes both arms charge something."""
    torch.manual_seed(1)
    model = Net(2, 6, 2, _make_args(woe_replay_mode=mode))
    model.eval()
    x = torch.randn(8, 2, 128)
    y = torch.randint(0, 3, (8,))
    model._store_classification_replay(x, y, 0)
    with torch.no_grad():
        model.net.model.fc.weight.add_(torch.randn_like(model.net.model.fc.weight))
    assert float(model._classification_replay_loss(0)) > 0.0


def test_the_two_arms_differ_only_in_what_they_distil() -> None:
    """Same draw, same form, different target -- and they are not the same number.

    If the logit and evidence penalties coincided there would be nothing to
    measure. They cannot: ``z_k = w+_k - w-_k``, so a change that moves both
    channels together moves the evidence penalty and leaves the logit one at
    zero. That is the ignorance degree of freedom the comparison is about, and
    this constructs exactly such a change.
    """
    torch.manual_seed(2)
    x = torch.randn(8, 2, 128)
    y = torch.randint(0, 3, (8,))

    penalties = {}
    for mode in ("evidence_sym", "logit"):
        torch.manual_seed(2)
        model = Net(2, 6, 2, _make_args(woe_replay_mode=mode))
        model.eval()
        model._store_classification_replay(x, y, 0)
        with torch.no_grad():
            # Scale the readout: every logit scales with it, but so does the
            # magnitude of both evidence channels, and by different amounts.
            model.net.model.fc.weight.mul_(1.5)
        penalties[mode] = float(model._classification_replay_loss(0))

    assert penalties["evidence_sym"] > 0.0
    assert penalties["logit"] > 0.0
    assert penalties["evidence_sym"] != pytest.approx(penalties["logit"], rel=1e-3)


def test_logit_buffer_stores_logits_and_evidence_buffer_stores_evidence() -> None:
    """Each mode populates only the snapshot it needs."""
    torch.manual_seed(3)
    x = torch.randn(4, 2, 128)
    y = torch.randint(0, 3, (4,))

    logit_model = Net(2, 6, 2, _make_args(woe_replay_mode="logit"))
    logit_model._store_classification_replay(x, y, 0)
    assert len(logit_model.replay_buffer.logits) == 4
    assert logit_model.replay_buffer.logits[0].shape == (6,)
    assert logit_model.replay_buffer.evidence_plus == []

    evidence_model = Net(2, 6, 2, _make_args(woe_replay_mode="evidence_sym"))
    evidence_model._store_classification_replay(x, y, 0)
    assert len(evidence_model.replay_buffer.evidence_plus) == 4
    assert evidence_model.replay_buffer.logits == []


def test_symmetric_mode_charges_improvement_where_one_sided_does_not() -> None:
    """``evidence_sym`` charges any drift; ``evidence`` charges only decay.

    Raising the readout scale increases ``w_plus`` on every item, which the
    one-sided decay penalty is explicitly indifferent to (improvement is free)
    and the symmetric one is not.
    """
    torch.manual_seed(4)
    x = torch.randn(8, 2, 128)
    y = torch.randint(0, 3, (8,))

    charged = {}
    for mode in ("evidence", "evidence_sym"):
        torch.manual_seed(4)
        model = Net(2, 6, 2, _make_args(woe_replay_mode=mode))
        model.eval()
        model._store_classification_replay(x, y, 0)
        with torch.no_grad():
            model.net.model.fc.weight.mul_(2.0)
            model.net.model.fc.bias.mul_(2.0)
        charged[mode] = float(model._classification_replay_loss(0))

    assert charged["evidence_sym"] > charged["evidence"]


# ----------------------------------------------------------------------
# Storing phi instead of the input
# ----------------------------------------------------------------------
def test_feature_buffer_stores_features_at_half_the_footprint() -> None:
    """A feature buffer holds phi (512 floats) where an input buffer holds 1024."""
    torch.manual_seed(5)
    x = torch.randn(4, 2, 512)
    y = torch.randint(0, 3, (4,))

    input_model = Net(2, 6, 2, _make_args(woe_replay_store="input"))
    input_model._store_classification_replay(x, y, 0)
    feature_model = Net(2, 6, 2, _make_args(woe_replay_store="feature"))
    feature_model._store_classification_replay(x, y, 0)

    stored_input = input_model.replay_buffer.inputs[0]
    stored_feature = feature_model.replay_buffer.inputs[0]
    assert stored_feature.shape == (feature_model.feature_dim,)
    assert stored_feature.numel() * 2 == stored_input.numel()


def test_feature_replay_runs_and_sends_no_gradient_to_the_backbone() -> None:
    """Replaying stored features reaches the readout and stops there.

    The backbone is bypassed entirely -- which is the cost of the footprint
    saving, and worth asserting rather than assuming, because a silent backbone
    pass would make the arm an expensive duplicate of input replay.
    """
    torch.manual_seed(6)
    model = Net(2, 6, 2, _make_args(woe_replay_store="feature", woe_replay_mode="ce"))
    x = torch.randn(8, 2, 512)
    y = torch.randint(0, 3, (8,))
    model._store_classification_replay(x, y, 0)

    loss = model._classification_replay_loss(1)
    assert float(loss) > 0.0
    model.zero_grad(set_to_none=True)
    loss.backward()

    readout = model.net.model.fc
    assert readout.weight.grad is not None and readout.weight.grad.abs().sum() > 0
    backbone_grads = [
        parameter.grad
        for name, parameter in model.net.model.named_parameters()
        if not name.startswith("fc.") and parameter.grad is not None
    ]
    assert all(
        float(grad.abs().sum()) == 0.0 for grad in backbone_grads
    ), "replayed features must not propagate a gradient into the backbone"


def test_feature_store_composes_with_evidence_distillation() -> None:
    """The two axes are independent: a feature buffer can also distil evidence."""
    torch.manual_seed(7)
    model = Net(
        2,
        6,
        2,
        _make_args(woe_replay_store="feature", woe_replay_mode="evidence_sym"),
    )
    model.eval()
    x = torch.randn(8, 2, 512)
    y = torch.randint(0, 3, (8,))
    model._store_classification_replay(x, y, 0)
    # Snapshot taken on the input, before encoding, so an unchanged network is
    # still exactly consistent with what the buffer holds.
    assert float(model._classification_replay_loss(0)) == pytest.approx(0.0, abs=1e-5)
    with torch.no_grad():
        model.net.model.fc.weight.add_(torch.randn_like(model.net.model.fc.weight))
    assert float(model._classification_replay_loss(0)) > 0.0


def test_end_to_end_two_tasks_with_feature_storage() -> None:
    """The full observe loop runs across a task boundary on a feature buffer."""
    torch.manual_seed(8)
    model = Net(2, 6, 2, _make_args(woe_replay_store="feature"))
    x = torch.randn(8, 2, 512)
    _observe(model, x, torch.randint(0, 3, (8,)), 0)
    _observe(model, x, torch.randint(3, 6, (8,)), 1)
    model.on_task_end()
    assert len(model.replay_buffer) > 0
    assert model.replay_buffer.inputs[0].shape == (model.feature_dim,)


def test_invalid_store_raises() -> None:
    with pytest.raises(ValueError, match="woe_replay_store"):
        Net(2, 6, 2, _make_args(woe_replay_store="bogus"))


def test_buffer_rejects_a_missing_logit_snapshot() -> None:
    buffer = ReservoirReplayBuffer(capacity=8, store_logits=True)
    with pytest.raises(ValueError, match="store_logits=True"):
        buffer.add(torch.randn(2, 2, 8), torch.zeros(2, dtype=torch.long), task_id=0)
