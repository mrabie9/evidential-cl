"""Tests for the deterministic distillation teacher (``--woe_teacher_dropout``).

``_lwf_distillation_loss`` scores the frozen teacher with ``bn_training=True`` on
purpose: consecutive tasks here are different radar datasets, so a teacher
normalised with the previous task's running statistics is evaluated under
distribution shift (readme B4 measures the cost at 0.068 of final macro recall).
But ``ResNet1D.forward`` implements that request as ``self.model.train(True)``,
which is a *module-wide* mode switch -- it also turns the backbone's four trunk
dropout modules back on, no matter that ``_snapshot_teacher`` called ``.eval()``.
The distillation target is therefore resampled every step.

``woe_teacher_dropout='disable'`` zeroes ``p`` on the teacher copy, which
separates the two effects that share that switch: batch statistics stay, noise
goes. Covered here: that the default really is stochastic, that the flag makes
the teacher deterministic, that it does *not* disturb the batch-statistic
normalisation the B4 measurement depends on, that the student's own dropout is
untouched, and that the flag is inert before a teacher exists.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.woe_si import Net as WoeSiNet


@pytest.fixture(autouse=True)
def _trunk_dropout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-enable the historical p=0.2 trunk dropout these tests are about.

    ``ResNet1D`` now defaults to no trunk dropout, which would leave nothing for
    the teacher flag to switch off.
    """
    monkeypatch.setenv("RESNET1D_DROPOUT", "0.2")


def _woe_args(**overrides) -> SimpleNamespace:
    """Minimal namespace with the fields ``ResNet1D`` / WoE-SI expect."""
    base = dict(
        arch="resnet1d",
        cuda=False,
        use_detector_arch=False,
        class_weighted_ce=False,
        data_scaling="none",
        use_iq_aug_features=False,
        iq_aug_feature_type="power",
        classes_per_task=[3, 3],
        nc_per_task_list="",
        nc_per_task=None,
        noise_label=None,
        loader="task_incremental_loader",
        alpha_init=1e-3,
        optimizer="sgd",
        clipgrad=100.0,
        cls_lambda=1.0,
        det_memories=0,
        det_replay_batch=64,
        inner_steps=1,
        lr=0.01,
        woe_lambda=0.0,
        woe_xi=1e-3,
        woe_centering_mode="centered_uniform",
        woe_mu_momentum=0.9,
        woe_importance_stride=1,
        woe_conflict_weighting=False,
        woe_reg_level="parameter",
        woe_anchor_mode="proximal",
        woe_omega_winsorise=0.0,
        woe_omega_transform="relu",
        woe_omega_accum="sum",
        woe_importance_scalar="i2",
        woe_evidence_scale="weight",
        woe_evidence_belief_tau=1.0,
        woe_evidence_asymmetric=False,
        woe_evidence_distill_lambda=0.0,
        woe_lwf_lambda=1.0,
        woe_lwf_temperature=5.0,
        woe_lc_lambda=0.0,
        woe_lc_readout_only=False,
        woe_lc_term="i2",
        woe_teacher_dropout="keep",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _model_with_teacher(**overrides) -> WoeSiNet:
    """A WoE-SI net that has taken one teacher snapshot, as after task 0."""
    torch.manual_seed(0)
    model = WoeSiNet(2, 6, 2, _woe_args(**overrides))
    model.current_task = 0
    model._snapshot_teacher()
    return model


def _teacher_logits(model: WoeSiNet, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = model.teacher.forward_features(x, bn_training=True)
        return model.teacher.forward_classifier(features, bn_training=True)


def _batch(n: int = 8) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randn(n, 2, 1024)


# ----------------------------------------------------------------------
# The defect, and the fix
# ----------------------------------------------------------------------
def test_default_teacher_is_stochastic() -> None:
    """Two forwards of identical frozen weights on identical input disagree."""
    model = _model_with_teacher(woe_teacher_dropout="keep")
    x = _batch()
    torch.manual_seed(1)
    first = _teacher_logits(model, x)
    torch.manual_seed(2)
    second = _teacher_logits(model, x)
    assert not torch.allclose(first, second)


def test_disable_makes_the_teacher_deterministic() -> None:
    """With dropout off the same forward is reproducible bit for bit."""
    model = _model_with_teacher(woe_teacher_dropout="disable")
    x = _batch()
    torch.manual_seed(1)
    first = _teacher_logits(model, x)
    torch.manual_seed(2)
    second = _teacher_logits(model, x)
    assert torch.equal(first, second)


def test_disable_zeroes_only_the_teachers_dropout() -> None:
    """The student keeps its own regularisation; only the copy is changed."""
    model = _model_with_teacher(woe_teacher_dropout="disable")
    teacher_p = [
        m.p
        for m in model.teacher.modules()
        if isinstance(m, nn.modules.dropout._DropoutNd)
    ]
    student_p = [
        m.p for m in model.net.modules() if isinstance(m, nn.modules.dropout._DropoutNd)
    ]
    assert teacher_p, "backbone has no dropout modules; the test proves nothing"
    assert all(p == 0.0 for p in teacher_p)
    assert all(p > 0.0 for p in student_p)


# ----------------------------------------------------------------------
# What must NOT change
# ----------------------------------------------------------------------
def test_batch_statistic_normalisation_survives() -> None:
    """`bn_training=True` still normalises with the batch, not the buffers.

    This is the property readme B4 measured at 0.068 of final macro recall, and
    it shares a switch with dropout, so it is the thing most at risk from this
    change. A deterministic teacher scored on two batches with deliberately
    different statistics must still give different logits; if the fix had
    silently frozen the running statistics instead, the shift would be absorbed.
    """
    model = _model_with_teacher(woe_teacher_dropout="disable")
    x = _batch()
    shifted = x * 5.0 + 3.0
    assert not torch.allclose(
        _teacher_logits(model, x), _teacher_logits(model, shifted)
    )


def test_buffers_stay_frozen_under_the_fix() -> None:
    """Zeroed BatchNorm momentum still keeps the snapshot's buffers put."""
    model = _model_with_teacher(woe_teacher_dropout="disable")
    before = [
        b.clone()
        for m in model.teacher.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
        for b in (m.running_mean, m.running_var)
    ]
    _teacher_logits(model, _batch() * 5.0 + 3.0)
    after = [
        b
        for m in model.teacher.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
        for b in (m.running_mean, m.running_var)
    ]
    assert before, "backbone has no BatchNorm; the test proves nothing"
    assert all(torch.equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize("mode", ["keep", "disable"])
def test_no_teacher_means_no_distillation(mode: str) -> None:
    """The flag cannot change task 0, where the penalty is exactly 0."""
    torch.manual_seed(0)
    model = WoeSiNet(2, 6, 2, _woe_args(woe_teacher_dropout=mode))
    assert model.teacher is None
    logits = torch.randn(4, 6)
    loss = model._lwf_distillation_loss(logits, _batch(4), t=0)
    assert float(loss.sum().item()) == 0.0


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="woe_teacher_dropout"):
        WoeSiNet(2, 6, 2, _woe_args(woe_teacher_dropout="off"))
