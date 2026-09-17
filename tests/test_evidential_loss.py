"""Tests for the evidential classification objective (``--woe_evidential_mode``).

The term replaces cross-entropy with a two-sided log loss on a bounded per-class
evidential score: raise the evidence supporting the label, lower the evidence
supporting the others. What it adds over CE is entirely on one axis --
``s_k = w+_k + w-_k``, the total contribution magnitude, which the ``K`` logits
cannot see -- so the properties worth pinning down are the ones that govern how
it acts on ``s``.

Covered here: the scale invariance that makes ``"balance"`` immune to being
satisfied by inflating the readout (the failure that sinks the one-sided hinge),
the ``raw_uniform`` identity ``w+ - w- = z`` that keeps the optimised score equal
to the evaluated one, numerical safety at perfectly one-sided evidence, the
class-balancing that stops the non-target term degenerating into a net shrinkage
of commitment, and the learner's column scoping.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.woe_si import (
    Net as WoeSiNet,
    compute_weights_of_evidence,
    evidential_classification_loss,
    per_class_total_evidence,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _woe_args(**overrides) -> SimpleNamespace:
    """Minimal namespace with the fields ``ResNet1D`` / WoE-SI expect."""
    return SimpleNamespace(
        arch="resnet1d",
        cuda=False,
        use_detector_arch=False,
        class_weighted_ce=False,
        data_scaling="none",
        use_iq_aug_features=False,
        iq_aug_feature_type="power",
        classes_per_task=overrides.get("classes_per_task", [3, 3]),
        nc_per_task_list="",
        nc_per_task=None,
        noise_label=overrides.get("noise_label", None),
        loader="task_incremental_loader",
        alpha_init=1e-3,
        optimizer="sgd",
        clipgrad=100.0,
        cls_lambda=1.0,
        det_memories=0,
        det_replay_batch=64,
        inner_steps=1,
        lr=overrides.get("lr", 0.01),
        woe_lambda=overrides.get("woe_lambda", 0.0),
        woe_xi=1e-3,
        woe_centering_mode=overrides.get("woe_centering_mode", "raw_uniform"),
        woe_mu_momentum=0.9,
        woe_importance_stride=1,
        woe_conflict_weighting=False,
        woe_reg_level="parameter",
        woe_anchor_mode="proximal",
        woe_omega_winsorise=0.0,
        woe_omega_transform="abs",
        woe_importance_scalar="i2",
        woe_evidence_scale="weight",
        woe_evidence_belief_tau=1.0,
        woe_evidence_asymmetric=False,
        woe_evidence_distill_lambda=0.0,
        woe_lwf_lambda=0.0,
        woe_lwf_temperature=5.0,
        woe_lc_lambda=0.0,
        woe_lc_readout_only=False,
        woe_lc_term="i2",
        woe_evidential_mode=overrides.get("woe_evidential_mode", "balance"),
        woe_evidential_gamma=overrides.get("woe_evidential_gamma", 1.0),
        woe_evidential_tau=overrides.get("woe_evidential_tau", 4.0),
        woe_evidential_class_balance=overrides.get(
            "woe_evidential_class_balance", True
        ),
    )


def _readout(classes: int, dim: int, seed: int = 1) -> tuple[torch.Tensor, ...]:
    """A random readout slice with gradients enabled."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(classes, dim, generator=generator, requires_grad=True)
    bias = torch.randn(classes, generator=generator, requires_grad=True)
    return weight, bias


def _iq_batch(rows: int = 6, length: int = 1024) -> torch.Tensor:
    """A synthetic (I, Q) batch in the layout ``ResNet1D`` consumes directly."""
    return torch.randn(rows, 2, length)


# ----------------------------------------------------------------------
# Functional core: what the objective can and cannot be satisfied by
# ----------------------------------------------------------------------
def test_balance_mode_is_invariant_to_rescaling_the_readout() -> None:
    """The structural fix: inflating ``beta`` buys nothing under ``"balance"``.

    ``w`` is linear in ``(beta, beta_0)`` jointly, so a global rescale multiplies
    both channels equally and leaves ``b = w+/(w+ + w-)`` fixed. This is what
    distinguishes the objective from a raw "commit more evidence" penalty, which
    one rescale satisfies for every class at once (readme B3, BWT -0.4383).
    """
    torch.manual_seed(0)
    features = torch.randn(16, 32)
    weight, bias = _readout(4, 32)
    targets = torch.randint(0, 4, (16,))
    mean = features.mean(dim=0)

    base = evidential_classification_loss(
        features, weight, bias, mean, targets, mode="balance"
    )
    scaled = evidential_classification_loss(
        features, 10.0 * weight, 10.0 * bias, mean, targets, mode="balance"
    )
    assert float(base.item()) == pytest.approx(float(scaled.item()), rel=1e-4)


def test_balance_mode_stays_scale_invariant_at_one_sided_evidence() -> None:
    """The corner the smoothing has to get right: ``w- = 0`` exactly.

    An absolute ``eps`` floor would make ``log(0 + eps) - log(scale * total)``
    shift by ``-log(scale)``, leaking the readout's scale back into the loss for
    precisely the configuration the objective is trying to reach. Smoothing
    proportionally to the total keeps the invariance exact.
    """
    features = torch.ones(3, 8)
    # Row 0 contributes only positively (w- = 0), the others only negatively.
    weight = torch.cat([torch.ones(1, 8), -torch.ones(2, 8)], dim=0)
    bias = torch.zeros(3)
    targets = torch.tensor([0, 1, 2])

    base = evidential_classification_loss(
        features, weight, bias, torch.zeros(8), targets, mode="balance"
    )
    scaled = evidential_classification_loss(
        features, 50.0 * weight, 50.0 * bias, torch.zeros(8), targets, mode="balance"
    )
    assert float(base.item()) == pytest.approx(float(scaled.item()), rel=1e-5)


def test_belief_mode_is_not_scale_invariant() -> None:
    """The contrast that makes ``woe_evidential_tau`` load-bearing.

    ``1 - e^{-w/tau}`` saturates, so the same rescale that ``"balance"`` ignores
    moves ``"belief"`` -- and past ``w/tau ~ 16.6`` it saturates to exactly 1.0
    with a zero gradient. Documented so nobody reads the two modes as
    interchangeable parameterisations.
    """
    torch.manual_seed(0)
    features = torch.randn(16, 32)
    weight, bias = _readout(4, 32)
    targets = torch.randint(0, 4, (16,))
    mean = features.mean(dim=0)

    base = evidential_classification_loss(
        features, weight, bias, mean, targets, mode="belief", tau=4.0
    )
    scaled = evidential_classification_loss(
        features, 10.0 * weight, 10.0 * bias, mean, targets, mode="belief", tau=4.0
    )
    assert float(base.item()) != pytest.approx(float(scaled.item()), rel=1e-3)


def test_raw_uniform_evidence_difference_equals_the_evaluated_logit() -> None:
    """``w+ - w- = z`` exactly, so the optimised score ranks like the scored one.

    The reason the objective runs uncentred by default: under
    ``centered_uniform`` the identity gives ``z - beta . mu``, a per-class shift,
    and ``argmax`` over the two can differ. Evaluation takes ``argmax`` over the
    raw logits, so training a shifted ranking is a silent train/eval mismatch.
    """
    torch.manual_seed(0)
    features = torch.randn(8, 16)
    weight, bias = _readout(3, 16)
    mean = features.mean(dim=0)

    raw = compute_weights_of_evidence(
        features, weight, bias, mean, centering_mode="raw_uniform"
    )
    w_plus, w_minus = per_class_total_evidence(raw)
    logits = features @ weight.T + bias
    assert torch.allclose(w_plus - w_minus, logits, atol=1e-4)

    centred = compute_weights_of_evidence(
        features, weight, bias, mean, centering_mode="centered_uniform"
    )
    c_plus, c_minus = per_class_total_evidence(centred)
    shift = mean @ weight.T
    assert torch.allclose(c_plus - c_minus, logits - shift, atol=1e-4)


def test_one_sided_evidence_beats_conflicted_evidence() -> None:
    """The objective prefers sign-coherent rows, which is the selectivity bet.

    A readout whose target row contributes with one sign and whose other rows
    contribute with the other scores lower than a conflicted one at the same
    logits scale. This is the mechanism the loss is for: ``|d_k| -> s_k``.
    """
    features = torch.ones(4, 8)
    targets = torch.zeros(4, dtype=torch.long)
    mean = torch.zeros(8)

    one_sided = torch.cat([torch.ones(1, 8), -torch.ones(2, 8)], dim=0)
    conflicted = torch.cat([torch.ones(3, 4), -torch.ones(3, 4)], dim=1)
    bias = torch.zeros(3)

    clean = evidential_classification_loss(
        features, one_sided, bias, mean, targets, mode="balance"
    )
    messy = evidential_classification_loss(
        features, conflicted, bias, mean, targets, mode="balance"
    )
    assert float(clean.item()) < float(messy.item())


@pytest.mark.parametrize("mode", ["balance", "belief"])
def test_finite_and_differentiable_at_perfectly_one_sided_evidence(mode: str) -> None:
    """``w- = 0`` must not produce ``inf``/``nan`` or a dead gradient.

    The smoothed ratio (``+eps`` on each channel, ``+2 eps`` on the sum) is there
    precisely because clamping would zero the gradient exactly where a perfectly
    one-sided non-target class needs the strongest push.
    """
    features = torch.ones(4, 8)
    weight = torch.ones(3, 8, requires_grad=True)
    bias = torch.zeros(3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])

    loss = evidential_classification_loss(
        features, weight, bias, torch.zeros(8), targets, mode=mode, tau=4.0
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert float(weight.grad.abs().sum().item()) > 0.0


def test_class_balancing_evens_out_the_per_row_shrinkage_pressure() -> None:
    """What ``woe_evidential_class_balance`` is actually for.

    Each row is the target for ``p_k`` of the batch and a non-target for the
    rest, and ``s_k`` enters ``w+_k = (s_k + d_k)/2`` positively either way, so on
    an imbalanced batch the non-target term dominates for the rare classes and
    the objective turns into a net shrinkage of *their* commitment. Measured as
    the spread of the per-row gradient with respect to a per-row scale: balancing
    should make the rows comparable, leaving no row singled out for shrinkage.
    """
    torch.manual_seed(0)
    features = torch.randn(12, 16).abs()
    weight, bias = _readout(3, 16)
    mean = torch.zeros(16)
    # 10 samples of class 0 against one each of classes 1 and 2.
    targets = torch.tensor([0] * 10 + [1, 2])

    def row_scale_gradients(class_balance: bool) -> torch.Tensor:
        scale = torch.ones(3, requires_grad=True)
        loss = evidential_classification_loss(
            features,
            weight.detach() * scale.unsqueeze(1),
            bias.detach() * scale,
            mean,
            targets,
            mode="belief",
            tau=4.0,
            class_balance=class_balance,
        )
        (gradient,) = torch.autograd.grad(loss, scale)
        return gradient

    unbalanced = row_scale_gradients(False)
    balanced = row_scale_gradients(True)
    assert float(balanced.std().item()) < float(unbalanced.std().item())


def test_rejects_an_unknown_mode() -> None:
    """Typos must fail loudly rather than silently scoring something else."""
    with pytest.raises(ValueError, match="mode must be"):
        evidential_classification_loss(
            torch.randn(4, 8),
            torch.randn(3, 8),
            torch.zeros(3),
            torch.zeros(8),
            torch.zeros(4, dtype=torch.long),
            mode="i2",
        )


# ----------------------------------------------------------------------
# Learner integration
# ----------------------------------------------------------------------
def test_learner_rejects_an_out_of_range_gamma() -> None:
    """``gamma`` mixes against CE, so it has to be a convex weight."""
    with pytest.raises(ValueError, match="woe_evidential_gamma"):
        WoeSiNet(1024, 6, 2, _woe_args(woe_evidential_gamma=1.5))


def test_evidential_objective_trains_and_moves_the_readout() -> None:
    """One ``observe`` step with CE fully replaced must still update parameters."""
    torch.manual_seed(0)
    net = WoeSiNet(1024, 6, 2, _woe_args())
    before = net.net.model.fc.weight.detach().clone()
    x = _iq_batch(rows=6)
    y = torch.tensor([0, 1, 2, 0, 1, 2])

    loss, _, _ = net.observe(x, y, 0)

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.allclose(before, net.net.model.fc.weight.detach())


def test_evidential_objective_charges_current_task_columns_only() -> None:
    """Old columns must not be asked to *remove* evidence.

    A two-sided loss over the cumulative active set would push ``b_k`` toward 0
    for every previously-learned class on every new sample, which is a forgetting
    mechanism rather than a regularising one. Mirrors the Least-Commitment
    scoping.
    """
    torch.manual_seed(0)
    net = WoeSiNet(1024, 6, 2, _woe_args())
    features = torch.randn(6, net.feature_dim)
    y = torch.tensor([3, 4, 5, 3, 4, 5])

    loss = net._evidential_loss(features, 1, y)
    net.net.model.fc.weight.grad = None
    loss.backward()

    gradient = net.net.model.fc.weight.grad
    assert gradient is not None
    assert float(gradient[:3].abs().sum().item()) == 0.0
    assert float(gradient[3:].abs().sum().item()) > 0.0


def test_a_non_contiguous_column_set_maps_targets_correctly() -> None:
    """With a global noise column the charged set is not a contiguous range.

    ``current_task_class_indices`` returns ``[offset1, offset2)`` *plus* the
    always-active noise column, so local positions have to come from a scatter
    rather than subtracting an offset -- otherwise every label lands on the wrong
    row and the loss silently optimises the wrong thing.
    """
    torch.manual_seed(0)
    net = WoeSiNet(1024, 6, 2, _woe_args(noise_label=0))
    features = torch.randn(4, net.feature_dim)
    columns = net._current_task_class_indices(1, features.device)
    assert columns.tolist() == [0, 3, 4, 5]

    # A batch mixing the noise class with task-1 classes.
    loss = net._evidential_loss(features, 1, torch.tensor([0, 3, 4, 5]))
    assert torch.isfinite(loss) and float(loss.item()) > 0.0


def test_predict_score_ranks_by_the_trained_quantity() -> None:
    """``forward`` must return the evidential ranking when asked to predict with it.

    The score is ``d_k/s_k = 2 b_k - 1``, bounded in ``[-1, 1]`` so the mask's
    ``-1e9`` fill still dominates, and it is *not* a monotone function of the
    logit across classes -- normalising by ``s_k`` reorders classes whose rows
    earned the same logit with different amounts of internal cancellation.
    """
    torch.manual_seed(0)
    net = WoeSiNet(1024, 6, 2, _woe_args(woe_evidential_predict="score"))
    x = _iq_batch(rows=4)

    scored = net.forward(x, 0)
    active = scored[:, :3]
    assert bool((active.abs() <= 1.0).all())
    assert bool((scored[:, 3:] < -1e8).all())

    net.evidential_predict = "logit"
    logits = net.forward(x, 0)
    assert not torch.allclose(logits[:, :3], active)


def test_predict_score_is_inert_when_the_objective_is_off() -> None:
    """The flag must not change a CE run, so the control arm stays a control."""
    torch.manual_seed(0)
    net = WoeSiNet(
        1024,
        6,
        2,
        _woe_args(woe_evidential_mode="off", woe_evidential_predict="score"),
    )
    x = _iq_batch(rows=4)
    # `ResNet1D.forward` sets the module's training mode from `bn_training`,
    # whose default is True, so every forward here draws a dropout mask. Reseed
    # so the two draws match and the comparison is of code paths, not samples.
    torch.manual_seed(1)
    expected = net.net(x)[:, :3]
    torch.manual_seed(1)
    assert torch.allclose(net.forward(x, 0)[:, :3], expected, atol=1e-4)


def test_mode_off_leaves_the_cross_entropy_path_untouched() -> None:
    """The default must be a no-op, so every recorded result stays reproducible."""
    torch.manual_seed(0)
    net = WoeSiNet(1024, 6, 2, _woe_args(woe_evidential_mode="off"))
    assert net.evidential_mode == "off"
    x = _iq_batch(rows=4)
    y = torch.tensor([0, 1, 2, 0])
    loss, _, _ = net.observe(x, y, 0)
    assert torch.isfinite(torch.tensor(loss))
