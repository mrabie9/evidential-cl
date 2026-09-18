"""Tests for the Denoeux ``I_p`` family (``--woe_lc_p``, ``woe_importance_scalar=i1``).

Denoeux fixes ``p = 2`` in Eq 10 for tractability and says so; the exponent is a
free parameter of the family. ``p = 1`` is not a milder ``p = 2`` -- since
``w+_k + w-_k = sum_j |w_jk|``, ``I_1`` is the L1 norm of the whole
weight-of-evidence matrix, so minimising it is a **sparsity** criterion on the
evidence (few features carry support, the rest go vacuous) where ``p = 2``
spreads it. That is what makes ``p`` a bridge between the regularisation and
architectural (PackNet/HAT masking) families rather than another knob.

Covered here: the exact ``logit + conflict`` decomposition at *both* exponents
(the L1 identity ``a + b = |a - b| + 2 min(a, b)`` is the non-obvious half), the
``J^p`` normalisation, that ``p = 1`` really does concentrate evidence where
``p = 2`` spreads it -- the claim the bridge rests on -- and that the exponent of
the objective and the exponent of the tracked importance scalar are independent
axes, which E1 makes a correctness requirement rather than a style preference.
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
    _least_commitment_terms,
    compute_weights_of_evidence,
    information_content,
    least_commitment_penalty,
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
        woe_omega_transform="abs",
        woe_omega_accum="sum",
        woe_importance_scalar=overrides.get("woe_importance_scalar", "i2"),
        woe_evidence_scale="weight",
        woe_evidence_belief_tau=1.0,
        woe_evidence_asymmetric=False,
        woe_evidence_distill_lambda=0.0,
        woe_lwf_lambda=0.0,
        woe_lwf_temperature=5.0,
        woe_lc_lambda=overrides.get("woe_lc_lambda", 0.0),
        woe_lc_readout_only=False,
        woe_lc_term=overrides.get("woe_lc_term", "i2"),
        woe_lc_p=overrides.get("woe_lc_p", 2),
    )


def _random_evidence(batch: int = 6, classes: int = 4, features: int = 9):
    torch.manual_seed(0)
    phi = torch.randn(batch, features)
    weight = torch.randn(classes, features)
    bias = torch.randn(classes)
    mean = torch.randn(features)
    return phi, weight, bias, mean


# ----------------------------------------------------------------------
# The family, and its decomposition
# ----------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 2])
def test_decomposition_is_exact_at_both_exponents(p: int) -> None:
    """``logit + conflict == I_p`` to floating point, for p = 1 and p = 2.

    At ``p = 2`` this is the familiar ``||z'||^2 + 2 sum_k w+ w-``. At ``p = 1``
    it relies on ``a + b = |a - b| + 2 min(a, b)`` for non-negative ``a, b``,
    which is what lets the ``woe_lc_term`` ablation carry over unchanged.
    """
    phi, weight, bias, mean = _random_evidence()
    weights = compute_weights_of_evidence(phi, weight, bias, mean)
    terms = _least_commitment_terms(weights, p=p)
    assert torch.allclose(terms["logit"] + terms["conflict"], terms["i2"], atol=1e-5)


@pytest.mark.parametrize("p", [1, 2])
def test_information_content_is_nonnegative_and_vacuous_at_zero(p: int) -> None:
    """``I_p >= 0`` always, and exactly 0 for a vacuous readout."""
    phi, weight, bias, mean = _random_evidence()
    weights = compute_weights_of_evidence(phi, weight, bias, mean)
    assert (information_content(weights, p=p) >= 0).all()

    vacuous = compute_weights_of_evidence(
        phi, torch.zeros_like(weight), torch.zeros_like(bias), mean
    )
    assert torch.allclose(
        information_content(vacuous, p=p), torch.zeros(phi.shape[0]), atol=1e-6
    )


def test_i1_equals_the_l1_norm_of_the_weights_of_evidence() -> None:
    """``I_1 = sum_jk |w_jk|`` -- the identity the sparsity reading rests on."""
    phi, weight, bias, mean = _random_evidence()
    weights = compute_weights_of_evidence(phi, weight, bias, mean)
    assert torch.allclose(
        information_content(weights, p=1), weights.abs().sum(dim=(1, 2)), atol=1e-5
    )


@pytest.mark.parametrize("p", [1, 2])
def test_penalty_is_normalised_by_j_to_the_p(p: int) -> None:
    """The penalty is the ``J^p``-divided batch mean of the raw ``I_p``."""
    phi, weight, bias, mean = _random_evidence()
    weights = compute_weights_of_evidence(phi, weight, bias, mean)
    expected = information_content(weights, p=p).mean() / float(phi.shape[1] ** p)
    actual = least_commitment_penalty(phi, weight, bias, mean, p=p)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_penalty_scales_as_the_p_th_power_of_the_readout() -> None:
    """Doubling the readout doubles ``I_1`` and quadruples ``I_2``.

    Both channel totals are homogeneous of degree 1 in ``(beta, beta_0)``, so
    ``I_p`` is homogeneous of degree ``p``. This is why lambda cannot transfer
    between exponents even after the ``J^p`` division.
    """
    phi, weight, bias, mean = _random_evidence()
    for p, factor in ((1, 2.0), (2, 4.0)):
        base = least_commitment_penalty(phi, weight, bias, mean, p=p)
        doubled = least_commitment_penalty(phi, 2.0 * weight, 2.0 * bias, mean, p=p)
        assert torch.allclose(doubled, factor * base, rtol=1e-4)


# ----------------------------------------------------------------------
# The claim: p = 1 concentrates evidence, p = 2 spreads it
# ----------------------------------------------------------------------
def test_p1_drives_features_to_vacuity_where_p2_saturates() -> None:
    """``I_1`` sparsifies the evidence; ``I_2`` levels it and then stops.

    The claim is about ``w_jk`` (the evidence), so that is what is measured --
    not the readout weights -- and it is measured **with a fit term present**,
    because both objectives are minimised outright by a vacuous readout and
    neither says anything on its own. Alongside cross-entropy, at strengths that
    both leave the task perfectly separable, ``p = 1`` drives roughly half the
    features below 5% of the peak while ``p = 2`` plateaus around a fifth no
    matter how hard it is pushed. That gap is the sparsity criterion the bridge
    to the architectural family (PackNet/HAT masking) rests on.

    Note the constraint has to be left off: normalising to a fixed evidence
    budget makes the ``p = 1`` objective its own constraint -- degenerate, with
    nothing left to optimise -- while ``p = 2`` can still rearrange under it, and
    the comparison then measures the normalisation rather than the exponents.
    """

    def vacuous_fraction(phi, weight, bias, mean) -> float:
        """Share of features whose mean |evidence| is under 5% of the peak."""
        evidence = compute_weights_of_evidence(phi, weight, bias, mean).abs()
        per_feature = evidence.mean(dim=(0, 1))
        return float((per_feature < 0.05 * per_feature.max()).float().mean())

    torch.manual_seed(0)
    batch, features, classes = 256, 32, 4
    labels = torch.randint(0, classes, (batch,))
    # A genuine class signal, so cross-entropy has something to hold onto and the
    # penalty is trading against a real fit rather than acting unopposed.
    phi = torch.randn(batch, features) + torch.randn(classes, features)[labels] * 1.5
    mean = phi.mean(dim=0)

    fractions = {}
    for p, lc_lambda in ((1, 1.0), (2, 3.0)):
        torch.manual_seed(0)
        weight = (torch.randn(classes, features) * 0.3).requires_grad_(True)
        bias = torch.zeros(classes, requires_grad=True)
        optimiser = torch.optim.Adam([weight, bias], lr=0.02)
        for _ in range(400):
            optimiser.zero_grad()
            logits = phi @ weight.T + bias
            loss = torch.nn.functional.cross_entropy(
                logits, labels
            ) + lc_lambda * least_commitment_penalty(phi, weight, bias, mean, p=p)
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            accuracy = float(
                ((phi @ weight.T + bias).argmax(dim=1) == labels).float().mean()
            )
        # Both arms must still fit, or the comparison is between a working model
        # and a collapsed one rather than between two evidence geometries.
        assert accuracy > 0.95, f"p={p} collapsed the fit (accuracy {accuracy:.3f})"
        fractions[p] = vacuous_fraction(phi, weight.detach(), bias.detach(), mean)

    assert fractions[1] > 1.5 * fractions[2], (
        "p=1 should leave far more features vacuous than p=2 "
        f"({fractions[1]:.3f} vs {fractions[2]:.3f})"
    )


# ----------------------------------------------------------------------
# The two exponents are independent axes
# ----------------------------------------------------------------------
def test_objective_exponent_does_not_touch_the_tracked_scalar() -> None:
    """``woe_lc_p`` changes the objective and leaves the path integral alone.

    E1 found that an objective moving ``I_2`` moves ``Omega`` with it, so the
    codebase deliberately keeps "what is charged" and "what is measured" on
    separate flags. This asserts the wiring honours that.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 2, 128)
    for lc_p in (1, 2):
        model = WoeSiNet(2, 6, 2, _woe_args(woe_lc_p=lc_p, woe_lc_lambda=1.0))
        assert model.lc_p == lc_p
        # The tracked scalar stays at the method's own p=2 regardless.
        assert model.importance_scalar == "i2"
        features = model.net.forward_features(x, bn_training=False)
        active = model._active_class_indices(0, features.device)
        tracked = model._least_commitment_scalar(features, active)
        reference = model._least_commitment_scalar(features, active, p=2)
        assert torch.allclose(tracked, reference)


def test_i1_importance_scalar_tracks_the_p1_content() -> None:
    """``woe_importance_scalar='i1'`` measures ``I_1``, not ``I_2``."""
    torch.manual_seed(0)
    x = torch.randn(4, 2, 128)
    model = WoeSiNet(2, 6, 2, _woe_args(woe_importance_scalar="i1"))
    tracked = model._compute_information_content(x, 0, update_feature_mean=True)

    features = model.net.forward_features(x, bn_training=False)
    active = model._active_class_indices(0, features.device)
    assert torch.allclose(
        tracked, model._least_commitment_scalar(features, active, p=1), atol=1e-6
    )
    assert not torch.allclose(
        tracked, model._least_commitment_scalar(features, active, p=2), atol=1e-6
    )


def test_i1_importance_scalar_produces_finite_positive_omega() -> None:
    """End to end: the p=1 scalar drives a usable path integral."""
    torch.manual_seed(1)
    model = WoeSiNet(2, 6, 2, _woe_args(woe_importance_scalar="i1"))
    x = torch.randn(8, 2, 128)
    for task, y in enumerate((torch.randint(0, 3, (8,)), torch.randint(3, 6, (8,)))):
        for _ in range(3):
            model.observe(x, y, task)
    model.on_task_end()
    total = sum(
        float(getattr(model, f"{model._param_to_key[name]}_woe_omega").sum().item())
        for name in model._tracked_names
    )
    assert total > 0.0 and torch.isfinite(torch.tensor(total))


def test_invalid_exponent_raises() -> None:
    phi, weight, bias, mean = _random_evidence()
    with pytest.raises(ValueError, match="p must be one of"):
        least_commitment_penalty(phi, weight, bias, mean, p=3)
    with pytest.raises(ValueError, match="woe_lc_p"):
        WoeSiNet(2, 6, 2, _woe_args(woe_lc_p=3))


def test_conflict_half_at_p1_is_twice_the_channel_minimum() -> None:
    """The DS-specific half at p=1 is ``2 sum_k min(w+, w-)``, by construction."""
    phi, weight, bias, mean = _random_evidence()
    weights = compute_weights_of_evidence(phi, weight, bias, mean)
    w_plus, w_minus = per_class_total_evidence(weights)
    expected = 2.0 * torch.minimum(w_plus, w_minus).sum(dim=1)
    assert torch.allclose(_least_commitment_terms(weights, p=1)["conflict"], expected)
