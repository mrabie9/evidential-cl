"""Tests for the Least-Commitment objective (``--woe_lc_lambda``).

The term minimises the Dempster-Shafer information content ``I_2(m)`` of the
readout's mass function on the current task, alongside cross-entropy: commit no
more evidence than the data requires, so evidential room is left for later
tasks. It is shared by ``model.woe_si`` (where ``I_2`` is otherwise *measured*,
as the SI path integral's tracked scalar) and ``model.eralg4``.

Covered here: the algebra the term is justified by (it decomposes into a
centred-logit norm plus a per-class conflict penalty), its exact quadratic
scaling in the readout, the class/row scoping in both learners, and that turning
it on actually reduces commitment.
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

from model.eralg4 import Net as ErAlg4Net
from model.woe_si import (
    Net as WoeSiNet,
    compute_weights_of_evidence,
    evidence_to_belief,
    information_content,
    least_commitment_penalty,
    per_class_total_evidence,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _woe_args(**overrides) -> object:
    """Minimal namespace with the fields ``ResNet1D`` / WoE-SI expect."""
    args = SimpleNamespace(
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
        noise_label=None,
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
        woe_centering_mode="centered_uniform",
        woe_mu_momentum=0.9,
        woe_importance_stride=1,
        woe_conflict_weighting=False,
        woe_reg_level="parameter",
        woe_anchor_mode=overrides.get("woe_anchor_mode", "proximal"),
        woe_omega_winsorise=0.0,
        woe_omega_transform="relu",
        woe_importance_scalar="i2",
        woe_evidence_scale="weight",
        woe_evidence_belief_tau=1.0,
        woe_evidence_asymmetric=False,
        woe_evidence_distill_lambda=0.0,
        woe_lwf_lambda=0.0,
        woe_lwf_temperature=5.0,
        woe_lc_lambda=overrides.get("woe_lc_lambda", 0.0),
        woe_lc_readout_only=overrides.get("woe_lc_readout_only", False),
        woe_lc_term=overrides.get("woe_lc_term", "i2"),
        woe_lc_tau=overrides.get("woe_lc_tau", 4.0),
    )
    return args


def _eralg4_args(**overrides) -> SimpleNamespace:
    """Minimal namespace for ``model.eralg4.Net`` on the plain ER path."""
    return SimpleNamespace(
        arch="resnet1d",
        cuda=False,
        dataset="iq",
        use_detector_arch=False,
        class_weighted_ce=False,
        data_scaling="none",
        use_iq_aug_features=False,
        iq_aug_feature_type="power",
        classes_per_task=[3, 3],
        nc_per_task=None,
        noise_label=None,
        loader="task_incremental_loader",
        alpha_init=1e-3,
        lr=0.01,
        opt_lr=1e-1,
        learn_lr=overrides.get("learn_lr", False),
        inner_steps=1,
        memories=32,
        replay_batch_size=4,
        grad_clip_norm=2.0,
        second_order=False,
        meta_batches=2,
        det_lambda=1.0,
        cls_lambda=1.0,
        det_memories=0,
        det_replay_batch=8,
        memory_loss_lambda=1.0,
        woe_lc_lambda=overrides.get("woe_lc_lambda", 0.0),
        woe_lc_readout_only=overrides.get("readout_only", False),
        woe_lc_term=overrides.get("term", "i2"),
        woe_centering_mode="centered_uniform",
        woe_mu_momentum=0.9,
    )


def _iq_batch(rows: int = 6, length: int = 1024) -> torch.Tensor:
    """A synthetic (I, Q) batch in the layout ``ResNet1D`` consumes directly."""
    return torch.randn(rows, 2, length)


# ----------------------------------------------------------------------
# Functional core
# ----------------------------------------------------------------------
def test_penalty_is_zero_for_a_vacuous_readout() -> None:
    """A zero readout commits nothing, which is the objective's minimiser."""
    torch.manual_seed(0)
    features = torch.randn(8, 16)
    weight = torch.zeros(4, 16)
    bias = torch.zeros(4)
    penalty = least_commitment_penalty(features, weight, bias, features.mean(dim=0))
    assert float(penalty.item()) == 0.0


def test_penalty_scales_quadratically_with_the_readout() -> None:
    """``w`` is linear in ``(beta, beta_0)`` jointly, so ``I_2`` is exactly quadratic.

    This is what makes the term a commitment penalty rather than a fit penalty:
    halving the readout quarters it regardless of what the features look like.
    """
    torch.manual_seed(1)
    features = torch.randn(8, 16)
    weight = torch.randn(4, 16)
    bias = torch.randn(4)
    mean = features.mean(dim=0)

    full = least_commitment_penalty(features, weight, bias, mean)
    half = least_commitment_penalty(features, 0.5 * weight, 0.5 * bias, mean)
    assert float(half.item()) == pytest.approx(0.25 * float(full.item()), rel=1e-5)


def test_information_content_splits_into_logit_norm_and_conflict() -> None:
    """``I_2 = ||z'||^2 + 2 * sum_k w+_k w-_k``, the decomposition the term rests on.

    ``w+_k - w-_k = sum_j w_jk = z'_k`` is the centred logit, so minimising
    ``I_2`` is a confidence penalty *plus* a penalty on per-class conflict. The
    second half is the part the DS reading contributes over plain logit decay.
    """
    torch.manual_seed(2)
    features = torch.randn(5, 12)
    weight = torch.randn(3, 12)
    bias = torch.randn(3)
    mean = features.mean(dim=0)

    weights = compute_weights_of_evidence(features, weight, bias, mean)
    w_plus, w_minus = per_class_total_evidence(weights)
    centred_logits = w_plus - w_minus

    identity = centred_logits.pow(2).sum(dim=1) + 2.0 * (w_plus * w_minus).sum(dim=1)
    assert torch.allclose(information_content(weights), identity, atol=1e-5)


def test_the_two_halves_sum_to_the_whole() -> None:
    """``logit + conflict == i2`` exactly, so the split loses nothing."""
    torch.manual_seed(11)
    features = torch.randn(8, 16)
    weight = torch.randn(4, 16)
    bias = torch.randn(4)
    mean = features.mean(dim=0)

    parts = {
        term: least_commitment_penalty(features, weight, bias, mean, term=term)
        for term in ("i2", "logit", "conflict")
    }
    assert float(parts["logit"] + parts["conflict"]) == pytest.approx(
        float(parts["i2"]), rel=1e-6
    )
    # AM-GM bounds the conflict half by the *whole*, not by the other half:
    # w+^2 + w-^2 >= 2 w+ w-. The two halves are not ordered -- balanced
    # channels (w+ ~ w-) send `logit` to 0 and `conflict` to nearly all of I_2,
    # while one-sided evidence does the reverse. That is why lambda cannot
    # transfer between the terms, and why they are different objectives rather
    # than rescalings of one.
    assert 0.0 <= float(parts["conflict"]) <= float(parts["i2"]) + 1e-6
    assert 0.0 <= float(parts["logit"]) <= float(parts["i2"]) + 1e-6


def test_conflict_term_ignores_a_one_sided_readout() -> None:
    """Conflict is zero when no class has both support and counter-support.

    A readout whose evidence is all one sign per class commits plenty (``i2`` is
    large) but conflicts with itself not at all, so the DS-specific half must
    read exactly 0 where the plain confidence penalty does not. This is the
    property that makes ``conflict`` a different objective rather than a rescaled
    one.
    """
    features = torch.ones(4, 6)
    mean = torch.zeros(6)
    weight = torch.ones(2, 6)  # every w_jk > 0 -> w_minus == 0 for both classes
    bias = torch.zeros(2)

    assert (
        float(least_commitment_penalty(features, weight, bias, mean, term="conflict"))
        == 0.0
    )
    assert (
        float(least_commitment_penalty(features, weight, bias, mean, term="logit"))
        > 0.0
    )


def test_unknown_term_is_rejected() -> None:
    """A typo in the ablation flag must fail loudly, not silently charge I_2."""
    features = torch.randn(3, 5)
    with pytest.raises(ValueError, match="term must be one of"):
        least_commitment_penalty(
            features,
            torch.randn(2, 5),
            torch.randn(2),
            features.mean(dim=0),
            term="conflictt",
        )


# ----------------------------------------------------------------------
# WoE-SI wiring
# ----------------------------------------------------------------------
def test_woe_si_penalty_charges_only_current_task_columns() -> None:
    """Prior-task readout rows must receive no gradient from the LC term.

    Charging them would ask the model to *un-commit* evidence it already holds
    on old classes, which is a forgetting mechanism rather than a regulariser.
    """
    torch.manual_seed(3)
    model = WoeSiNet(1024, 6, 2, _woe_args())
    features = model.net.forward_features(_iq_batch())
    penalty = model._least_commitment_loss(features, t=1)
    penalty.backward()

    grad = model.net.model.fc.weight.grad
    assert grad is not None
    assert float(grad[0:3].abs().sum().item()) == 0.0
    assert float(grad[3:6].abs().sum().item()) > 0.0


def test_woe_si_penalty_matches_the_tracked_importance_scalar() -> None:
    """The minimised and the measured ``I_2`` are one code path, not two.

    On task 0 the current-task columns *are* the active columns, so the LC term
    must equal the scalar the SI path integral tracks -- otherwise ``woe_lambda``
    and ``woe_lc_lambda`` would be quoted on silently different scales.
    """
    torch.manual_seed(4)
    model = WoeSiNet(1024, 6, 2, _woe_args())
    x = _iq_batch()
    with torch.no_grad():
        features = model.net.forward_features(x, bn_training=False)
        lc = model._least_commitment_loss(features, t=0)
        tracked = model._compute_information_content(x, t=0, update_feature_mean=False)
    assert float(lc.item()) == pytest.approx(float(tracked.item()), rel=1e-5)


def test_woe_si_readout_only_spares_the_backbone() -> None:
    """``woe_lc_readout_only`` must confine the term to the readout.

    ``I_2`` is minimised both by a vanishing readout and by ``phi -> mu``
    (feature collapse), and only the first is the intended objective. The flag
    exists to tell the two apart, so the backbone must receive exactly zero.
    """
    torch.manual_seed(9)
    model = WoeSiNet(1024, 6, 2, _woe_args(woe_lc_readout_only=True))
    features = model.net.forward_features(_iq_batch())
    model._least_commitment_loss(features, t=0).backward()

    assert model.net.model.fc.weight.grad.abs().sum().item() > 0.0
    backbone = [
        parameter.grad
        for name, parameter in model.net.model.named_parameters()
        if not name.startswith("fc.") and parameter.grad is not None
    ]
    assert all(float(grad.abs().sum().item()) == 0.0 for grad in backbone)


def test_eralg4_readout_only_spares_the_backbone() -> None:
    """The same isolation, on the learner that has no path integral at all."""
    torch.manual_seed(10)
    model = ErAlg4Net(1024, 6, 2, _eralg4_args(woe_lc_lambda=1.0, readout_only=True))
    features = model.net.forward_features(_iq_batch())
    model._least_commitment_loss(features, t=0).backward()

    assert model.net.model.fc.weight.grad.abs().sum().item() > 0.0
    backbone = [
        parameter.grad
        for name, parameter in model.net.model.named_parameters()
        if not name.startswith("fc.") and parameter.grad is not None
    ]
    assert all(float(grad.abs().sum().item()) == 0.0 for grad in backbone)


def test_woe_si_lc_reduces_commitment() -> None:
    """Training with the term on must leave the model less committed."""
    x = _iq_batch(rows=8)
    y = torch.randint(0, 3, (8,), dtype=torch.long)

    def train(lc_lambda: float) -> float:
        torch.manual_seed(5)
        model = WoeSiNet(1024, 6, 2, _woe_args(woe_lc_lambda=lc_lambda))
        for _ in range(5):
            model.observe(x, y, 0)
        with torch.no_grad():
            features = model.net.forward_features(x, bn_training=False)
            return float(model._least_commitment_loss(features, t=0).item())

    assert train(1e5) < train(0.0)


# ----------------------------------------------------------------------
# ER (eralg4) wiring
# ----------------------------------------------------------------------
def test_eralg4_penalty_matches_the_shared_function() -> None:
    """ER's penalty is the same number woe_si computes on the same features."""
    torch.manual_seed(6)
    model = ErAlg4Net(1024, 6, 2, _eralg4_args(woe_lc_lambda=1.0))
    features = torch.randn(4, model.net.feature_dim)
    penalty = model._least_commitment_loss(features, t=1)

    readout = model.net.model.fc
    expected = least_commitment_penalty(
        features,
        readout.weight[3:6],
        readout.bias[3:6],
        features.mean(dim=0),
    )
    assert float(penalty.item()) == pytest.approx(float(expected.item()), rel=1e-6)


def test_eralg4_observe_runs_with_the_penalty_on() -> None:
    """End-to-end smoke over the plain ER loop, including a replay-populated step."""
    torch.manual_seed(7)
    model = ErAlg4Net(1024, 6, 2, _eralg4_args(woe_lc_lambda=1e3))
    x = _iq_batch(rows=4)
    y = torch.randint(0, 3, (4,), dtype=torch.long)
    for _ in range(2):  # second call draws replay rows, exercising the row split
        loss, _recall, _logits = model.observe(x, y, 0)
    assert torch.isfinite(torch.tensor(loss))
    assert model.lc_feature_mean is not None


def test_eralg4_feature_mean_resets_at_a_task_boundary() -> None:
    """The centring reference is per-task; carrying it over measures the wrong origin."""
    torch.manual_seed(8)
    model = ErAlg4Net(1024, 6, 2, _eralg4_args(woe_lc_lambda=1e3))
    model.observe(_iq_batch(rows=4), torch.zeros(4, dtype=torch.long), 0)
    first = model.lc_feature_mean.clone()
    model.observe(_iq_batch(rows=4), torch.full((4,), 3, dtype=torch.long), 1)
    assert not torch.allclose(first, model.lc_feature_mean)


def test_eralg4_rejects_the_unimplemented_learn_lr_path() -> None:
    """Silently ignoring the term on the la_ER path would misreport the run."""
    with pytest.raises(ValueError, match="learn_lr"):
        ErAlg4Net(1024, 6, 2, _eralg4_args(woe_lc_lambda=1.0, learn_lr=True))


def test_woe_si_term_flag_does_not_reach_the_tracked_scalar() -> None:
    """``woe_lc_term`` must change the penalty and leave the anchor's scalar alone.

    The tracked scalar defines what the path integral *measures* and is fixed by
    the method. If the ablation flag reached it, every conflict-only arm would
    also be running a different anchor and nothing could be attributed.
    """
    torch.manual_seed(12)
    model = WoeSiNet(1024, 6, 2, _woe_args(woe_lc_term="conflict"))
    x = _iq_batch()
    with torch.no_grad():
        features = model.net.forward_features(x, bn_training=False)
        penalty = model._least_commitment_loss(features, t=0)
        tracked = model._compute_information_content(x, t=0, update_feature_mean=False)
        full = model._least_commitment_scalar(
            features, model._current_task_class_indices(0, features.device)
        )
    assert float(tracked.item()) == pytest.approx(float(full.item()), rel=1e-5)
    assert float(penalty.item()) < float(tracked.item())


def test_eralg4_term_flag_reaches_the_penalty() -> None:
    """The same flag, wired through the ER learner's config dataclass."""
    torch.manual_seed(13)
    model = ErAlg4Net(1024, 6, 2, _eralg4_args(woe_lc_lambda=1.0, term="conflict"))
    assert model.lc_term == "conflict"
    features = torch.randn(4, model.net.feature_dim)
    readout = model.net.model.fc
    expected = least_commitment_penalty(
        features,
        readout.weight[3:6],
        readout.bias[3:6],
        features.mean(dim=0),
        term="conflict",
    )
    got = model._least_commitment_loss(features, t=1)
    assert float(got.item()) == pytest.approx(float(expected.item()), rel=1e-6)


def test_term_split_accumulates_and_resets_per_task(monkeypatch) -> None:
    """The split tracker must fill during a task and clear at consolidation.

    It runs in the importance path, so it must work with the objective *off*
    (``woe_lc_lambda=0``) — that is the whole point: measure the balance without
    a penalty on either half moving it.
    """
    monkeypatch.setenv("WOE_LC_DEBUG", "1")
    torch.manual_seed(14)
    model = WoeSiNet(1024, 6, 2, _woe_args(woe_lc_lambda=0.0))
    x, y = _iq_batch(rows=4), torch.randint(0, 3, (4,), dtype=torch.long)

    for _ in range(3):
        model.observe(x, y, 0)
    assert model._term_split_steps == 3
    # The `early_*` pair mirrors the first WOE_SPLIT_EARLY steps, which is the
    # window available before a task's capacity is spent.
    assert set(model._term_split_sums) == {
        "logit",
        "conflict",
        "early_logit",
        "early_conflict",
    }

    model.on_task_end()
    assert model._term_split_steps == 0
    assert model._term_split_sums == {}


def test_term_split_is_inert_without_the_env_var() -> None:
    """No accumulation, hence no per-step cost, when debugging is off."""
    torch.manual_seed(15)
    model = WoeSiNet(1024, 6, 2, _woe_args(woe_lc_lambda=0.0))
    model.observe(_iq_batch(rows=4), torch.zeros(4, dtype=torch.long), 0)
    assert model._term_split_steps == 0
    assert model._term_split_sums == {}


# ----------------------------------------------------------------------
# Conflict-seeking (`woe_lc_term="kappa"`, negative `woe_lc_lambda`)
# ----------------------------------------------------------------------
def test_kappa_is_bounded_where_the_raw_conflict_term_is_not() -> None:
    """The property the whole term exists for: a negative lambda cannot run away.

    Rewarding conflict means minimising ``-conflict``, and the raw term is
    quadratic in the readout scale with no upper bound, so that reward is
    unbounded. Worse, cross-entropy only ever sees ``w+ - w- = z'``, so the
    inflation direction is one it cannot object to. ``kappa`` keeps the same
    "both channels large" incentive but saturates in ``[0, 1]``.
    """
    torch.manual_seed(21)
    features = torch.randn(8, 16).abs()
    weight = torch.randn(4, 16) * 0.5
    bias = torch.zeros(4)
    mean = features.mean(dim=0)

    def at(scale: float, term: str) -> float:
        return float(
            least_commitment_penalty(
                features, scale * weight, scale * bias, mean, term=term, tau=4.0
            )
        )

    # Quadratic in the scale, to four decades and beyond.
    assert at(10.0, "conflict") == pytest.approx(100.0 * at(1.0, "conflict"), rel=1e-4)
    # Bounded, and already saturated by the time the raw term has grown 100x.
    for scale in (0.5, 1.0, 10.0, 1000.0):
        assert 0.0 <= at(scale, "kappa") <= 1.0
    assert at(1000.0, "kappa") == pytest.approx(1.0, abs=1e-6)


def test_kappa_is_zero_for_one_sided_evidence() -> None:
    """Same defining property as the raw conflict half, so it is the same wish.

    A readout with no counter-support anywhere has nothing to be in conflict
    with, and the Dempster conflict between a mass function and a vacuous one is
    exactly zero.
    """
    features = torch.ones(4, 6)
    mean = torch.zeros(6)
    weight = torch.ones(2, 6)  # every w_jk > 0 -> w_minus == 0 for both classes
    bias = torch.zeros(2)

    assert (
        float(
            least_commitment_penalty(
                features, weight, bias, mean, term="kappa", tau=4.0
            )
        )
        == 0.0
    )


def test_kappa_rejects_a_non_positive_tau() -> None:
    """``tau`` divides the evidence, and a run must not start with a bad one."""
    features = torch.randn(3, 5)
    with pytest.raises(ValueError, match="tau must be positive"):
        least_commitment_penalty(
            features,
            torch.randn(2, 5),
            torch.randn(2),
            features.mean(dim=0),
            term="kappa",
            tau=0.0,
        )


def test_kappa_carries_no_feature_count_divisor() -> None:
    """``I_p`` is divided by ``J^p`` to be a per-feature average; a mass is not.

    Dividing a bounded ``[0, 1]`` quantity by ``J^p`` would put it far under the
    cross-entropy it is traded against and make ``woe_lc_lambda`` meaningless.
    """
    torch.manual_seed(22)
    narrow = torch.randn(6, 8)
    weight = torch.randn(3, 8)
    penalty = least_commitment_penalty(
        narrow, weight, torch.zeros(3), narrow.mean(dim=0), term="kappa", tau=4.0
    )
    w_plus, w_minus = per_class_total_evidence(
        compute_weights_of_evidence(narrow, weight, torch.zeros(3), narrow.mean(dim=0))
    )
    expected = (
        evidence_to_belief(w_plus, 4.0) * evidence_to_belief(w_minus, 4.0)
    ).mean()
    assert float(penalty) == pytest.approx(float(expected), rel=1e-6)


def test_negative_lambda_turns_the_term_into_a_reward() -> None:
    """End-to-end wiring: the sign of ``woe_lc_lambda`` reaches the loss.

    The conflict-seeking arm is nothing but this sign flip, so it is worth a test
    that the objective actually subtracts rather than being clamped somewhere on
    the way in.
    """
    torch.manual_seed(23)
    model = WoeSiNet(1024, 6, 2, _woe_args(woe_lc_lambda=-1.0, woe_lc_term="kappa"))
    assert model.lc_lambda == -1.0
    assert model.lc_tau == 4.0

    features = torch.randn(4, model.net.model.fc.in_features, requires_grad=True)
    kappa = model._least_commitment_loss(features, t=0)
    assert 0.0 <= float(kappa.detach()) <= 1.0
    (model.lc_lambda * kappa).backward()
    # A reward on kappa pushes the features the way that *raises* it.
    ascent = -features.grad
    assert float(ascent.norm()) > 0.0
