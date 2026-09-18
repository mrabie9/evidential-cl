"""Weight-of-Evidence Synaptic Intelligence (WoE-SI) continual learner.

This module implements "Option C" of a Dempster-Shafer (DS) based importance
estimator for regularisation-based continual learning. It is structurally a
Synaptic Intelligence learner (Zenke et al. 2017), but the tracked scalar that
defines per-parameter importance is **not** the task loss. It is the
Least-Commitment information content ``I_2(m)`` of the DS mass function ``m``
that underlies the classifier's softmax output (Denoeux 2019, "Logistic
Regression, Neural Networks and Dempster-Shafer Theory: A New Perspective",
arXiv:1807.01846v2).

Concretely, for a standard linear readout ``z_k = sum_j beta_jk * phi_j + beta_0k``
on top of penultimate backbone features ``phi``:

* per-feature/per-class weights of evidence (Denoeux Eq 25)::

      w_jk(x) = beta_jk * phi'_j(x) + alpha_jk

  with centered features ``phi'_j = phi_j - mu_j`` and Least-Commitment offsets
  ``alpha_jk = beta_0k / J`` (the centered multi-category generalisation of the
  binary solution in Sec 4.1 Eq 35; ``sum_j alpha_jk = beta_0k``, Eq 29).

* per-class total weights of evidence (Eq 27)::

      w_k_plus  = sum_j relu( w_jk)     # mass supporting {theta_k}
      w_k_minus = sum_j relu(-w_jk)     # mass supporting complement of {theta_k}

* Least-Commitment information content (Eq 10, p=2)::

      I_2(m) = sum_k [ w_k_plus^2 + w_k_minus^2 ]

``I_2(m)`` is a differentiable scalar measuring how *committed* (informative,
far from vacuous) the evidence is. Substituting it for the loss in the SI path
integral yields, per parameter ``theta_i`` and optimiser step::

      h_i      = d I_2(m) / d theta_i           (mean over minibatch)
      delta_i  = theta_i(after step) - theta_i(before step)
      omega_i += h_i * delta_i

so importance is each parameter's accumulated share of the *committed evidence*
built up over the task. Parameters that drove the model from vacuity toward
committed evidence get anchored; the rest stay free.

CAVEAT (documented, not a bug): the DS construction is exact only at the linear
readout (penultimate features -> logits). For the ResNet backbone, ``I_2(m)`` is
a function of the readout weights and the penultimate activations; its gradient
nonetheless flows through the whole backbone via backprop, so *every* parameter
receives an importance. See DESIGN_NOTE.md.

The class mirrors ``model/si.py`` exactly: importance buffers, a per-step
accumulation hook, an end-of-task consolidation hook, and a quadratic penalty
added to the training loss. It is a drop-in CL method (``--model woe_si``)
runnable in both TIL and CIL with no harness changes.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model.resnet1d import ResNet1D
from model.replay_utils import (
    ReplayInputMixin,
    unpack_y_to_class_labels,
)
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import (
    classification_cross_entropy,
    compute_inverse_frequency_class_weights,
)

# ======================================================================
# Pure functional core (imported directly by the unit tests)
# ======================================================================
_CENTERING_MODES = (
    "centered_uniform",
    "raw_uniform",
    "prop2_uniform",
    "full_lc",
)

# Which mu centres the weights of evidence (PR-3, docs/woe-cl/preregistration.md).
#
#   "ema"            -- the historical `woe_feature_mean`: a within-current-task
#                       EMA at `woe_mu_momentum`, seeded from the first batch and
#                       zeroed at every task boundary.
#   "frozen_pretask" -- the unweighted mean of phi over the task's *full* training
#                       set, computed before the task's first gradient step and
#                       held fixed for the whole task.
#
# This matters because of where mu enters. `I_2` decomposes as
#
#     I_2 = 1/2 ||z||^2 + 1/2 sum_k ( sum_j |w_jk| )^2
#
# and since `sum_j w_jk = z_k` identically, **mu cancels from the first term and
# appears only in the second**. The first is the logit-norm half that ties with
# CE; the second is the DS-specific half. So the entire DS-specific content of the
# tracked scalar is measured against whatever mu is, and B6's null is open to the
# objection that it reflects a lagging reference rather than inert DS content.
# `frozen_pretask` removes the lag so the objection can be tested rather than
# argued about.
_MU_MODES = ("ema", "frozen_pretask")
# Granularity / mechanism of the anchor:
#   "parameter" -- per-weight SI path integral (the original WoE-SI penalty).
#   "channel"   -- the same path integral, but omega collapsed to per-output-channel
#                  at consolidation so whole filters are protected, not single weights.
#   "output"    -- no path integral at all: distil the DS output evidence
#                  (w_plus / w_minus) of a frozen end-of-task teacher (LwF-style).
_REG_LEVELS = ("parameter", "channel", "output")
# How the quadratic anchor is applied to the parameters:
#   "loss"     -- add lambda * sum_i Omega_i (theta_i - theta_i*)^2 to the training
#                 loss and let the optimiser descend it (the original behaviour).
#   "proximal" -- keep the anchor out of the backward pass entirely and apply its
#                 closed-form minimiser as a post-step update. See
#                 ``Net._apply_proximal_anchor`` for the derivation and why it is
#                 unconditionally stable where the loss form is not.
_ANCHOR_MODES = ("loss", "proximal")
# Scale the functional (evidence) penalties are measured on:
#   "weight" -- raw weights of evidence (w_plus, w_minus), unbounded above.
#   "belief" -- 1 - exp(-w / tau), the mass each channel commits, bounded in
#               [0, 1). See ``evidence_to_belief``.
_EVIDENCE_SCALES = ("weight", "belief")
# Scalar whose path integral defines per-parameter importance. Ablation axis:
# does the Dempster-Shafer construction earn its place, or would any monotone
# "the network became more committed" scalar select the same parameters?
#   "i2"   -- Denoeux I_2(m), the DS information content (the method).
#   "z2"   -- squared norm of the active logits. Cheapest possible stand-in;
#             measured cos(dI_2/dtheta, dz2/dtheta) = 0.850 +/- 0.075 over all
#             10 tasks, so it should select nearly the same parameters.
#   "phi2" -- squared norm of the penultimate features. Knows nothing about the
#             readout or the classes at all.
#   "ce"   -- the task loss, i.e. plain Synaptic Intelligence (Zenke et al.).
#             The canonical baseline. Tracked as *negative* CE so "the scalar
#             went up" still means "the model improved", matching SI's sign
#             convention and keeping the relu in consolidation meaningful.
#   "i1"   -- the p=1 member of the same Denoeux family as "i2", i.e. the L1 norm
#             of the weight-of-evidence matrix. Not a fifth arbitrary stand-in
#             like z2/phi2 but the *sibling* of the method's own scalar, so it
#             asks a sharper question than B6 did: given that any monotone
#             commitment measure selects nearly the same parameters, does the
#             exponent -- the one part of I_p Denoeux picked for tractability
#             rather than principle -- matter either?
_IMPORTANCE_SCALARS = ("i2", "z2", "phi2", "ce", "i1", "logit", "conflict")
# How the signed path integral omega^t is projected onto the non-negative Omega
# the quadratic anchor needs. Some projection is mandatory, not stylistic: a
# negative Omega makes the loss-form penalty unbounded below (an anti-anchor that
# drives theta away from theta* without limit), and in the proximal form sends
# b = 2*lr*lambda*Omega negative -- extrapolating away from theta* for
# b in (-1, 0), with a pole at b = -1 and a reflection below it.
#   "relu" -- keep positive contributions only (the original behaviour).
#   "abs"  -- keep the magnitude, so a strongly *negative* path integral is
#             treated as important rather than as irrelevant.
# The sign of omega describes the parameter's journey; the anchor pins its
# destination theta*, which is the end-of-task value the network already fits the
# task at. Measured on task 0, "relu" zeroes 46.7% of parameters and discards
# 24.2% of the total |omega| mass -- and 28.3% of the top 1% most influential
# parameters by |omega|, which are then left entirely free.
#
# Replay-free 10-task TIL, one-shot, proximal anchor, lr 0.003, n=3:
#   "abs"  at woe_lambda 2.4e5 -> 0.5008 +/- 0.0031
#   "relu" at woe_lambda 1e6   -> 0.4878 +/- 0.0045
# paired +0.0130 on every seed, t = 8.35 (df=2, p < 0.05). lambda does NOT
# transfer between the two: size the grid by the *count* of anchored parameters
# ("abs" roughly doubles it, compounding over consolidations), not by total
# Omega mass, which only rises 1.32x and points a decade too high.
# "uniform" is the control the project never ran: Omega constant across
# parameters, so the anchor becomes a plain proximal pull toward theta* (L2-SP
# with a stability-preserving update) and the path integral contributes nothing.
# It is the only arm that can separate "the *denominator* does no work" -- which
# the xi sweep established -- from "the path integral does no work", which does
# not follow from it and had no control until now.
# "displacement" is the other half of A9's question. `uniform` removes the whole
# of Omega; this removes only the *gradient* factor, setting h = dScalar/dtheta to
# a constant so that omega = sum_steps Delta = the task's net displacement. Read
# against `abs` it isolates which factor of `omega = sum h.Delta` supplies the
# ranking -- the tracked scalar's gradient, or how far the parameter moved.
_OMEGA_TRANSFORMS = ("relu", "abs", "uniform", "displacement")
# How per-task importance is *combined across tasks* into the cumulative Omega.
# Every arm in this project so far has used "sum" without ever asking whether it
# is the right rule; it is the one structural choice the anchor makes that has
# never been varied.
#
# The Dempster-Shafer reading makes it a derivation rather than a preference.
# A simple support function is canonically parameterised by a weight function
# w(A) in (0, 1], and a *weight of evidence* is -log w(A). Dempster's rule
# multiplies the w(A) of the sources it combines, which is why weights of
# evidence *add* -- and it is valid only for **distinct** bodies of evidence.
# Sequential tasks are the textbook case of non-distinct evidence: they share a
# backbone, and task t is initialised from task t-1's solution, so its evidence
# is causally downstream of the evidence already accumulated. Denoeux's
# **cautious rule** (Denoeux 2008, "Conjunctive and disjunctive combination of
# belief functions induced by non-distinct bodies of evidence") replaces the
# product with the minimum-based t-norm, i.e. w(A) <- min_t w_t(A). On the
# weight-of-evidence scale that is exactly a **maximum**, since -log is
# decreasing:
#
#     Dempster (distinct):     Omega_i = sum_t  Omega_i^t
#     cautious (non-distinct): Omega_i = max_t  Omega_i^t
#
#   "sum" -- Dempster's rule (the original behaviour, and every published
#            SI/EWC importance accumulator).
#   "max" -- the cautious rule.
#
# The practical stake is that summed importance saturates: a parameter useful to
# many tasks accumulates without bound and is eventually frozen outright, which
# is why online-EWC and SI are both patched with hand-chosen decay factors. A max
# is idempotent -- re-learning the same task changes nothing -- and bounded by
# the single most demanding task, so if it works it *derives* a fix that the
# literature currently applies as a hyper-parameter.
#
# Note "max" leaves the *support* of Omega unchanged (a max of non-negatives is
# non-zero wherever any term is), so the count of anchored parameters is
# identical to "sum" and only the magnitudes drop. Per A6, count is the thing
# lambda should be sized by -- but here count is held fixed by construction, so
# lambda must instead be re-centred by the measured mass ratio.
# "*_norm" grant the cautious rule its own precondition. Denoeux's rule combines
# weights of evidence that are *on a common scale*, and the raw per-task path
# integrals are not: task 0 runs from random initialisation and takes the whole
# drop in the tracked scalar, and every later task both starts from a trained
# representation and travels less under a progressively stronger anchor. `max`
# is maximally exposed to that incommensurability (it keeps only the largest
# term) where `sum` partially launders it. Under "*_norm" each task's omega is
# scaled to unit total mass before combination, and the combined Omega is then
# rescaled so its total matches what plain summation of the *raw* integrals would
# have given -- so lambda transfers and only the rule differs. "sum_norm" is the
# control that separates the rule from the normalisation.
_OMEGA_ACCUMS = ("sum", "max", "sum_norm", "max_norm")
_OMEGA_NORMALISED = ("sum_norm", "max_norm")


def compute_weights_of_evidence(
    features: torch.Tensor,
    readout_weight: torch.Tensor,
    readout_bias: torch.Tensor,
    feature_mean: torch.Tensor,
    centering_mode: str = "centered_uniform",
) -> torch.Tensor:
    """Compute Denoeux Eq 25 weights of evidence ``w_jk(x)``.

    Args:
        features: Penultimate features ``phi`` with shape ``(batch, J)``.
        readout_weight: Linear readout weight ``beta`` with shape ``(K, J)``
            (``torch.nn.Linear.weight`` convention, ``beta_kj``).
        readout_bias: Linear readout bias ``beta_0`` with shape ``(K,)``.
        feature_mean: Running feature mean ``mu`` with shape ``(J,)`` (the EMA of
            ``phi`` over the current task). Ignored for ``"raw_uniform"``.
        centering_mode: One of ``{"centered_uniform", "raw_uniform", "full_lc"}``.
            ``"centered_uniform"`` (default) centres features and uses the
            Least-Commitment uniform offset ``alpha_jk = beta_0k / J``.
            ``"raw_uniform"`` skips centring (uses raw ``phi``) but keeps the same
            offset. ``"prop2_uniform"`` is Denoeux's own centred solution --
            same centring, but the offset is ``beta'_0k / J`` with
            ``beta'_0k = beta_0k + sum_q beta_qk mu_q``, which is what Sec 4.1
            derives once the features are centred and what Prop 2 Eq 38 gives in
            the multi-category case. ``"full_lc"`` (exact Sec 4.2 identification)
            is not implemented and raises ``NotImplementedError``.

    **The default is not Denoeux's identification, and the gap is not small.**
    ``"centered_uniform"`` imposes ``sum_j alpha_jk = beta_0k`` while feeding the
    formula *centred* features. Denoeux's Sec 4.1 does the algebra the other way:
    writing ``w_jk = beta_jk phi'_j + alpha'_jk`` forces
    ``sum_j alpha'_jk = beta'_0k = beta_0k + sum_q beta_qk mu_q``, so the uniform
    split is ``beta'_0k / J``. The consequence is structural rather than
    cosmetic: under Eq 38 ``sum_j w_jk = z_k`` exactly -- the total weight of
    evidence *is* the logit -- whereas under the default it is ``z_k - beta_k.mu``.
    Measured on the PR-3 ``ema`` task-9 checkpoint, the omitted
    ``sum_q beta*_qk mu_q`` is **20-106x larger than ``beta*_0k``** (per-feature
    offset 2.5e-3 against the default's 4.7e-5), i.e. the two modes are reading
    genuinely different quantities. ``"centered_uniform"`` is kept as the default
    because every recorded result in ``docs/woe-cl/README.md`` was measured under
    it; ``"prop2_uniform"`` is the one to quote against the paper.

    Returns:
        Weights of evidence ``w`` with shape ``(batch, K, J)`` where
        ``w[b, k, j] = beta_kj * phi'_bj + beta_0k / J``.

    Usage:
        >>> w = compute_weights_of_evidence(phi, fc.weight, fc.bias, mu)
        >>> w.shape
        torch.Size([batch, K, J])
    """
    if centering_mode not in _CENTERING_MODES:
        raise ValueError(
            f"centering_mode must be one of {_CENTERING_MODES}, got {centering_mode!r}"
        )
    if centering_mode == "full_lc":
        # The exact Least-Commitment identification of Sec 4.2 (solving for the
        # alpha_jk that minimise total information content subject to Eq 29) is
        # intentionally left unimplemented; the centred-uniform default is the
        # documented design choice. See DESIGN_NOTE.md, Design Decision #1.
        raise NotImplementedError(
            "centering_mode='full_lc' (exact Denoeux Sec 4.2 identification) is "
            "not implemented; use 'centered_uniform' (default) or 'raw_uniform'."
        )

    feature_count = features.shape[1]
    if centering_mode == "raw_uniform":
        centered_features = features
    else:  # centered_uniform, prop2_uniform
        centered_features = features - feature_mean.unsqueeze(0)

    # w[b, k, j] = beta_kj * phi'_bj + alpha_kj.
    evidence = readout_weight.unsqueeze(0) * centered_features.unsqueeze(1)
    if centering_mode == "prop2_uniform":
        # alpha_kj = beta'_0k / J with beta'_0k = beta_0k + sum_q beta_kq mu_q,
        # so that sum_j w_jk = z_k exactly (Denoeux Sec 4.1 / Prop 2 Eq 38).
        effective_bias = readout_bias + readout_weight @ feature_mean
    else:
        effective_bias = readout_bias
    offset = (effective_bias / feature_count).view(1, -1, 1)
    return evidence + offset


def per_class_total_evidence(
    weights_of_evidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Denoeux Eq 27 per-class total weights of evidence.

    Args:
        weights_of_evidence: Tensor ``w`` with shape ``(batch, K, J)``.

    Returns:
        Tuple ``(w_plus, w_minus)`` each with shape ``(batch, K)``: the total
        evidence supporting ``{theta_k}`` and its complement, respectively.
    """
    w_plus = torch.relu(weights_of_evidence).sum(dim=2)
    w_minus = torch.relu(-weights_of_evidence).sum(dim=2)
    return w_plus, w_minus


def evidence_to_belief(
    total_evidence: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Map a total weight of evidence onto the belief mass it commits.

    A weight of evidence is the *logarithmic* parameterisation of a simple
    support function: weight ``w`` leaves vacuous mass ``exp(-w)``, so the mass
    committed to the focal set is ``1 - exp(-w)``. Dempster's rule combines
    simple support functions sharing a focal set by *adding* weights, which is
    why ``w_plus`` is a plain sum over features -- this transform belongs after
    that sum, never inside it.

    The map sends ``[0, inf) -> [0, 1)``, so it is bounded where ``w`` is not.
    That is the point: a one-sided "must not decrease" penalty on ``w`` can be
    satisfied by inflating the readout (``w`` is linear in it, so one global
    rescale satisfies every such constraint at once), whereas the same penalty on
    the belief scale sees a gradient carrying a factor ``exp(-w)`` and stops
    paying for inflation. It also makes equal drops count equally in
    decision-relevant terms: 8.0 -> 7.5 moves belief by 0.0002, 0.5 -> 0.0 moves
    it by 0.393, where a squared penalty in ``w`` scores the two identically.

    ``temperature`` rescales the input to ``1 - exp(-w / tau)``. Weights of
    evidence are sums over ``J`` features and can sit far out on the flat tail of
    the curve, where every drop looks equally negligible; setting ``tau`` near the
    typical ``w`` returns the operating point to the responsive region.

    ``tau`` is not cosmetic. The saturation is *hard* in float32: past
    ``w / tau ~ 16.6`` the result rounds to exactly 1.0 and the gradient (which
    carries a factor ``exp(-w / tau)``) is exactly zero, so a penalty built on
    this transform is silently inert for any evidence above that. Measure the
    typical ``w_plus`` before choosing ``tau``.

    Note this is the mass the *channel in isolation* commits to its focal set.
    It is ``Bel({theta_k})`` of the full class-``k`` mass function only before
    combining with the opposing channel, which discounts it by the conflict
    ``kappa`` (see :func:`_conflict_factor`).

    Args:
        total_evidence: Non-negative total evidence of any shape, typically
            ``w_plus`` or ``w_minus`` from :func:`per_class_total_evidence`.
        temperature: Positive scale divided into the evidence before the
            exponential. ``1.0`` is the plain Dempster-Shafer transform.

    Returns:
        Belief in ``[0, 1)``, same shape as ``total_evidence``.

    Usage:
        >>> w_plus, w_minus = per_class_total_evidence(w)
        >>> belief_plus = evidence_to_belief(w_plus)
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    # -expm1(-x) is 1 - exp(-x) without the catastrophic cancellation at small x.
    return -torch.expm1(-total_evidence / temperature)


# Exponents of the Denoeux I_p family that are implemented. p=2 is the paper's
# tractable choice and the one every recorded result in docs/woe-cl/README.md was
# measured at; p=1 turns the objective into an L1 (sparsity) criterion on the
# weights of evidence. See :func:`information_content`.
_LC_EXPONENTS = (1, 2)


def information_content(
    weights_of_evidence: torch.Tensor,
    conflict_weighting: bool = False,
    p: int = 2,
) -> torch.Tensor:
    """Compute the per-sample Least-Commitment information content ``I_p(m)``.

    Implements Denoeux Eq 10 over the ``2K`` focal sets (the ``K`` singletons
    ``{theta_k}`` with weight ``w_k_plus`` and the ``K`` complements with weight
    ``w_k_minus``)::

        I_p(m) = sum_k [ w_k_plus^p + w_k_minus^p ]

    Denoeux picks ``p = 2`` for tractability and says so explicitly; the exponent
    is a free parameter of the family, not a property of the theory. It is left
    at ``2`` by default here because every recorded result in
    ``docs/woe-cl/README.md`` was measured there.

    ``p = 1`` is the other member worth having, and it is a *different mechanism*
    rather than a milder one. Since ``w_k_plus + w_k_minus = sum_j |w_jk|``,
    ``I_1`` is the **L1 norm of the whole weight-of-evidence matrix** -- so
    minimising it is a sparsity criterion on the evidence: a few features carry
    the support and the rest are driven to vacuity (``w_jk = 0``, contributing to
    neither channel). That is structurally what PackNet and HAT achieve by
    masking, which makes ``p`` a one-parameter bridge between the regularisation
    and architectural families rather than another regularisation knob. ``p = 2``
    spreads evidence instead, because a squared penalty charges the largest
    contributor hardest.

    Args:
        weights_of_evidence: Tensor ``w`` with shape ``(batch, K, J)``.
        conflict_weighting: If ``True``, multiply each class term by a
            kappa-style conflict factor ``(1 + kappa_k)`` derived from the
            pairwise overlap of positive and negative evidence (Eqs 21/31).
            Default ``False`` (the plain ``I_p`` path). See DESIGN_NOTE.md.
        p: Exponent of the family, ``1`` or ``2``. Default ``2``.

    Returns:
        Per-sample information content with shape ``(batch,)``. Always ``>= 0``.

    Raises:
        ValueError: If ``p`` is not in :data:`_LC_EXPONENTS`.
    """
    if p not in _LC_EXPONENTS:
        raise ValueError(f"p must be one of {_LC_EXPONENTS}, got {p!r}")
    w_plus, w_minus = per_class_total_evidence(weights_of_evidence)
    per_class = w_plus.pow(p) + w_minus.pow(p)
    if conflict_weighting:
        per_class = per_class * _conflict_factor(w_plus, w_minus)
    return per_class.sum(dim=1)


def _least_commitment_terms(
    weights_of_evidence: torch.Tensor,
    conflict_weighting: bool = False,
    p: int = 2,
) -> Dict[str, torch.Tensor]:
    """Split ``I_p`` into the two halves of its exact decomposition.

    ``w+_k - w-_k = sum_j w_jk = z'_k`` is the centred logit, so for ``p = 2``::

        I_2 = sum_k (w+_k^2 + w-_k^2)
            = ||z'||_2^2  +  2 * sum_k w+_k * w-_k
              \\__________/     \\__________________/
                "logit"             "conflict"

    The same split survives at ``p = 1``, which is not obvious and is worth
    stating: for non-negative ``a, b``, ``a + b = |a - b| + 2 min(a, b)``, so::

        I_1 = sum_k (w+_k + w-_k)
            = ||z'||_1  +  2 * sum_k min(w+_k, w-_k)

    -- the same "decisiveness plus contradiction" reading, in L1. So the
    ``woe_lc_term`` ablation (E2) transfers to the ``p = 1`` objective unchanged,
    and the exponent and the term are genuinely independent axes.

    Both halves are returned per sample, along with the total, so a caller can
    charge the whole thing or either piece. ``logit + conflict == i2`` holds to
    floating-point exactly at both exponents, which is asserted in the tests.

    Note the key of the total stays ``"i2"`` at every ``p``. It names the *slot*
    (the undivided information content) rather than the exponent, so
    ``woe_lc_term`` keeps one set of choices across the family; the exponent is
    carried by ``p`` alone.

    Args:
        weights_of_evidence: Tensor ``w`` with shape ``(batch, K, J)``.
        conflict_weighting: The A5 ``(1 + kappa)`` re-weighting, applied to the
            total only. Not to be confused with the ``"conflict"`` half -- that
            is a term *of* ``I_p``, this is a multiplier *on* it.
        p: Exponent of the family, ``1`` or ``2``. Default ``2``.

    Returns:
        ``{"i2": ..., "logit": ..., "conflict": ...}``, each ``(batch,)``.
    """
    if p not in _LC_EXPONENTS:
        raise ValueError(f"p must be one of {_LC_EXPONENTS}, got {p!r}")
    w_plus, w_minus = per_class_total_evidence(weights_of_evidence)
    centred_logits = w_plus - w_minus
    if p == 1:
        logit = centred_logits.abs().sum(dim=1)
        conflict = 2.0 * torch.minimum(w_plus, w_minus).sum(dim=1)
    else:
        logit = centred_logits.pow(2).sum(dim=1)
        conflict = 2.0 * (w_plus * w_minus).sum(dim=1)
    return {
        "i2": information_content(
            weights_of_evidence, conflict_weighting=conflict_weighting, p=p
        ),
        "logit": logit,
        "conflict": conflict,
    }


# Terms the *objective* can charge. The first three are the undivided I_p and
# its exact two halves; "kappa" is not a piece of I_p at all but the bounded
# Dempster conflict, which exists so that a *negative* lambda (conflict-seeking)
# is well posed -- see :func:`least_commitment_penalty`.
_LC_TERMS = ("i2", "logit", "conflict", "kappa")

# Scalars a shadow path integral can be built from (WOE_OMEGA_SHADOW). The three
# LC terms plus the B6 stand-ins, so one run can carry the halves of I_2 and the
# rival scalars side by side on a single trajectory.
_SHADOW_SCALARS = ("i2", "logit", "conflict", "ce", "z2", "phi2")


def least_commitment_penalty(
    features: torch.Tensor,
    readout_weight: torch.Tensor,
    readout_bias: torch.Tensor,
    feature_mean: torch.Tensor,
    centering_mode: str = "centered_uniform",
    conflict_weighting: bool = False,
    term: str = "i2",
    p: int = 2,
    tau: float = 4.0,
) -> torch.Tensor:
    """Batch-mean ``I_p(m)`` as a *minimisation target* (the LC objective).

    Everywhere else in this module ``I_2`` is a measurement -- the scalar whose
    path integral defines per-parameter importance. Here it is a loss term:
    added to the cross-entropy it asks the network to **commit no more evidence
    than the data requires**, which is the Least-Commitment principle the DS
    construction is already built on (the module identifies ``alpha_jk =
    beta_0k / J``, the least-committed offsets satisfying Denoeux Eq 29). The
    continual-learning motive is that evidence not spent on the current task is
    evidential room still available to later ones.

    What it actually penalises is legible from the algebra. With
    ``w+_k - w-_k = z'_k`` (the centred logit)::

        I_2 = sum_k (w+_k^2 + w-_k^2) = ||z'||^2 + 2 * sum_k w+_k * w-_k

    so the term is a squared-norm penalty on the centred logits *plus* a penalty
    on per-class conflict (evidence for and against the same class both being
    large). The first half is the well-established confidence penalty; the
    second is the part the DS reading contributes, and it is what distinguishes
    this from plain logit decay.

    ``term`` selects which part of that decomposition is charged, which is the
    ablation that decides whether the DS reading earns its place here. ``"logit"``
    is a plain confidence penalty on the centred logits and has nothing to do
    with evidence theory; ``"conflict"`` is the DS-specific half; ``"i2"``
    (default) is their sum, the Denoeux quantity.

    **The halves are not weaker versions of the whole -- they pull in different
    directions.** AM-GM bounds ``conflict`` by ``I_2``, not by ``logit``, and the
    two halves trade off: balanced channels (``w+ ~ w-``, a class the evidence is
    undecided about) send ``logit`` to ~0 and ``conflict`` to nearly all of
    ``I_2``, while one-sided evidence does the reverse. So ``"conflict"`` is
    minimised by making each class's evidence *purely* one-sided -- all support
    or all counter-support -- and is entirely indifferent to its magnitude. A
    class with enormous ``w+`` and zero ``w-`` scores zero conflict. It is a
    non-contradiction penalty that rewards decisiveness, which is not the same
    objective as "commit less", and arguably not a least-commitment objective at
    all. Charged alone it is the sharpest available test of whether the
    Dempster-Shafer content does anything a confidence penalty does not -- the
    question B6 asks of the tracked importance scalar, asked here of the
    objective -- but read the result as its own mechanism, not as a dialled-down
    ``I_2``.

    **Negative ``lambda`` (conflict-seeking) and why ``term="kappa"`` exists.**
    A negative weight turns any of these from a penalty into a reward, which is
    the "maximise conflict to force feature specialisation" hypothesis: if each
    class's score is a *small residue of large opposing evidence* rather than a
    large one-sided sum, features must take strong and differing positions
    instead of all voting the same way. On the raw ``"conflict"`` term that
    hypothesis is **ill-posed**, and not marginally so. ``conflict`` is
    quadratic in the readout scale and unbounded above, and -- crucially --
    ``w+_k - w-_k = z'_k`` is the only thing cross-entropy sees, so the network
    can inflate both channels together along a direction the CE gradient is
    exactly blind to. The reward is unbounded, the degenerate direction is free,
    and the run walks off it.

    ``"kappa"`` is the bounded form of the same wish::

        kappa_k = (1 - e^{-w+_k / tau}) (1 - e^{-w-_k / tau})   in [0, 1]

    the exact Dempster conflict between the two simple mass functions feeding
    class ``k`` (Eqs 21/31, the quantity :func:`_conflict_factor` returns as
    ``1 + kappa``). It rewards *both* channels being substantial -- so it keeps
    the magnitude incentive the raw product has, which is the part that reads as
    specialisation -- but it saturates at 1, so the reward runs out and ``tau``
    is the knob that says how much opposing evidence counts as enough. Charged
    with a positive lambda it is instead a bounded non-contradiction penalty,
    which is the free control arm.

    ``kappa`` is **not** scale-invariant and that is deliberate; a scale-free
    conflict share (``2w+w- / (w+^2 + w-^2)``) is maximised at ``w+ = w-``, i.e.
    by sending every centred logit to zero, which is the ``"logit"`` penalty
    wearing a different hat and has nothing to do with specialisation. The price
    is that ``tau`` must sit near the measured ``w+`` (4.1-5.3 on this project's
    runs): past ``w / tau ~ 16.6`` the transform rounds to exactly 1.0 in float32
    and the gradient is exactly zero. It is averaged over classes rather than
    summed, so it stays in ``[0, 1]`` at any ``K``, and it carries **no** ``J^p``
    divisor -- it is already a bounded ratio, so dividing would only push it
    under the float32 noise floor and make ``lambda`` incomparable to the CE it
    is traded against.

    ``J^p``-normalised to match :meth:`Net._compute_information_content`, so the
    ``"i2"`` penalty and the tracked importance scalar are literally the same
    number and ``woe_lc_lambda`` is interpretable against the measured ``Omega``
    scale. ``J^p`` rather than a fixed ``J^2`` because each channel total is a
    sum over ``J`` features raised to the ``p``, so this is the divisor that
    makes the term a per-feature *average* at every exponent -- the property the
    normalisation exists for. It does **not** put the exponents on a common
    scale: with weights of evidence measured at single digits, ``I_1 / J``
    and ``I_2 / J^2`` differ by roughly that magnitude again, so
    ``woe_lc_lambda`` must be re-swept per ``p``.

    The two halves carry the same divisor but **not** the same magnitude
    (``w+^2 + w-^2 >= 2 w+ w-`` by AM-GM, with equality only when the channels
    are balanced; at ``p = 1`` the same holds as ``w+ + w- >= 2 min(w+, w-)``),
    so lambda does not transfer between them -- measure with ``WOE_LC_DEBUG=1``
    before sweeping.

    Args:
        features: Penultimate features ``phi`` with shape ``(batch, J)``.
        readout_weight: Readout rows for the classes to charge, ``(K, J)``.
        readout_bias: Matching readout biases, ``(K,)``.
        feature_mean: Centring reference ``mu`` with shape ``(J,)``.
        centering_mode: Passed through to :func:`compute_weights_of_evidence`.
        conflict_weighting: The A5 ``(1 + kappa)`` re-weighting; applies to
            ``term="i2"`` only.
        term: One of ``{"i2", "logit", "conflict", "kappa"}``. See above.
        p: Exponent of the ``I_p`` family, ``1`` or ``2``. Default ``2``.
            ``1`` makes the objective an L1 sparsity criterion on the weights of
            evidence rather than a spreading one; see
            :func:`information_content`.
        tau: Evidence scale of the ``"kappa"`` term's belief transform; ignored
            by every other term. Must be positive and should sit near the
            measured ``w_plus``.

    Returns:
        Non-negative scalar; exactly ``0`` for a vacuous readout
        (``beta = 0``, ``beta_0 = 0``), which is the unconstrained minimiser of
        all four terms at every ``p`` -- and, for ``"kappa"``, the reason a
        *positive* lambda on it still has a vacuous escape while a negative one
        does not.

    Raises:
        ValueError: If ``term`` is not one of :data:`_LC_TERMS`, ``p`` not one
            of :data:`_LC_EXPONENTS`, or ``tau`` non-positive with
            ``term="kappa"``.

    Usage:
        >>> loss = ce + lc_lambda * least_commitment_penalty(
        ...     phi, fc.weight[active], fc.bias[active], mu, term="conflict"
        ... )
    """
    if term not in _LC_TERMS:
        raise ValueError(f"term must be one of {_LC_TERMS}, got {term!r}")
    weights = compute_weights_of_evidence(
        features,
        readout_weight,
        readout_bias,
        feature_mean,
        centering_mode=centering_mode,
    )
    if term == "kappa":
        # Bounded, so no J^p divisor and no `p`: the quantity is already a mass
        # in [0, 1] and the divisor exists to turn a sum over J features into a
        # per-feature average.
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        w_plus, w_minus = per_class_total_evidence(weights)
        kappa = evidence_to_belief(w_plus, tau) * evidence_to_belief(w_minus, tau)
        return kappa.mean(dim=1).mean()
    per_sample = _least_commitment_terms(
        weights, conflict_weighting=conflict_weighting, p=p
    )[term]
    feature_count = features.shape[1]
    return per_sample.mean() / float(feature_count**p)


_EVIDENTIAL_MODES = ("off", "balance", "belief")


def _evidential_log_scores(
    w_plus: torch.Tensor,
    w_minus: torch.Tensor,
    mode: str,
    tau: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-class ``(log b_k, log(1 - b_k))`` for the evidential objective.

    Two bounded readings of "how much does this channel believe ``{theta_k}``",
    both in ``[0, 1]`` so that a two-sided log loss on them is well-posed:

    * ``"balance"`` -- ``b_k = w+_k / (w+_k + w-_k)``, the share of the class's
      total contribution magnitude that supports it. Writing ``s_k = w+_k + w-_k``
      and ``d_k = w+_k - w-_k = z'_k`` gives ``b_k = (1 + d_k/s_k) / 2``: a
      *normalised* margin, exactly invariant to rescaling the readout row. That
      invariance is the point. The raw-evidence version of this objective is
      satisfiable by inflating ``beta`` (``w`` is linear in it, so one global
      rescale satisfies every "commit more" constraint at once) which is what
      makes the one-sided hinge unsound; a ratio cannot be bought that way.
    * ``"belief"`` -- ``b_k = (1 - e^{-w+/tau}) e^{-w-/tau}``, the Dempster-Shafer
      singleton belief: the mass the supporting channel commits, discounted by
      the mass the opposing channel commits elsewhere. Faithful to the evidence
      semantics but *not* scale-invariant, so it needs ``tau`` near the measured
      ``w_plus`` (~4 here) to sit in the responsive region -- see
      :func:`evidence_to_belief` on the hard float32 saturation past ``w/tau ~
      16.6``.

    Both are computed in a form with no cancellation. ``"balance"`` smooths the
    ratio by a term *proportional* to the total (plus an absolute floor for the
    vacuous case), so ``b`` and ``1 - b`` stay exact complements with live
    gradients even at ``w- = 0``: clamping would zero the gradient exactly where
    a perfectly one-sided non-target class needs the strongest push, and an
    absolute ``eps`` would leak the readout's scale back into the loss there.
    ``"belief"`` uses ``1 - b = (1 - e^{-c}) + e^{-(a+c)}``, a sum of non-negative
    terms, rather than subtracting near-equal quantities.

    Args:
        w_plus: Supporting evidence ``(batch, K)``, non-negative.
        w_minus: Opposing evidence ``(batch, K)``, non-negative.
        mode: ``"balance"`` or ``"belief"``.
        tau: Evidence scale for ``"belief"``; ignored by ``"balance"``.
        eps: Smoothing added to the ``"balance"`` ratio.

    Returns:
        ``(log_b, log1m_b)``, each ``(batch, K)``.

    Raises:
        ValueError: If ``mode`` is not a scoring mode.
    """
    if mode == "balance":
        # Smoothing *proportional* to the total, plus an absolute floor. The
        # proportional part is what keeps the score exactly scale-invariant even
        # where a channel is exactly zero -- an absolute eps alone shifts
        # log(0 + eps) - log(scale * total) by -log(scale), so a perfectly
        # one-sided class would leak the readout's scale back into the loss. The
        # floor only decides the vacuous case (both channels 0), where it gives
        # b = 1/2, the uninformative score.
        floor = 1e-12
        total = w_plus + w_minus
        denominator = total * (1.0 + 2.0 * eps) + 2.0 * floor
        return (
            torch.log(w_plus + eps * total + floor) - torch.log(denominator),
            torch.log(w_minus + eps * total + floor) - torch.log(denominator),
        )
    if mode == "belief":
        support = w_plus / tau
        against = w_minus / tau
        # b = (1 - e^-a) e^-c; both pieces below are strictly positive.
        log_b = torch.log(-torch.expm1(-support) + eps) - against
        log1m_b = torch.log(-torch.expm1(-against) + torch.exp(-(support + against)))
        return log_b, log1m_b
    raise ValueError(f"mode must be 'balance' or 'belief', got {mode!r}")


def evidential_classification_loss(
    features: torch.Tensor,
    readout_weight: torch.Tensor,
    readout_bias: torch.Tensor,
    feature_mean: torch.Tensor,
    targets: torch.Tensor,
    centering_mode: str = "raw_uniform",
    mode: str = "balance",
    tau: float = 4.0,
    class_balance: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Evidential replacement for cross-entropy: commit evidence *for* the label.

    A per-class two-sided log loss on a bounded evidential score ``b_k``, with a
    one-hot target::

        L = -[ log b_y  +  c * sum_{k != y} log(1 - b_k) ]

    "raise the evidence supporting the correct class, lower the evidence
    supporting the others" -- the objective cross-entropy cannot express,
    because CE sees the readout only through the ``K`` centred logits.

    **What this adds over CE, exactly.** With ``d_k = w+_k - w-_k = z'_k`` and
    ``s_k = w+_k + w-_k``, the pair ``(w+, w-)`` carries one degree of freedom per
    class that the logits do not: ``s_k = sum_j |beta_kj phi'_j + beta_0k/J|``, the
    total contribution magnitude. CE constrains only ``d``. So the novel content
    of this loss lives entirely on ``s``, and (in ``"balance"`` mode) it asks for
    ``|d_k| -> s_k``: every one of the ``J`` contributions to a class sharing one
    sign. That is a *selectivity* objective on the readout rows -- satisfied in
    the limit by rows supported on sign-coherent feature groups -- which is a
    different bet from any of the penalties in this module, all of which
    constrain the model relative to its own past.

    **``c`` is not a free hyper-parameter and must not be 1.** Each class row is
    the target for ``1/K`` of a balanced batch and a non-target for the other
    ``(K-1)/K``, and ``s_k`` enters ``w+_k = (s_k + d_k)/2`` with a positive
    coefficient either way. At ``c = 1`` the aggregate pressure on ``s_k`` is
    therefore ``(1/K)(+1) + ((K-1)/K)(-1) = -(K-2)/K``: a net *shrinkage* of
    commitment at ~0.7 weight for ``K = 6-7``, which is approximately the
    Least-Commitment objective that measured monotonically harmful here (readme
    E1). ``c = 1/(K-1)`` cancels it, leaving no first-order incentive to shrink
    ``s`` at all -- so the only way left to reduce the loss is to make the
    *conditional* distributions differ, which is the intended mechanism.
    ``class_balance`` additionally reweights samples by inverse label frequency
    (the scheme :func:`classification_cross_entropy` already uses), which is what
    makes the effective composition uniform and ``1/(K-1)`` the right constant
    under the real, very unequal class priors.

    ``centering_mode`` defaults to ``"raw_uniform"`` rather than the module
    default. Centred features give ``w+_k - w-_k = z_k - beta_k . mu``, i.e. the
    score this loss optimises is the evaluated logit shifted by a per-class
    constant, and ``argmax`` over the two can differ -- harmless while the
    weights of evidence only feed the importance path, load-bearing once they are
    the objective. Raw features give ``w+_k - w-_k = z_k`` exactly, the quantity
    ``main.py`` scores, and remove the dependence on a running mean that is reset
    at every task boundary. The cost is the marginal identity ``E[w+] = E[w-]``
    that centring enforces, which is what stops "commit more evidence" being
    satisfiable by global inflation -- redundant here, since both scores are
    bounded.

    Args:
        features: Penultimate features ``phi`` ``(batch, J)``, attached so the
            loss reaches the backbone.
        readout_weight: Readout rows for the classes charged, ``(K, J)``.
        readout_bias: Matching biases, ``(K,)``.
        feature_mean: Centring reference ``mu`` ``(J,)``; unused when
            ``centering_mode="raw_uniform"``.
        targets: Labels as *local* indices into the ``K`` rows, ``(batch,)``.
        centering_mode: Passed to :func:`compute_weights_of_evidence`.
        mode: ``"balance"`` or ``"belief"``; see :func:`_evidential_log_scores`.
        tau: Evidence scale for ``"belief"``.
        class_balance: Weight samples by inverse label frequency in the batch.
        eps: Ratio smoothing for ``"balance"``.

    Returns:
        Scalar loss, non-negative and bounded below by 0 (attained only in the
        unreachable limit ``b_y = 1``, ``b_{k != y} = 0``).

    Raises:
        ValueError: If ``mode`` is not ``"balance"`` or ``"belief"``.

    Usage:
        >>> loss = evidential_classification_loss(
        ...     phi, fc.weight[active], fc.bias[active], mu, y_local
        ... )
    """
    if mode not in ("balance", "belief"):
        raise ValueError(f"mode must be 'balance' or 'belief', got {mode!r}")
    weights = compute_weights_of_evidence(
        features,
        readout_weight,
        readout_bias,
        feature_mean,
        centering_mode=centering_mode,
    )
    w_plus, w_minus = per_class_total_evidence(weights)
    log_b, log1m_b = _evidential_log_scores(w_plus, w_minus, mode, tau, eps=eps)

    class_count = log_b.shape[1]
    target_index = targets.long().view(-1, 1)
    log_b_target = log_b.gather(1, target_index).squeeze(1)
    # sum over k != y, without materialising a mask
    log1m_non_target = log1m_b.sum(dim=1) - log1m_b.gather(1, target_index).squeeze(1)
    non_target_weight = 1.0 / float(class_count - 1) if class_count > 1 else 0.0
    per_sample = -(log_b_target + non_target_weight * log1m_non_target)

    if not class_balance:
        return per_sample.mean()
    class_weights = compute_inverse_frequency_class_weights(
        targets.long(), class_count, features.device
    )
    sample_weights = class_weights[targets.long()]
    return (sample_weights * per_sample).sum() / sample_weights.sum().clamp_min(eps)


def _conflict_factor(w_plus: torch.Tensor, w_minus: torch.Tensor) -> torch.Tensor:
    """Kappa-style conflict factor for the optional ablation (Eqs 21/31).

    The two simple mass functions feeding class ``k`` place belief
    ``1 - exp(-w_k_plus)`` on ``{theta_k}`` and ``1 - exp(-w_k_minus)`` on its
    complement. As ``{theta_k}`` and its complement are disjoint, Dempster's
    combination assigns their product to the empty set, i.e. the degree of
    conflict ``kappa_k``. We return ``1 + kappa_k`` so that classes whose
    evidence is internally conflicting receive *more* importance weight, letting
    the user test whether conflict-awareness changes parameter selection.

    Args:
        w_plus: Positive total evidence ``(batch, K)``.
        w_minus: Negative total evidence ``(batch, K)``.

    Returns:
        Conflict factor with shape ``(batch, K)``, all ``>= 1``.
    """
    kappa = evidence_to_belief(w_plus) * evidence_to_belief(w_minus)
    return 1.0 + kappa


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class WoeSiConfig:
    """Hyper-parameters with sensible fallbacks pulled from ``args``.

    The continual-learning knobs mirror SI's ``si_c``/``si_epsilon`` naming so
    the YAML config schema extends cleanly:

    * ``woe_lambda`` -- penalty strength ``lambda`` (analogue of SI's ``si_c``).
    * ``woe_xi`` -- damping ``xi`` in the per-task normaliser (default ``1e-3``).
    * ``woe_centering_mode`` -- ``alpha``/centering scheme (Design Decision #1).
    * ``woe_mu_momentum`` -- EMA momentum for feature means (default ``0.9``).
    * ``woe_importance_stride`` -- compute ``h_i`` every ``k`` steps (default 1).
    * ``woe_conflict_weighting`` -- enable the conflict ablation (default False).
    * ``woe_reg_level`` -- regularisation granularity / mechanism, one of
      ``{"parameter", "channel", "output"}`` (default ``"parameter"``). See
      ``_REG_LEVELS``. ``"output"`` is a functional (evidence-distillation)
      penalty and is *not* on the SI path-integral scale, so it needs its own
      ``woe_lambda``.
    * ``woe_omega_winsorise`` -- quantile in ``(0, 1)`` at which the cumulative
      ``Omega`` is capped after each consolidation; ``0`` (default) disables it.
      The path integral is extremely heavy-tailed in practice, so a handful of
      parameters can otherwise carry curvature the optimiser cannot integrate.
    * ``woe_anchor_mode`` -- ``"loss"`` (default) or ``"proximal"``; see
      ``_ANCHOR_MODES``. Ignored when ``woe_reg_level`` is ``"output"``, which is
      a functional penalty with no per-parameter anchor to apply.
    """

    inner_steps: int = 1
    lr: float = 0.001

    woe_lambda: float = 0.1
    woe_xi: float = 1e-3
    woe_centering_mode: str = "centered_uniform"
    woe_mu_momentum: float = 0.9
    woe_importance_stride: int = 1
    woe_conflict_weighting: bool = False
    woe_reg_level: str = "parameter"
    woe_omega_winsorise: float = 0.0
    woe_anchor_mode: str = "loss"

    optimizer: str = "sgd"
    clipgrad: Optional[float] = 0.0
    cls_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "WoeSiConfig":
        cfg = WoeSiConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


# ======================================================================
# Learner
# ======================================================================
class Net(ReplayInputMixin, nn.Module):
    """Weight-of-Evidence Synaptic Intelligence learner built on ``ResNet1D``.

    Mirrors ``model.si.Net``: the only behavioural difference is that the
    accumulated importance ``omega`` tracks the path integral of the DS
    information content ``I_2(m)`` instead of the task loss.
    """

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()
        del n_inputs  # ResNet1D fixes its own receptive field

        assert n_tasks > 0, "WoE-SI requires at least one task"

        self.cfg = WoeSiConfig.from_args(args)
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.is_task_incremental = True

        self.net = ResNet1D(n_outputs, args)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.incremental_loader_name = getattr(args, "loader", None)
        # CIL <-> full shared head; TIL <-> task-masked head. Mirrors the rule in
        # utils.misc_utils._effective_cil_upto_for_loader.
        self.is_cil = self.incremental_loader_name == "class_incremental_loader"
        self.opt = self._build_optimizer()

        if self.cfg.woe_centering_mode not in _CENTERING_MODES:
            raise ValueError(
                f"woe_centering_mode must be one of {_CENTERING_MODES}, "
                f"got {self.cfg.woe_centering_mode!r}"
            )
        if self.cfg.woe_reg_level not in _REG_LEVELS:
            raise ValueError(
                f"woe_reg_level must be one of {_REG_LEVELS}, "
                f"got {self.cfg.woe_reg_level!r}"
            )
        if self.cfg.woe_anchor_mode not in _ANCHOR_MODES:
            raise ValueError(
                f"woe_anchor_mode must be one of {_ANCHOR_MODES}, "
                f"got {self.cfg.woe_anchor_mode!r}"
            )
        self.omega_winsorise = float(self.cfg.woe_omega_winsorise)
        if not 0.0 <= self.omega_winsorise < 1.0:
            raise ValueError(
                "woe_omega_winsorise must lie in [0, 1) (0 disables capping), "
                f"got {self.omega_winsorise!r}"
            )
        self.anchor_mode = str(self.cfg.woe_anchor_mode)
        # "output" is a functional penalty with no per-parameter anchor, so the
        # proximal path never applies there.
        self.use_proximal_anchor = (
            self.anchor_mode == "proximal" and self.cfg.woe_reg_level != "output"
        )
        self.reg_level = str(self.cfg.woe_reg_level)
        self.woe_lambda = float(self.cfg.woe_lambda)
        self.xi = float(self.cfg.woe_xi)
        self.centering_mode = str(self.cfg.woe_centering_mode)
        self.mu_momentum = float(self.cfg.woe_mu_momentum)
        self.mu_mode = str(getattr(args, "woe_mu_mode", "ema"))
        if self.mu_mode not in _MU_MODES:
            raise ValueError(
                f"woe_mu_mode must be one of {_MU_MODES}, got {self.mu_mode!r}"
            )
        # Bound `IncrementalLoader.get_tasks`, attached to args by main.py beside
        # `get_samples_per_task`. The pre-pass needs task t's *full* training set
        # while the learner holds no reference to the loader; this is the hook.
        # None (e.g. unit tests, or a harness that does not attach it) is not an
        # error under "ema" and is a hard error under "frozen_pretask", which
        # cannot do its job without the data.
        self._task_loader_fn = getattr(args, "get_task_train_loader", None)
        if self.mu_mode == "frozen_pretask" and self._task_loader_fn is None:
            raise ValueError(
                "woe_mu_mode='frozen_pretask' needs args.get_task_train_loader "
                "(bound from IncrementalLoader.get_tasks in main.py); it is absent."
            )
        self.importance_stride = max(1, int(self.cfg.woe_importance_stride))
        self.conflict_weighting = bool(self.cfg.woe_conflict_weighting)
        self.evidence_scale = str(getattr(args, "woe_evidence_scale", "weight"))
        if self.evidence_scale not in _EVIDENCE_SCALES:
            raise ValueError(
                f"woe_evidence_scale must be one of {_EVIDENCE_SCALES}, "
                f"got {self.evidence_scale!r}"
            )
        self.evidence_belief_tau = float(getattr(args, "woe_evidence_belief_tau", 1.0))
        if self.evidence_belief_tau <= 0.0:
            raise ValueError(
                "woe_evidence_belief_tau must be positive, got "
                f"{self.evidence_belief_tau}"
            )
        self.evidence_asymmetric = bool(getattr(args, "woe_evidence_asymmetric", False))
        self.omega_transform = str(getattr(args, "woe_omega_transform", "relu"))
        if self.omega_transform not in _OMEGA_TRANSFORMS:
            raise ValueError(
                f"woe_omega_transform must be one of {_OMEGA_TRANSFORMS}, "
                f"got {self.omega_transform!r}"
            )
        self.omega_accum = str(getattr(args, "woe_omega_accum", "sum"))
        # Running total of the *raw* per-task masses, used by the "*_norm" rules
        # to restore the sum arm's Omega scale after combining normalised terms.
        self._omega_raw_total = 0.0
        # Per-task displacement stats for the SI-denominator saturation check.
        self._delta_sq_sum = 0.0
        self._delta_contrib_sum = 0.0
        self._delta_floored = 0
        self._delta_numel = 0
        if self.omega_accum not in _OMEGA_ACCUMS:
            raise ValueError(
                f"woe_omega_accum must be one of {_OMEGA_ACCUMS}, "
                f"got {self.omega_accum!r}"
            )
        self.importance_scalar = str(getattr(args, "woe_importance_scalar", "i2"))
        if self.importance_scalar not in _IMPORTANCE_SCALARS:
            raise ValueError(
                f"woe_importance_scalar must be one of {_IMPORTANCE_SCALARS}, "
                f"got {self.importance_scalar!r}"
            )
        self.evidence_distill_lambda = float(
            getattr(args, "woe_evidence_distill_lambda", 0.0)
        )
        if self.evidence_distill_lambda != 0.0 and self.cfg.woe_reg_level == "output":
            raise ValueError(
                "woe_evidence_distill_lambda adds the evidence-distillation term "
                "alongside a parameter anchor; woe_reg_level='output' already "
                "applies that term weighted by woe_lambda. Use one or the other."
            )
        # Least-Commitment objective: I_2 minimised on the current task alongside
        # CE, so the network commits only the evidence the data requires and
        # leaves evidential room for later tasks. Off (0.0) by default.
        self.lc_lambda = float(getattr(args, "woe_lc_lambda", 0.0))
        # I_2 has two minimisers, and only one of them is the intended one.
        # w_jk = beta_kj (phi_j - mu_j) + beta_0k/J is zero either when the
        # readout vanishes (a confidence penalty, which is the point) or when
        # phi -> mu, i.e. every sample collapses onto the running feature mean.
        # The second route is nearly free for the current task -- CE only needs
        # the classes separable and the readout can rescale -- but the backbone
        # is shared, so old tasks' readouts are left reading features squeezed
        # toward a centroid computed on the *new* task's data. Detaching the
        # features confines the term to the readout and isolates which of the
        # two routes any measured effect came from.
        self.lc_readout_only = bool(getattr(args, "woe_lc_readout_only", False))
        # Which half of I_2 the objective charges. "i2" is the Denoeux quantity;
        # "conflict" is the DS-specific half alone; "logit" is the plain
        # confidence penalty that carries no evidence theory, and is the control
        # that makes a "conflict" result interpretable.
        # Evidential objective: replace (or mix with) CE by a two-sided log loss
        # on a bounded evidential score, "commit evidence for the label and not
        # for the others". Unlike every other term here it constrains the current
        # task rather than the model's relation to its own past, and unlike the
        # Least-Commitment objective its pressure on total commitment is
        # asymmetric in the label. Off by default.
        self.evidential_mode = str(getattr(args, "woe_evidential_mode", "off"))
        if self.evidential_mode not in _EVIDENTIAL_MODES:
            raise ValueError(
                f"woe_evidential_mode must be one of {_EVIDENTIAL_MODES}, "
                f"got {self.evidential_mode!r}"
            )
        # Mixing weight against CE: 1.0 replaces CE outright, 0.5 runs both.
        self.evidential_gamma = float(getattr(args, "woe_evidential_gamma", 1.0))
        if not 0.0 <= self.evidential_gamma <= 1.0:
            raise ValueError(
                "woe_evidential_gamma must lie in [0, 1], got "
                f"{self.evidential_gamma!r}"
            )
        self.evidential_tau = float(getattr(args, "woe_evidential_tau", 4.0))
        if self.evidential_tau <= 0.0:
            raise ValueError(
                f"woe_evidential_tau must be positive, got {self.evidential_tau!r}"
            )
        self.evidential_class_balance = bool(
            getattr(args, "woe_evidential_class_balance", True)
        )
        # Which score the *prediction* uses. Evaluation argmaxes whatever
        # `forward` returns, and b_k ranks by d_k/s_k where the logit ranks by
        # d_k, so "logit" measures a model on a rule it was not trained for.
        self.evidential_predict = str(getattr(args, "woe_evidential_predict", "logit"))
        if self.evidential_predict not in ("logit", "score"):
            raise ValueError(
                "woe_evidential_predict must be 'logit' or 'score', got "
                f"{self.evidential_predict!r}"
            )
        self.lc_term = str(getattr(args, "woe_lc_term", "i2"))
        if self.lc_term not in _LC_TERMS:
            raise ValueError(
                f"woe_lc_term must be one of {_LC_TERMS}, got {self.lc_term!r}"
            )
        # Exponent of the I_p family the *objective* charges. The tracked
        # importance scalar keeps its own exponent (woe_importance_scalar), so
        # the two never move together.
        self.lc_p = int(getattr(args, "woe_lc_p", 2))
        if self.lc_p not in _LC_EXPONENTS:
            raise ValueError(
                f"woe_lc_p must be one of {_LC_EXPONENTS}, got {self.lc_p!r}"
            )
        # Evidence scale of the bounded `kappa` term only. Its own flag rather
        # than a reuse of woe_evidential_tau, which belongs to a loss that
        # *replaces* cross-entropy: sharing one number would tie two mechanisms
        # that are never on together and can want different operating points.
        self.lc_tau = float(getattr(args, "woe_lc_tau", 4.0))
        if self.lc_tau <= 0.0:
            raise ValueError(f"woe_lc_tau must be positive, got {self.lc_tau!r}")
        self.lwf_lambda = float(getattr(args, "woe_lwf_lambda", 0.0))
        self.lwf_temperature = float(getattr(args, "woe_lwf_temperature", 5.0))
        if self.lwf_temperature <= 0.0:
            raise ValueError(
                f"woe_lwf_temperature must be positive, got {self.lwf_temperature}"
            )
        self.lwf_kl = nn.KLDivLoss(reduction="batchmean")
        self.teacher_dropout = str(getattr(args, "woe_teacher_dropout", "keep"))
        if self.teacher_dropout not in {"keep", "disable"}:
            raise ValueError(
                "woe_teacher_dropout must be 'keep' or 'disable', got "
                f"{self.teacher_dropout!r}"
            )
        self.clipgrad = self.cfg.clipgrad
        self.cls_lambda = float(self.cfg.cls_lambda)

        self.feature_dim = int(self.net.feature_dim)
        self.current_task: Optional[int] = None
        self._step_in_task = 0
        self._param_to_key: Dict[str, str] = {}
        self._tracked_names: List[str] = []
        # name -> live Parameter object (same tensors the forward graph uses).
        self._tracked_params: Dict[str, nn.Parameter] = {}
        # Transient per-window scratch (h at window start, params at window start).
        self._window_h: Dict[str, torch.Tensor] = {}
        self._window_p_start: Dict[str, torch.Tensor] = {}
        # Output-mode (evidence-distillation) state: frozen end-of-task snapshot of
        # the net and the feature mean used to centre its evidence. Both stay None
        # until the first consolidation, so the penalty is exactly 0 on task 0.
        self.teacher: Optional[nn.Module] = None
        # Per-task accumulators for the logit/conflict split of I_2, filled
        # only under WOE_LC_DEBUG=1. See _accumulate_term_split.
        self._term_split_sums: Dict[str, torch.Tensor] = {}
        self._term_split_steps = 0
        # Shadow path integrals: extra scalars differentiated on the *same*
        # trajectory as the tracked one, so their Omega fields can be compared
        # without the confound that each arm ran its own trajectory at its own
        # lambda. Purely observational -- nothing here reaches the loss. See
        # _capture_shadow_gradients.
        self._shadow_scalars: Tuple[str, ...] = tuple(
            s.strip()
            for s in os.environ.get("WOE_OMEGA_SHADOW", "").split(",")
            if s.strip()
        )
        for _name in self._shadow_scalars:
            if _name not in _SHADOW_SCALARS:
                raise ValueError(
                    f"WOE_OMEGA_SHADOW entry must be one of {_SHADOW_SCALARS}, "
                    f"got {_name!r}"
                )
        self._shadow_h: Dict[str, Dict[str, torch.Tensor]] = {}
        self._grad_stat_sums: Dict[str, float] = {}
        self._grad_stat_steps = 0
        self._shadow_w: Dict[str, Dict[str, torch.Tensor]] = {}
        self._shadow_omega: Dict[str, Dict[str, torch.Tensor]] = {}
        # Per-task accumulators for the argmax agreement between the raw logit,
        # the centred logit and the evidential score, filled only under
        # WOE_EV_DEBUG=1. See _accumulate_score_agreement.
        self._agreement_sums: Dict[str, torch.Tensor] = {}
        self._agreement_steps = 0
        self.teacher_feature_mean: Optional[torch.Tensor] = None
        self._initialise_woe_state()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int, **kwargs) -> torch.Tensor:
        if self.evidential_predict == "score" and self.evidential_mode != "off":
            # Predict with the score that was trained. `b_k` is monotone in
            # `d_k/s_k`, and argmax over that is not argmax over `d_k`: the two
            # disagreed on 3-18% of training samples, with the evidential ranking
            # the *more* accurate of the pair. Training one score and evaluating
            # another is a measurement artefact, not a property of the objective.
            # Split the forward the same way `observe` does -- numerically
            # identical to `self.net(x)`, and it draws no extra randomness
            # because dropout lives in the backbone.
            features = self.net.forward_features(x)
            logits = self._evidential_ranking_score(features)
        else:
            logits = self.net(x)
        if not self.is_task_incremental:
            return logits
        cil = kwargs.get("cil_all_seen_upto_task")
        return misc_utils.apply_task_incremental_logit_mask(
            logits,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil,
            loader=self.incremental_loader_name,
        )

    # ------------------------------------------------------------------
    def observe(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> Tuple[float, float, torch.Tensor | None]:
        if self.current_task is None:
            self.current_task = t
        elif t != self.current_task:
            self._consolidate_current_task()
            self.current_task = t

        # PR-3: the frozen mu pre-pass. Placed here because this is the last point
        # that is still *before* task t's first gradient step -- consolidation of
        # t-1 has just run and cleared `woe_mu_frozen_set`, and nothing below has
        # touched a parameter yet, so the pass sees the model exactly as t-1 left
        # it. Must precede `_dropout_conflict_debug` below, which reads mu.
        #
        # Runs in **both** mu modes, deliberately and against first instinct.
        #
        # The pass is not inert at full scale. Measured, not assumed: with it,
        # seed 0 gives 0.5224 / 0.4984 / -0.0240; without it, 0.5206 / 0.5008 /
        # -0.0198, which is the recorded B6 control reproduced bit-identically.
        # Each of those two trajectories is itself perfectly reproducible across
        # separate launches, so the pass shifts the run deterministically rather
        # than adding noise. Saving and restoring CPU+CUDA RNG around it changed
        # nothing, so the mechanism is not a stolen random draw; the likeliest
        # remaining candidate is workspace-dependent cuDNN algorithm selection
        # (`cudnn.deterministic` fixes the algorithm *given* the available
        # workspace, not across allocator states), but that is unproven and is
        # labelled as such.
        #
        # `frozen_pretask` cannot run without the pass -- it *is* the
        # intervention -- so the only way both arms can share a trajectory is for
        # both to pay the same cost. Running it in one arm only would leave the
        # arms differing by the mu read *plus* a ~0.8 sigma perturbation, which
        # is the confound that matters, since Delta' (frozen vs ema) is what
        # decides PR-3. Common-mode is the cheaper error by a wide margin.
        #
        # The consequence is registered in PR-3 Amendment 1 and must not be
        # forgotten: neither arm is comparable to the recorded README figures any
        # more. Within-experiment comparisons only.
        if self._task_loader_fn is not None and not bool(self.woe_mu_frozen_set.item()):
            self._compute_pretask_mu(t)

        self.net.train()
        if self._step_in_task == 0:
            # Once per task, and gated on the env var rather than on the
            # evidential mode so a CE control run prints the same baseline.
            self._dropout_conflict_debug(x, t)
        metric_logits = None
        # "parameter"/"channel" anchor a per-weight SI path integral; "output"
        # uses a functional evidence-distillation penalty with no path integral.
        use_path_integral = self.reg_level in ("parameter", "channel")
        for _ in range(self.cfg.inner_steps):
            # ----- 1) DS importance gradient h_i = dI_2/dtheta_i -----------
            # Computed at the *pre-step* parameters on a dedicated backward so it
            # never pollutes the CE gradient. Sampled once per importance window.
            window_start = self._step_in_task % self.importance_stride == 0
            if use_path_integral and window_start:
                self._capture_importance_gradient(x, y, t)
                self._capture_shadow_gradients(x, y, t)

            # ----- 2) Standard CE update (drives the parameters) -----------
            self.opt.zero_grad()
            y_cls = unpack_y_to_class_labels(y)
            # Split out of `forward_heads` so the penultimate features are
            # available to the Least-Commitment term without a second forward
            # pass. Numerically identical: `forward_heads` runs exactly these two
            # calls plus the (discarded here) detection head, which draws no
            # randomness and so cannot shift the dropout stream.
            features = self.net.forward_features(x)
            cls_logits = self.net.forward_classifier(features)
            logits_for_loss = cls_logits
            if self.is_task_incremental:
                logits_for_loss = misc_utils.apply_task_incremental_logit_mask(
                    cls_logits,
                    t,
                    self.classes_per_task,
                    self.n_outputs,
                    cil_all_seen_upto_task=t,
                    loader=self.incremental_loader_name,
                )
            targets_for_loss = y_cls.long()
            loss_ce = classification_cross_entropy(
                logits_for_loss,
                targets_for_loss,
                class_weighted_ce=self.class_weighted_ce,
            )
            preds = torch.argmax(logits_for_loss, dim=1)
            cls_tr_rec = macro_recall(preds, y_cls.long())

            if self.reg_level == "output":
                reg = self._evidence_distillation_loss(x, t)
            elif self.use_proximal_anchor:
                # The anchor is applied in closed form after the optimiser step
                # instead, so it contributes nothing to this backward pass -- and
                # therefore nothing to the global gradient-norm clip budget.
                reg = torch.zeros(1, device=self._device())
            else:
                reg = self._surrogate_loss()
            loss_cls = loss_ce
            if self.evidential_mode != "off":
                # Shares the CE forward, so the term costs one readout slice and
                # no extra backbone pass. gamma=1 replaces CE outright.
                evidential = self._evidential_loss(features, t, y_cls)
                loss_cls = (
                    1.0 - self.evidential_gamma
                ) * loss_ce + self.evidential_gamma * evidential
                self._evidential_debug(loss_ce, evidential)
                self._accumulate_score_agreement(features, logits_for_loss, t, y_cls)
            loss = self.cls_lambda * loss_cls + self.woe_lambda * reg
            if self.evidence_distill_lambda != 0.0:
                # DS evidence distillation running *alongside* the parameter
                # anchor rather than replacing it, mirroring how the LwF term
                # attaches. reg_level='output' already applies this term as `reg`,
                # so the two paths are mutually exclusive (validated in __init__).
                loss = loss + self.evidence_distill_lambda * (
                    self._evidence_distillation_loss(x, t)
                )
            if self.lc_lambda != 0.0:
                # Least commitment on the current task. Shares the CE forward, so
                # the term costs one readout slice and no extra backbone pass.
                lc = self._least_commitment_loss(features, t)
                self._lc_debug(loss_ce, lc)
                loss = loss + self.lc_lambda * lc
            if self.lwf_lambda != 0.0:
                loss = loss + self.lwf_lambda * self._lwf_distillation_loss(
                    cls_logits, x, t
                )
            # Rehearsal hook: no-op in base WoE-SI, a reservoir-replay CE term in
            # the woe_si_replay subclass. Sampled before the current batch is
            # written to the buffer so a sample never rehearses on itself.
            loss = loss + self._classification_replay_loss(t)

            loss.backward()
            if self.clipgrad is not None and self.clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
            self.opt.step()
            if self.use_proximal_anchor:
                self._apply_proximal_anchor()

            # ----- 3) Accumulate the I_2 path integral over the window -----
            window_end = (
                self._step_in_task % self.importance_stride
                == self.importance_stride - 1
            )
            if use_path_integral and window_end:
                self._accumulate_path_integral()

            self._step_in_task += 1
            metric_logits = logits_for_loss.detach()

        # Store the current batch after the update so the reservoir reflects the
        # stream. No-op in base WoE-SI (see the rehearsal hooks below).
        self._store_classification_replay(x, y, t)
        return float(loss.item()), cls_tr_rec, metric_logits

    # ------------------------------------------------------------------
    def on_task_end(self) -> None:
        """Optional hook to consolidate the final task."""
        self._consolidate_current_task()

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.net.parameters()
        optim = (self.cfg.optimizer or "adam").lower()
        lr = float(self.cfg.lr)

        if optim in {"adam", "adamw"}:
            opt_cls = torch.optim.AdamW if optim == "adamw" else torch.optim.Adam
            return opt_cls(params, lr=lr)
        if optim == "adagrad":
            return torch.optim.Adagrad(params, lr=lr)
        if optim in {"sgd", "sgd_momentum_decay"}:
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        return torch.optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------
    def _is_tracked(self, name: str, param: nn.Parameter) -> bool:
        del name
        return bool(param.requires_grad)

    # ------------------------------------------------------------------
    def _initialise_woe_state(self) -> None:
        """Register SI-style buffers plus the running feature mean ``mu``."""
        for name, param in self.net.named_parameters():
            if not self._is_tracked(name, param):
                continue
            key = name.replace(".", "__")
            self._param_to_key[name] = key
            self._tracked_names.append(name)
            self._tracked_params[name] = param
            initial = param.detach().clone()
            # theta^* anchor (theta at the start of the current task).
            self.register_buffer(f"{key}_woe_prev", initial.clone())
            # Cumulative importance across tasks.
            self.register_buffer(f"{key}_woe_omega", torch.zeros_like(param))
            # Per-task path-integral accumulator omega^t.
            self.register_buffer(f"{key}_woe_w", torch.zeros_like(param))
        # Running mean mu_j of penultimate features over the current task.
        self.register_buffer("woe_feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer(
            "woe_feature_mean_initialised", torch.zeros(1, dtype=torch.bool)
        )
        # Frozen per-task mu (PR-3). Deliberately *not* named `*_feature_mean`:
        # four things in this repo already carry that name (`woe_feature_mean`,
        # `replay_task_feature_mean`, `eralg4.lc_feature_mean`,
        # `teacher_feature_mean`) and they disagree with one another, so a fifth
        # would be actively misleading. Registered as a buffer so it is
        # checkpointed alongside the mu it replaces.
        self.register_buffer("woe_mu_frozen", torch.zeros(self.feature_dim))
        self.register_buffer("woe_mu_frozen_set", torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------
    def _importance_scalar(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> torch.Tensor:
        """Differentiable scalar whose path integral defines importance.

        ``"i2"`` is WoE-SI proper. The others exist to ablate whether the
        Dempster-Shafer construction is doing any work, or whether any scalar
        that rises as the network commits would select the same parameters.
        Each is normalised to an ``O(1)`` scale like ``I_2`` is, but they are not
        on a *common* scale, so ``woe_lambda`` must be swept per scalar.

        Args:
            x: Current batch.
            y: Current labels; used only by ``"ce"``.
            t: Current task index.

        Returns:
            Scalar tensor, larger meaning "more committed / better fit".
        """
        if self.importance_scalar in ("i2", "i1", "logit", "conflict"):
            return self._compute_information_content(x, t, update_feature_mean=True)

        features = self.net.forward_features(x, bn_training=False)
        self._update_feature_mean(features.detach())
        feature_count = features.shape[1]

        if self.importance_scalar == "phi2":
            return features.pow(2).sum(dim=1).mean() / float(feature_count)

        logits = self.net.forward_classifier(features, bn_training=False)
        if self.importance_scalar == "z2":
            active = self._active_class_indices(t, features.device)
            selected = logits.index_select(1, active)
            return selected.pow(2).sum(dim=1).mean() / float(max(1, active.numel()))

        # "ce": plain Synaptic Intelligence. Negated so that an increase still
        # means the model improved, keeping the relu in consolidation (which
        # protects parameters that *built* something) pointing the right way.
        masked = misc_utils.apply_task_incremental_logit_mask(
            logits,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=t,
            loader=self.incremental_loader_name,
        )
        return -classification_cross_entropy(
            masked,
            unpack_y_to_class_labels(y).long(),
            class_weighted_ce=self.class_weighted_ce,
        )

    # ------------------------------------------------------------------
    def _compute_information_content(
        self, x: torch.Tensor, t: int, update_feature_mean: bool
    ) -> torch.Tensor:
        """Differentiable per-batch mean ``I_2(m)`` over the active logits.

        The forward uses ``bn_training=False`` so the dedicated importance pass
        does not perturb BatchNorm running statistics (they are owned by the CE
        path). Centred features use the running mean ``mu`` (detached), which is
        EMA-updated here from the current batch when ``update_feature_mean`` is
        set.
        """
        features = self.net.forward_features(x, bn_training=False)

        if update_feature_mean:
            self._update_feature_mean(features.detach())

        active = self._active_class_indices(t, features.device)
        # Denoeux's I_2 is an *unnormalised* double sum: each per-class total
        # w_k_plus = sum_j relu(w_jk) is O(J) in the feature dimension J, so
        # I_2 ~ O(K * J^2). Used verbatim as the SI importance signal that scales
        # the quadratic anchor penalty, it dwarfs the O(1) cross-entropy loss by
        # several orders of magnitude (J = feature_dim is in the hundreds, e.g.
        # 512), so the per-task path integral and the cumulative omega -- and
        # hence the surrogate loss -- explode across tasks (training losses of
        # 1e5-1e6 that collapse plasticity on every task after the first).
        # `least_commitment_penalty` divides by J^2 so the per-feature evidence is
        # *averaged* rather than summed; the importance signal then lives on an
        # O(K) ~ O(1) scale like SI's loss-based path integral. Because the path
        # integral telescopes to the change in this scalar, w_buf stays bounded
        # regardless of step count, and the *relative* per-parameter importance
        # the penalty actually uses is unchanged. The exact Denoeux sum is
        # preserved in the functional core (`information_content`) for the paper
        # correspondence and unit tests; only the learner's internal importance
        # signal is rescaled.
        self._accumulate_term_split(features, active)
        # The exponent of the *tracked* scalar. "i1" is the p=1 sibling of the
        # method's own I_2; it is deliberately not tied to woe_lc_p, which
        # governs the objective (see _least_commitment_scalar).
        exponent = 1 if self.importance_scalar == "i1" else 2
        # The two halves of the exact decomposition
        # I_2 = ||z'||^2 + 2 sum_k w+_k w-_k (see _least_commitment_terms).
        # Tracked live rather than as shadows because the path integral is
        # *linear* in the scalar, so omega_i2 = omega_logit + omega_conflict
        # exactly -- the only additive decomposition of omega in this family.
        # (The abs/relu projection at consolidation does not preserve that
        # additivity, which is precisely why the halves need running live
        # rather than being inferred from the i2 field.)
        term = (
            self.importance_scalar
            if self.importance_scalar in ("logit", "conflict")
            else "i2"
        )
        return self._least_commitment_scalar(features, active, term=term, p=exponent)

    # ------------------------------------------------------------------
    def _shadow_scalar_value(
        self, name: str, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> torch.Tensor:
        """One named scalar, differentiable, with **no** side effects.

        Deliberately not :meth:`_importance_scalar`: that one EMA-updates the
        feature mean, and a shadow measurement that moved ``mu`` would perturb
        the very trajectory it exists to observe. Everything else -- centring,
        the ``J^p`` divisor, the active column set, ``bn_training=False`` --
        matches the tracked path exactly, so the only difference between a
        shadow ``Omega`` and a real one is which scalar was differentiated.

        ``"logit"`` and ``"conflict"`` are the two halves of
        ``I_2 = ||z'||^2 + 2 sum_k w+_k w-_k`` (see
        :func:`_least_commitment_terms`). They are the informative basis: the
        alternative split ``I_2 = (||z'||^2 + sum_k (sum_j |w_jk|)^2) / 2`` has
        ``sum_k mass_k^2 = ||z'||^2 + 2 * conflict``, i.e. the mass term
        *contains* the logit term algebraically, so those two are positively
        related by construction and correlating them tests nothing.
        """
        features = self.net.forward_features(x, bn_training=False)
        if name in ("i2", "logit", "conflict"):
            active = self._active_class_indices(t, features.device)
            return self._least_commitment_scalar(features, active, term=name, p=2)
        if name == "phi2":
            return features.pow(2).sum(dim=1).mean() / float(features.shape[1])
        logits = self.net.forward_classifier(features, bn_training=False)
        if name == "z2":
            active = self._active_class_indices(t, features.device)
            selected = logits.index_select(1, active)
            return selected.pow(2).sum(dim=1).mean() / float(max(1, active.numel()))
        masked = misc_utils.apply_task_incremental_logit_mask(
            logits,
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=t,
            loader=self.incremental_loader_name,
        )
        return -classification_cross_entropy(
            masked,
            unpack_y_to_class_labels(y).long(),
            class_weighted_ce=self.class_weighted_ce,
        )

    # ------------------------------------------------------------------
    def _capture_shadow_gradients(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> None:
        """Differentiate each ``WOE_OMEGA_SHADOW`` scalar at the same window.

        The point is one trajectory, several ``Omega`` fields. Comparing the
        tracked scalars across *runs* (as the existing ``omega_dumps`` do)
        confounds "the scalar selects different parameters" with "the arms took
        different paths at different anchor strengths"; here every field is a
        path integral over the identical parameter sequence, so a disagreement
        can only come from the scalar.

        One extra forward and backward per scalar per importance window, none of
        it touching ``param.grad``, the feature mean, or BatchNorm statistics.
        Entirely inert when the env var is unset.
        """
        if not self._shadow_scalars:
            return
        params = [self._tracked_params[name] for name in self._tracked_names]
        for scalar in self._shadow_scalars:
            value = self._shadow_scalar_value(scalar, x, y, t)
            grads = torch.autograd.grad(
                value, params, retain_graph=False, allow_unused=True
            )
            store = self._shadow_h.setdefault(scalar, {})
            for name, param, grad in zip(self._tracked_names, params, grads):
                store[name] = (
                    torch.zeros_like(param) if grad is None else grad.detach().clone()
                )

    # ------------------------------------------------------------------
    def _shadow_buffer(self, kind: str, scalar: str, name: str) -> torch.Tensor:
        """Lazily allocated ``w``/``omega`` accumulator for a shadow scalar.

        Plain tensors rather than registered buffers: this is a diagnostic and
        has no business appearing in a checkpoint's ``state_dict``.
        """
        table = self._shadow_w if kind == "w" else self._shadow_omega
        store = table.setdefault(scalar, {})
        if name not in store:
            store[name] = torch.zeros_like(self._tracked_params[name])
        return store[name]

    # ------------------------------------------------------------------
    def _weights_of_evidence(
        self,
        features: torch.Tensor,
        readout: nn.Module,
        active: torch.Tensor,
        feature_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Denoeux Eq 25 weights of evidence ``w_jk`` over the ``active`` classes.

        Shared by the importance path (``_compute_information_content``) and the
        output-mode distillation (``_evidence_distillation_loss``) so both centre
        and slice the readout identically.
        """
        active_weight = readout.weight[active]
        active_bias = readout.bias[active]
        return compute_weights_of_evidence(
            features,
            active_weight,
            active_bias,
            feature_mean,
            centering_mode=self.centering_mode,
        )

    # ------------------------------------------------------------------
    def _least_commitment_scalar(
        self,
        features: torch.Tensor,
        class_indices: torch.Tensor,
        term: str = "i2",
        p: int = 2,
    ) -> torch.Tensor:
        """``J^p``-normalised batch-mean DS commitment over ``class_indices``.

        One code path for the two things this scalar is used for: measured (the
        SI path integral) and minimised (the Least-Commitment objective), so the
        two can never drift apart in normalisation or centring.

        ``term`` and ``p`` both default to the method's own choice and the
        importance path always takes those defaults. The tracked scalar defines
        what the anchor *measures*; ``woe_lc_term`` and ``woe_lc_p`` select only
        what the objective *charges*. Letting either flag reach both would change
        the anchor and the penalty together and confound every arm -- and E1
        showed that confound is not hypothetical, since an objective that moves
        ``I_2`` moves ``Omega`` with it. The exponent of the *tracked* scalar has
        its own flag (``woe_importance_scalar='i1'``), which is the B6-style
        ablation and a deliberately separate axis.
        """
        readout = self.net.model.fc
        return least_commitment_penalty(
            features,
            readout.weight[class_indices],
            readout.bias[class_indices],
            self._mu_for_evidence(),
            centering_mode=self.centering_mode,
            conflict_weighting=self.conflict_weighting,
            term=term,
            p=p,
            tau=self.lc_tau,
        )

    # ------------------------------------------------------------------
    def _least_commitment_loss(self, features: torch.Tensor, t: int) -> torch.Tensor:
        """Least-Commitment objective: minimise ``I_2`` on the current task.

        Charged on the **current task's** class columns only, not on every active
        column. In TIL the two coincide, but in CIL the active set is cumulative,
        and asking the network to *un-commit* evidence on classes it already
        learned is a forgetting mechanism, not a regularisation one. The
        counterpart -- protecting prior classes -- belongs to the terms that hold
        them fixed against a frozen teacher (``woe_lwf_lambda``,
        ``woe_evidence_distill_lambda``), which compose with this one.

        Deliberately does *not* EMA-update ``mu``: the importance path
        (``_capture_importance_gradient``) or the output-mode distillation
        already updates it once per step, and a second update per step would
        silently double the EMA rate.

        Args:
            features: Penultimate features of the current batch, still attached
                to the graph so the penalty reaches the backbone as well as the
                readout.
            t: Current task index.

        Returns:
            Scalar penalty; ``0`` if the current task has no columns.
        """
        current = self._current_task_class_indices(t, features.device)
        if current.numel() == 0:
            return torch.zeros(1, device=features.device)
        if self.lc_readout_only:
            features = features.detach()
        return self._least_commitment_scalar(
            features, current, term=self.lc_term, p=self.lc_p
        )

    # ------------------------------------------------------------------
    def _evidential_loss(
        self, features: torch.Tensor, t: int, y_cls: torch.Tensor
    ) -> torch.Tensor:
        """Evidential classification loss over the current task's columns.

        Charged on the current task's columns for the same reason the
        Least-Commitment term is (:meth:`_least_commitment_loss`): in CIL the
        active set is cumulative, and a two-sided loss over old columns would ask
        the model to *remove* evidence from classes it learned earlier, which is
        a forgetting mechanism. In TIL the two sets coincide.

        Labels arrive as global column indices and the loss needs positions
        within the ``K`` rows it was handed, so the mapping goes through a
        scatter rather than a subtraction -- the column set is
        ``[offset1, offset2)`` *plus* any always-active noise column, which need
        not be contiguous. Samples whose label falls outside the set (never
        expected under TIL) are dropped rather than silently folded onto row 0.

        Args:
            features: Penultimate features of the current batch, attached.
            t: Current task index.
            y_cls: Global class labels ``(batch,)``.

        Returns:
            Scalar loss; ``0`` if the task has no columns or no usable labels.
        """
        current = self._current_task_class_indices(t, features.device)
        if current.numel() == 0:
            return torch.zeros(1, device=features.device)
        lookup = torch.full(
            (self.n_outputs,), -1, dtype=torch.long, device=features.device
        )
        lookup[current] = torch.arange(current.numel(), device=features.device)
        local_targets = lookup[y_cls.long()]
        usable = local_targets >= 0
        if not bool(usable.any()):
            return torch.zeros(1, device=features.device)
        readout = self.net.model.fc
        return evidential_classification_loss(
            features[usable],
            readout.weight[current],
            readout.bias[current],
            self._mu_for_evidence(),
            local_targets[usable],
            centering_mode=self.centering_mode,
            mode=self.evidential_mode,
            tau=self.evidential_tau,
            class_balance=self.evidential_class_balance,
        )

    # ------------------------------------------------------------------
    def _evidential_ranking_score(self, features: torch.Tensor) -> torch.Tensor:
        """Full-head evidential score, a drop-in replacement for the logits.

        Returns ``d_k / s_k = (w+_k - w-_k) / (w+_k + w-_k)``, which is
        ``2 b_k - 1`` for the ``balance`` score and therefore ranks classes
        exactly as ``b_k`` does. Bounded in ``[-1, 1]``, so the ``-1e9`` fill the
        task-incremental mask writes still dominates every real column.

        Note what the transform *is*: under ``raw_uniform`` the numerator is the
        raw logit exactly, so this is a per-class **normalisation** of the logit
        by the total contribution magnitude. That is the whole content of the
        objective's novelty as a decision rule -- a logit is worth more when the
        readout row earned it with less internal cancellation.

        Args:
            features: Penultimate features ``(batch, J)``.

        Returns:
            Scores ``(batch, n_outputs)`` over the full head.
        """
        readout = self.net.model.fc
        weights = compute_weights_of_evidence(
            features,
            readout.weight,
            readout.bias,
            self._mu_for_evidence(),
            centering_mode=self.centering_mode,
        )
        w_plus, w_minus = per_class_total_evidence(weights)
        return (w_plus - w_minus) / (w_plus + w_minus + 1e-6)

    # ------------------------------------------------------------------
    def _evidential_debug(
        self, loss_ce: torch.Tensor, evidential: torch.Tensor
    ) -> None:
        """Print the evidential term against the CE it replaces (``WOE_EV_DEBUG=1``).

        Same motive as :meth:`_lc_debug`: no scale in this module has ever
        transferred, and a replacement objective has to be read against the one
        it displaces before any of its own hyper-parameters mean anything.
        """
        if os.environ.get("WOE_EV_DEBUG") != "1":
            return
        seen = getattr(self, "_ev_debug_steps", 0)
        if seen >= 12:
            return
        self._ev_debug_steps = seen + 1
        ce_value = float(loss_ce.item())
        ev_value = float(evidential.item())
        print(
            f"[EV] step={seen} ce={ce_value:.4f} mode={self.evidential_mode} "
            f"evidential={ev_value:.4f} gamma={self.evidential_gamma:g} "
            f"ratio={ev_value / max(ce_value, 1e-12):.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _accumulate_score_agreement(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        t: int,
        y_cls: torch.Tensor,
    ) -> None:
        """Accumulate ``argmax`` agreement between the three scores (``WOE_EV_DEBUG=1``).

        Training optimises an evidential score while ``main.py`` predicts with
        ``argmax`` over the raw logits, and the three candidate rankings are not
        the same function:

        * ``z``  -- raw logits, what evaluation scores
        * ``z'`` -- centred logits ``w+ - w-``; equals ``z - beta . mu``, so under
          ``centered_uniform`` it is the *shifted* ranking, and the shift
          ``beta_k . mu`` is an inner product of the row with a strictly positive
          mean (features are post-ReLU). Under ``raw_uniform`` the two coincide
          exactly, which this measurement is also the check on.
        * ``b``  -- the evidential score actually minimised.

        Agreement rates say whether the mismatch is immaterial (>99%, ignore it)
        or whether the prediction rule has to follow the objective. Accumulated
        on-device and synced once per task.
        """
        if os.environ.get("WOE_EV_DEBUG") != "1":
            return
        current = self._current_task_class_indices(t, features.device)
        if current.numel() == 0:
            return
        readout = self.net.model.fc
        weights = compute_weights_of_evidence(
            features.detach(),
            readout.weight[current].detach(),
            readout.bias[current].detach(),
            self._mu_for_evidence(),
            centering_mode=self.centering_mode,
        )
        w_plus, w_minus = per_class_total_evidence(weights)
        log_b, _ = _evidential_log_scores(
            w_plus, w_minus, self.evidential_mode, self.evidential_tau
        )
        raw = logits[:, current].argmax(dim=1)
        centred = (w_plus - w_minus).argmax(dim=1)
        belief = log_b.argmax(dim=1)
        lookup = torch.full(
            (self.n_outputs,), -1, dtype=torch.long, device=features.device
        )
        lookup[current] = torch.arange(current.numel(), device=features.device)
        truth = lookup[y_cls.long()]
        sums = self._agreement_sums
        for name, value in (
            ("z_vs_zc", (raw == centred).float().mean()),
            ("z_vs_b", (raw == belief).float().mean()),
            ("zc_vs_b", (centred == belief).float().mean()),
            ("acc_z", (raw == truth).float().mean()),
            ("acc_b", (belief == truth).float().mean()),
        ):
            sums[name] = sums.get(name, value * 0.0) + value
        self._agreement_steps += 1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _score_agreement_debug(self) -> None:
        """Print the per-task mean agreement rates and reset (``WOE_EV_DEBUG=1``)."""
        if os.environ.get("WOE_EV_DEBUG") != "1" or self._agreement_steps == 0:
            return
        steps = float(self._agreement_steps)
        parts = " ".join(
            f"{name}={float(value.item()) / steps:.4f}"
            for name, value in sorted(self._agreement_sums.items())
        )
        print(
            f"[EV] agree task={self.current_task} steps={int(steps)} "
            f"centering={self.centering_mode} {parts}",
            flush=True,
        )
        self._agreement_sums.clear()
        self._agreement_steps = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _readout_geometry_debug(self) -> None:
        """Print per-class readout norms at consolidation (``WOE_EV_DEBUG=1``).

        The failure mode the ``1/(K-1)`` weighting exists to prevent is a net
        shrinkage of the readout rows, and its cousin is a row collapsing to zero
        with a negative bias (which satisfies every non-target term at once).
        Both are visible here and in nothing else the module prints: a falling
        ``w_l2`` mean is shrinkage, a row whose norm collapses while its bias goes
        negative is the degenerate escape.
        """
        if os.environ.get("WOE_EV_DEBUG") != "1" or self.current_task is None:
            return
        current = self._current_task_class_indices(self.current_task, self._device())
        if current.numel() == 0:
            return
        readout = self.net.model.fc
        norms = readout.weight[current].detach().norm(dim=1)
        biases = readout.bias[current].detach()
        print(
            f"[EV] readout task={self.current_task} k={current.numel()} "
            f"w_l2_mean={float(norms.mean().item()):.4f} "
            f"w_l2_min={float(norms.min().item()):.4f} "
            f"w_l2_max={float(norms.max().item()):.4f} "
            f"bias_mean={float(biases.mean().item()):.4f} "
            f"bias_min={float(biases.min().item()):.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _dropout_conflict_debug(self, x: torch.Tensor, t: int) -> None:
        """Compare the conflict share with dropout on and off (``WOE_EV_DEBUG=1``).

        ``nn.Dropout(p=0.2)`` sits before the pooling that produces the features
        (``model/resnet1d.py``), so training features carry variance evaluation
        never sees. ``w+`` and ``w-`` are the positive and negative parts of the
        per-feature contributions, so that extra variance pushes more terms across
        zero *in both directions*, inflating conflict at train time only. The
        measured 84.6% conflict share is therefore a training-time figure, and
        part of the pathology an evidential loss aims at may not be present at
        eval. One extra forward, once per task.
        """
        if os.environ.get("WOE_EV_DEBUG") != "1":
            return
        current = self._current_task_class_indices(t, self._device())
        if current.numel() == 0:
            return
        readout = self.net.model.fc
        shares = {}
        for label, dropout_on in (("train", True), ("eval", False)):
            # `ResNet1D.forward` sets the module's training mode from
            # `bn_training`, so that flag -- not a train() call around the
            # forward, which it overwrites -- is what governs dropout. It also
            # governs BatchNorm, so the running statistics are frozen for the
            # duration by zeroing the momentum: a diagnostic must not move state
            # the CE path owns.
            frozen = [
                (module, module.momentum)
                for module in self.net.model.modules()
                if isinstance(module, nn.BatchNorm1d) and dropout_on
            ]
            for module, _ in frozen:
                module.momentum = 0.0
            try:
                features = self.net.forward_features(x, bn_training=dropout_on)
            finally:
                for module, momentum in frozen:
                    module.momentum = momentum
            weights = compute_weights_of_evidence(
                features.detach(),
                readout.weight[current].detach(),
                readout.bias[current].detach(),
                self._mu_for_evidence(),
                centering_mode=self.centering_mode,
            )
            parts = _least_commitment_terms(weights)
            total = float(parts["i2"].mean().item())
            conflict = float(parts["conflict"].mean().item())
            shares[label] = conflict / total if total > 0.0 else 0.0
        print(
            f"[EV] dropout task={t} conflict_share_train={shares['train']:.4f} "
            f"conflict_share_eval={shares['eval']:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _lc_debug(self, loss_ce: torch.Tensor, lc: torch.Tensor) -> None:
        """Print the LC term against the CE it accompanies (``WOE_LC_DEBUG=1``).

        No hyper-parameter in this module has transferred between mechanisms --
        the useful ``lambda`` has moved by up to three decades every time the
        normaliser or the scale changed -- so the first question about a new one
        is always "how big is the raw term". This answers it in the first few
        steps of a real run instead of by inference from published quantiles.
        Mirrors ``model.eralg4.Net._dbg``'s env-var gate and step cap.
        """
        if os.environ.get("WOE_LC_DEBUG") != "1":
            return
        seen = getattr(self, "_lc_debug_steps", 0)
        if seen >= 12:
            return
        self._lc_debug_steps = seen + 1
        ce_value = float(loss_ce.item())
        lc_value = float(lc.item())
        scale = "kappa_mean" if self.lc_term == "kappa" else "raw_over_j2"
        print(
            f"[LC] step={seen} ce={ce_value:.4f} term={self.lc_term} "
            f"{scale}={lc_value:.6e} lambda={self.lc_lambda:g} "
            f"penalty={self.lc_lambda * lc_value:.4f} "
            f"ratio={self.lc_lambda * lc_value / max(ce_value, 1e-12):.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _accumulate_term_split(
        self, features: torch.Tensor, class_indices: torch.Tensor
    ) -> None:
        """Accumulate the ``logit`` / ``conflict`` split of ``I_2`` (``WOE_LC_DEBUG=1``).

        ``I_2 = ||z'||^2 + 2 sum_k w+_k w-_k`` splits into a confidence-penalty
        half and a DS conflict half, and measured over the first steps of task 0
        the conflict half was **98.9%** of the total — so what looked like a
        confidence penalty was almost entirely a conflict penalty. That figure is
        a property of a near-random readout, though, and the logit half is
        ``(w+ - w-)^2``, which grows precisely as the model becomes decisive. The
        split late in training is the open question.

        Accumulated here rather than sampled at consolidation because
        consolidation has no batch in hand, and accumulated in the *importance*
        path rather than the penalty path for two reasons: it runs on every
        ``woe_si`` run, including ``woe_lc_lambda=0``, and a penalty on either
        half would otherwise be moving the balance it is being measured by.

        Sums stay on-device 0-dim tensors and are only synced at consolidation,
        so the per-step cost is one extra ``(batch, K, J)`` tensor and no GPU
        stall. Entirely inert when the env var is unset.

        Args:
            features: Penultimate features of the current batch (detached here).
            class_indices: The active columns the importance scalar spans.
        """
        if os.environ.get("WOE_LC_DEBUG") != "1":
            return
        readout = self.net.model.fc
        weights = compute_weights_of_evidence(
            features.detach(),
            readout.weight[class_indices].detach(),
            readout.bias[class_indices].detach(),
            self._mu_for_evidence(),
            centering_mode=self.centering_mode,
        )
        parts = _least_commitment_terms(
            weights, conflict_weighting=self.conflict_weighting
        )
        divisor = float(features.shape[1] * features.shape[1])
        # The task-end average is a *post-hoc* difficulty readout and so competes
        # with the diagonal F1, which is already available by then. An `early`
        # window over the first WOE_SPLIT_EARLY steps is the version that could
        # actually drive an allocation decision, because it is available before
        # the capacity is spent. Both are accumulated so one run answers whether
        # the early signal predicts the task's eventual diagonal.
        early_window = int(os.environ.get("WOE_SPLIT_EARLY", "32"))
        for name in ("logit", "conflict"):
            value = parts[name].mean() / divisor
            if name in self._term_split_sums:
                self._term_split_sums[name] = self._term_split_sums[name] + value
            else:
                self._term_split_sums[name] = value
            if self._term_split_steps < early_window:
                key = f"early_{name}"
                if key in self._term_split_sums:
                    self._term_split_sums[key] = self._term_split_sums[key] + value
                else:
                    self._term_split_sums[key] = value
        self._term_split_steps += 1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _accumulate_gradient_geometry(self) -> None:
        """Per-window norms and cosines of the importance gradients.

        The ``[LC] split`` shares measure the *value* of each half of ``I_2``,
        which is the wrong quantity for anything about ``Omega``: the path
        integral accumulates ``h . delta``, so a half can dominate the value and
        still contribute little gradient. This measures the gradient side --
        ``||h||`` per scalar, and the cosine between every pair -- at exactly the
        windows the path integral samples.

        The cosine here is the *instantaneous* collinearity, distinct from the
        agreement between accumulated ``Omega`` fields: two gradient fields can
        point the same way at every step and still integrate to different
        importance if their magnitudes differ, and can point differently step to
        step yet integrate to the same thing. Both numbers are needed to say
        which.

        Cosines are averaged over windows rather than formed from summed dot
        products, because the question is whether the fields align *at each
        step*; a ratio of sums would let a handful of large-gradient windows
        answer it. Requires ``WOE_LC_DEBUG=1`` and at least one shadow scalar.
        """
        if os.environ.get("WOE_LC_DEBUG") != "1" or not self._shadow_scalars:
            return
        # "live" rather than the scalar's own name so shadowing the tracked
        # scalar does not collide with it -- and so `cos_live_i2` is available
        # as a self-check that must read exactly 1.0.
        fields: Dict[str, Dict[str, torch.Tensor]] = {"live": self._window_h}
        fields.update(self._shadow_h)
        names = [n for n in fields if fields[n]]
        norms = {}
        for name in names:
            total = sum(float(g.pow(2).sum().item()) for g in fields[name].values())
            norms[name] = total**0.5
            self._grad_stat_sums[f"norm_{name}"] = (
                self._grad_stat_sums.get(f"norm_{name}", 0.0) + norms[name]
            )
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                dot = sum(
                    float((fields[a][k] * fields[b][k]).sum().item())
                    for k in fields[a]
                    if k in fields[b]
                )
                cos = dot / max(norms[a] * norms[b], 1e-30)
                self._grad_stat_sums[f"cos_{a}_{b}"] = (
                    self._grad_stat_sums.get(f"cos_{a}_{b}", 0.0) + cos
                )
        self._grad_stat_steps += 1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _gradient_geometry_debug(self) -> None:
        """Print the per-task gradient geometry and reset it."""
        if not self._grad_stat_steps:
            return
        steps = float(self._grad_stat_steps)
        norms = {
            k[len("norm_") :]: v / steps
            for k, v in self._grad_stat_sums.items()
            if k.startswith("norm_")
        }
        parts = " ".join(f"|h_{k}|={v:.4e}" for k, v in sorted(norms.items()))
        halves = norms.get("logit", 0.0) + norms.get("conflict", 0.0)
        share = norms.get("conflict", 0.0) / halves if halves > 0.0 else 0.0
        cosines = " ".join(
            f"{k[len('cos_') :]}={v / steps:.4f}"
            for k, v in sorted(self._grad_stat_sums.items())
            if k.startswith("cos_")
        )
        print(
            f"[LC] grad task={self.current_task} steps={int(steps)} {parts} "
            f"conflict_grad_share={share:.4f} {cosines}",
            flush=True,
        )
        self._grad_stat_sums.clear()
        self._grad_stat_steps = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _term_split_debug(self) -> None:
        """Print the per-task mean split and reset it (``WOE_LC_DEBUG=1``).

        One line per consolidation, so ten lines say whether the conflict half's
        early dominance survives a task's worth of training. Ratio of sums rather
        than mean of ratios: the per-step ratio is unstable when both halves are
        small, and the total is what the penalty actually charges.
        """
        if os.environ.get("WOE_LC_DEBUG") != "1" or self._term_split_steps == 0:
            return
        steps = float(self._term_split_steps)
        logit = float(self._term_split_sums["logit"].item()) / steps
        conflict = float(self._term_split_sums["conflict"].item()) / steps
        total = logit + conflict
        share = conflict / total if total > 0.0 else 0.0
        early_steps = min(steps, float(os.environ.get("WOE_SPLIT_EARLY", "32")))
        early_logit = float(self._term_split_sums["early_logit"].item()) / early_steps
        early_conf = float(self._term_split_sums["early_conflict"].item()) / early_steps
        early_total = early_logit + early_conf
        early_share = early_conf / early_total if early_total > 0.0 else 0.0
        print(
            f"[LC] split task={self.current_task} steps={int(steps)} "
            f"i2={total:.6e} conflict={conflict:.6e} logit={logit:.6e} "
            f"conflict_share={share:.4f} early_steps={int(early_steps)} "
            f"early_conflict_share={early_share:.4f}",
            flush=True,
        )
        self._term_split_sums.clear()
        self._term_split_steps = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _omega_debug(self, task_total: float = 0.0) -> None:
        """Print cumulative ``Omega`` at consolidation (``WOE_LC_DEBUG=1``).

        There is a genuine interaction between the Least-Commitment term and the
        anchor it is usually run alongside, and this is what makes it visible.
        The path integral accumulates ``h_i * delta_i`` and consolidation keeps
        only the positive part (``woe_omega_transform='relu'``), i.e. it protects
        parameters that *raised* ``I_2``. The LC term pushes ``I_2`` down, so it
        moves mass out of the positive part -- with ``woe_lc_lambda`` on, the
        anchor is weaker at the same ``woe_lambda``, and an LC arm is therefore
        not a clean 2x2 cell against an anchor-only one. Total Omega per task
        quantifies how much weaker.

        ``task_total`` is the mass *this* task contributed before it was combined
        into the cumulative buffer. Read against ``total_omega`` it gives the
        sum/max mass ratio needed to re-centre ``woe_lambda`` when switching
        ``woe_omega_accum``: under ``"sum"`` the cumulative total is by
        construction the running sum of the per-task ones, and under ``"max"``
        the gap between them is exactly the saturation the cautious rule removes.
        ``nonzero_frac`` is reported alongside because A6 established that lambda
        tracks the *count* of anchored parameters rather than their mass; the
        cautious rule is designed to leave that count untouched (a max of
        non-negatives is non-zero wherever any term is), and this is the check.

        Args:
            task_total: Summed pre-combination ``Omega`` of the task just
                consolidated, or ``0.0`` if the caller did not compute it.
        """
        if os.environ.get("WOE_LC_DEBUG") != "1":
            return
        buffers = [
            getattr(self, f"{self._param_to_key[name]}_woe_omega")
            for name in self._tracked_names
        ]
        total = sum(float(buf.sum().item()) for buf in buffers)
        counted = sum(int((buf > 0).sum().item()) for buf in buffers)
        numel = sum(int(buf.numel()) for buf in buffers)
        print(
            f"[LC] consolidated task={self.current_task} total_omega={total:.6e} "
            f"task_omega={task_total:.6e} accum={self.omega_accum} "
            f"nonzero_frac={counted / max(1, numel):.4f} "
            f"delta_rms={(self._delta_sq_sum / max(1, self._delta_numel)) ** 0.5:.6e} "
            f"delta_contrib={self._delta_contrib_sum / max(1, self._delta_numel):.4f} "
            f"delta_lt_xi_frac={self._delta_floored / max(1, self._delta_numel):.4f} "
            f"lc_lambda={self.lc_lambda:g}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _capture_importance_gradient(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> None:
        """Backward the importance scalar into a scratch buffer.

        Stores ``h_i = d(scalar)/dtheta_i`` and a snapshot of ``theta_i`` at the
        start of the importance window. ``torch.autograd.grad`` is used so
        parameter ``.grad`` fields (owned by the CE optimiser step) are left
        untouched. Which scalar is tracked is set by ``woe_importance_scalar``;
        ``"i2"`` is the DS information content that defines the method.
        """
        info_content = self._importance_scalar(x, y, t)
        params = [self._tracked_params[name] for name in self._tracked_names]
        grads = torch.autograd.grad(
            info_content, params, retain_graph=False, allow_unused=True
        )
        for name, param, grad in zip(self._tracked_names, params, grads):
            self._window_h[name] = (
                torch.zeros_like(param) if grad is None else grad.detach().clone()
            )
            self._window_p_start[name] = param.detach().clone()

    # ------------------------------------------------------------------
    def _accumulate_path_integral(self) -> None:
        """omega^t += h_i * delta_i over the window (Eq: SI path integral)."""
        for name in self._tracked_names:
            if name not in self._window_h:
                continue
            param = self._tracked_params[name]
            key = self._param_to_key[name]
            w_buf = getattr(self, f"{key}_woe_w")
            delta = param.detach() - self._window_p_start[name]
            w_buf.add_(self._window_h[name] * delta)
            # Same window, same delta -- only the gradient field differs.
            for scalar, grads in self._shadow_h.items():
                if name in grads:
                    self._shadow_buffer("w", scalar, name).add_(grads[name] * delta)
        self._accumulate_gradient_geometry()
        self._window_h.clear()
        self._window_p_start.clear()
        self._shadow_h.clear()

    # ------------------------------------------------------------------
    def _consolidate_current_task(self) -> None:
        """End-of-task consolidation: fold omega^t into the cumulative Omega.

        ``Omega_i^t = relu(omega_i^t) / (delta_total_i^2 + xi)`` -- ``relu``
        because we protect only parameters that *built* committed evidence
        (positive path-integral contribution). The anchor and per-task
        accumulator are reset for the next task, as is the feature mean ``mu``.

        In ``"channel"`` mode each multi-dim ``Omega`` buffer is collapsed to a
        per-output-channel scalar (mean over the within-filter dims) and broadcast
        back, so the quadratic anchor protects whole filters rather than single
        weights. In ``"output"`` mode the path integral is unused; instead a frozen
        teacher snapshot is taken here for the evidence-distillation penalty.
        """
        if self.current_task is None:
            return
        # Mass this task alone contributed, before it is combined. Together with
        # the cumulative total printed by _omega_debug this gives the sum/max
        # mass ratio in one run, which is what woe_lambda has to be re-centred by
        # when switching accumulation rules (no lambda in this project has ever
        # transferred across a change of Omega scale).
        task_total = 0.0
        trace_mass = os.environ.get("WOE_LC_DEBUG") == "1"
        self._delta_sq_sum = 0.0
        self._delta_contrib_sum = 0.0
        self._delta_floored = 0
        self._delta_numel = 0
        # WOE_OMEGA_DUMP=<dir> writes the per-task path integral to disk before
        # it is combined. One `sum` run then yields the *counterfactual* max-Omega
        # on an identical trajectory, which is the only way to compare the two
        # accumulation rules without the confound that the arms diverge after
        # task 0. Cheap: one flat float32 vector per task per run.
        dump_dir = os.environ.get("WOE_OMEGA_DUMP")
        dump_task = [] if dump_dir else None
        dump_parts = (
            {"numerator": [], "delta_sq": [], "signed": []} if dump_dir else None
        )
        dump_shadow_signed: Dict[str, list] = (
            {s: [] for s in self._shadow_scalars} if dump_dir else {}
        )
        normalised = self.omega_accum in _OMEGA_NORMALISED
        # Pass 1: build this task's path integral for every buffer. Held rather
        # than combined immediately because the normalising constant is the total
        # over *all* parameters, which is not known until the pass completes.
        per_name = {}
        raw_total = 0.0
        for name in self._tracked_names:
            param = self._tracked_params[name]
            key = self._param_to_key[name]
            prev = getattr(self, f"{key}_woe_prev")
            delta_total = param.detach() - prev
            w_buf = getattr(self, f"{key}_woe_w")
            if self.omega_transform == "displacement":
                # h == 1, so the path integral telescopes to the net displacement.
                per_name[name] = delta_total.abs().clone()
            elif self.omega_transform == "uniform":
                # Bypass numerator *and* denominator: every parameter gets the
                # same per-task importance, so Omega counts tasks and nothing
                # else. Deliberately not `ones / (delta^2 + xi)`, which would
                # smuggle the displacement back in.
                per_name[name] = torch.ones_like(param)
            else:
                projected = (
                    w_buf.abs() if self.omega_transform == "abs" else torch.relu(w_buf)
                )
                per_name[name] = projected / (delta_total.pow(2) + self.xi)
            if dump_parts is not None:
                # Numerator and squared displacement kept *separately* so Omega
                # can be recomputed offline at any xi. Without them the dump only
                # carries the post-denominator quantity and the floor cannot be
                # undone, which is the difference between diagnosing the
                # saturation and merely observing it.
                dump_parts["numerator"].append(
                    projected.detach().flatten().float().cpu()
                )
                dump_parts["delta_sq"].append(
                    delta_total.detach().pow(2).flatten().float().cpu()
                )
                # The path integral *before* abs/relu. `abs` discards the sign,
                # so two scalars whose importance gradients are anti-aligned can
                # still produce positively correlated Omega; recovering the sign
                # is the only way to tell that apart offline, and it is what
                # separates "the scalars agree" from "the transform hides that
                # they disagree".
                dump_parts["signed"].append(w_buf.detach().flatten().float().cpu())
            for scalar in self._shadow_scalars:
                shadow_w = self._shadow_buffer("w", scalar, name)
                if dump_dir:
                    dump_shadow_signed[scalar].append(
                        shadow_w.detach().flatten().float().cpu()
                    )
                shadow_projected = (
                    torch.relu(shadow_w)
                    if self.omega_transform == "relu"
                    else shadow_w.abs()
                )
                shadow_task = shadow_projected / (delta_total.pow(2) + self.xi)
                shadow_omega = self._shadow_buffer("omega", scalar, name)
                # Mirror the live accumulation rule so the shadow field is the
                # counterfactual Omega, not a differently-combined one.
                if self.omega_accum in ("max", "max_norm"):
                    torch.maximum(shadow_omega, shadow_task, out=shadow_omega)
                else:
                    shadow_omega.add_(shadow_task)
                shadow_w.zero_()
            if normalised or trace_mass:
                raw_total += float(per_name[name].sum().item())
            if trace_mass:
                # SI's denominator only corrects for path length while
                # delta^2 >> xi. Under a strong anchor delta collapses and the
                # denominator floors at xi, so the correction silently switches
                # off exactly when it is most needed. This measures whether that
                # is happening: the fraction of parameters already in the floored
                # regime, and the RMS displacement to compare against sqrt(xi).
                delta_sq = delta_total.pow(2)
                self._delta_sq_sum += float(delta_sq.sum().item())
                # `delta_sq < xi` is NOT a floor: at delta^2 = xi the displacement
                # still supplies half the denominator. The honest quantity is the
                # share it supplies, delta^2 / (delta^2 + xi), which is 0 when
                # fully floored and 1 when undamped and needs no threshold. The
                # count is kept alongside it only as the (loose) upper bound it
                # always was.
                self._delta_contrib_sum += float(
                    (delta_sq / (delta_sq + self.xi)).sum().item()
                )
                self._delta_floored += int((delta_sq < self.xi).sum().item())
                self._delta_numel += int(delta_sq.numel())
        if normalised:
            self._omega_raw_total += raw_total
            scale = 1.0 / max(raw_total, 1e-30)
            for tensor in per_name.values():
                tensor.mul_(scale)
        for name in self._tracked_names:
            param = self._tracked_params[name]
            key = self._param_to_key[name]
            prev = getattr(self, f"{key}_woe_prev")
            omega = getattr(self, f"{key}_woe_omega")
            w_buf = getattr(self, f"{key}_woe_w")
            task_omega = per_name[name]
            if trace_mass:
                task_total += float(task_omega.sum().item())
            if dump_task is not None:
                dump_task.append(task_omega.detach().flatten().float().cpu())
            if self.omega_accum in ("max", "max_norm"):
                # Cautious rule: the canonical weight functions combine by
                # minimum, hence the weights of evidence by maximum. See
                # _OMEGA_ACCUMS for the derivation.
                torch.maximum(omega, task_omega, out=omega)
            else:
                omega.add_(task_omega)
            if self.reg_level == "channel" and omega.dim() >= 2:
                # Collapse to per-output-channel (dim 0 is the filter/class axis),
                # mirroring model.eucr_consolidation.to_channel. Idempotent across
                # tasks: an already-channel-uniform Omega stays uniform.
                reduce_dims = tuple(range(1, omega.dim()))
                per_channel = omega.mean(dim=reduce_dims, keepdim=True)
                omega.copy_(per_channel.expand_as(omega))
            prev.copy_(param.detach())
            w_buf.zero_()
        if normalised:
            # Put the combined Omega back on the raw-sum arm's mass scale, so a
            # lambda grid transfers between arms and the comparison isolates the
            # rule rather than a global rescaling that lambda would absorb.
            combined = sum(
                float(getattr(self, f"{self._param_to_key[n]}_woe_omega").sum().item())
                for n in self._tracked_names
            )
            factor = self._omega_raw_total / max(combined, 1e-30)
            for n in self._tracked_names:
                getattr(self, f"{self._param_to_key[n]}_woe_omega").mul_(factor)
        # Cap the tail once every buffer for this task has been folded in, so the
        # quantile is taken over the final cumulative Omega.
        self._winsorise_omega()
        if dump_task is not None:
            cumulative = [
                getattr(self, f"{self._param_to_key[name]}_woe_omega")
                .detach()
                .flatten()
                .float()
                .cpu()
                for name in self._tracked_names
            ]
            shapes = [
                (name, tuple(self._tracked_params[name].shape))
                for name in self._tracked_names
            ]
            out = Path(dump_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "task": int(self.current_task),
                    "accum": self.omega_accum,
                    "task_omega": torch.cat(dump_task),
                    "numerator": torch.cat(dump_parts["numerator"]),
                    "delta_sq": torch.cat(dump_parts["delta_sq"]),
                    "signed": torch.cat(dump_parts["signed"]),
                    "shadow_signed": {
                        scalar: torch.cat(parts)
                        for scalar, parts in dump_shadow_signed.items()
                    },
                    "xi": float(self.xi),
                    "omega": torch.cat(cumulative),
                    "shadow_omega": {
                        scalar: torch.cat(
                            [
                                self._shadow_buffer("omega", scalar, name)
                                .detach()
                                .flatten()
                                .float()
                                .cpu()
                                for name in self._tracked_names
                            ]
                        )
                        for scalar in self._shadow_scalars
                    },
                    "shapes": shapes,
                },
                out / f"omega_task{int(self.current_task):02d}.pt",
            )
        self._omega_debug(task_total)
        self._term_split_debug()
        self._gradient_geometry_debug()
        self._score_agreement_debug()
        self._readout_geometry_debug()
        # Output mode distils a frozen end-of-task snapshot; capture it (and the
        # feature mean it must centre with) *before* the per-task stats are reset.
        # The LwF logit-distillation term needs the same teacher, and is available
        # alongside any reg_level, so either consumer triggers the snapshot.
        if (
            self.reg_level == "output"
            or self.lwf_lambda != 0.0
            or self.evidence_distill_lambda != 0.0
        ):
            self._snapshot_teacher()
        # PR-3 diagnostic: how far this task's EMA ended up from its true mean.
        # Emitted before the reset below, while both mu are still populated.
        self._mu_divergence_debug()
        # Reset running feature stats for the next task.
        self.woe_feature_mean.zero_()
        self.woe_feature_mean_initialised.zero_()
        self.woe_mu_frozen.zero_()
        self.woe_mu_frozen_set.zero_()
        self._step_in_task = 0
        self._window_h.clear()
        self._window_p_start.clear()

    # ------------------------------------------------------------------
    def _lwf_distillation_loss(
        self, student_logits: torch.Tensor, x: torch.Tensor, t: int
    ) -> torch.Tensor:
        """Learning-without-Forgetting logit distillation on previous classes.

        Temperature-scaled KL between the student's and a frozen teacher's softmax
        over the columns of completed tasks, scaled by ``T^2`` so its gradient
        magnitude is comparable to the cross-entropy it accompanies. This mirrors
        ``model.lwf.Net._distillation_loss`` exactly, so a WoE-SI run with
        ``woe_lambda=0`` and ``woe_lwf_lambda>0`` is an LwF control sharing this
        module's optimiser, masking and BatchNorm handling.

        Orthogonal to the ``I_2`` anchor by construction: the anchor constrains
        *parameters* via the DS path integral, this constrains the *function* at
        the readout. Both can be active at once, which is what makes a 2x2
        stacking test possible in one code path.

        Args:
            student_logits: Unmasked class logits of the live network.
            x: The current batch, re-run through the frozen teacher.
            t: Current task index. Returns 0 on the first task.

        Returns:
            Scalar distillation loss; exactly 0 before a teacher exists.
        """
        if self.teacher is None:
            return torch.zeros(1, device=student_logits.device)
        previous = self._previous_class_indices(t, student_logits.device)
        if previous.numel() == 0:
            return torch.zeros(1, device=student_logits.device)

        student_previous = student_logits.index_select(1, previous)
        with torch.no_grad():
            # bn_training=True scores the teacher on the *current batch's*
            # statistics rather than the running statistics it froze with. That
            # matches `model.lwf` (which calls `self.teacher(x)`, and ResNet1D's
            # forward runs `self.model.train(bn_training)`), and it is the right
            # choice here rather than an accident: consecutive tasks are different
            # radar datasets, so a teacher normalised with the previous task's
            # statistics is evaluated under distribution shift and its targets are
            # correspondingly degraded. Measured on task 1, freezing the
            # statistics instead drops the distillation loss from 1.059 to 0.475
            # and costs 0.068 of final macro recall (0.4766 -> 0.4087).
            #
            # `_snapshot_teacher` zeroes the teacher's BatchNorm momentum, so this
            # forward uses batch statistics *without* mutating the frozen running
            # buffers -- unlike `model.lwf`, where each distillation pass updates
            # them. Those buffers are never read in this mode, so the numerics
            # match `model.lwf` exactly; the teacher simply stays genuinely frozen.
            teacher_logits = self.teacher(x, bn_training=True)
            teacher_probs = torch.softmax(
                teacher_logits.index_select(1, previous) / self.lwf_temperature,
                dim=1,
            )
        student_log_probs = torch.log_softmax(
            student_previous / self.lwf_temperature, dim=1
        )
        return self.lwf_kl(student_log_probs, teacher_probs) * (self.lwf_temperature**2)

    # ------------------------------------------------------------------
    def _snapshot_teacher(self) -> None:
        """Freeze the current net + feature mean as the distillation teacher.

        Mirrors ``model.lwf.Net._update_teacher``: a ``deepcopy`` in eval mode with
        gradients disabled. The feature mean is snapshotted so the teacher centres
        its evidence with the statistics of the task it was frozen on.
        """
        self.teacher = copy.deepcopy(self.net)
        self.teacher.to(self._device())
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        # Zero the BatchNorm momentum so a `bn_training=True` teacher forward
        # normalises with batch statistics but leaves the running buffers exactly
        # where they were frozen: the update is
        # `(1 - momentum) * running + momentum * batch`. Without this, every
        # distillation pass would drift the "frozen" reference toward the current
        # task. `.eval()` alone cannot achieve it -- ResNet1D.forward calls
        # `self.model.train(bn_training)` and overrides module mode.
        for module in self.teacher.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.momentum = 0.0
        # Same override, other consequence: `model.train(True)` also switches the
        # backbone's four trunk dropout modules on, so the distillation target is
        # resampled every step. Zeroing `p` on the *copy* removes that noise and
        # leaves batch-statistic normalisation untouched -- the two effects share
        # a switch in ResNet1D but are separable here. The student keeps its own
        # dropout; only the frozen reference becomes deterministic.
        if self.teacher_dropout == "disable":
            for module in self.teacher.modules():
                if isinstance(module, nn.modules.dropout._DropoutNd):
                    module.p = 0.0
        self.teacher_feature_mean = self._mu_for_evidence().detach().clone()

    # ------------------------------------------------------------------
    def _surrogate_loss(self) -> torch.Tensor:
        """Quadratic anchor penalty ``sum_i Omega_i * (theta_i - theta_i^*)^2``.

        The ``lambda/2`` scaling of the spec is folded into ``woe_lambda`` at the
        call site (``self.woe_lambda * self._surrogate_loss()``), matching SI's
        ``si_c`` convention. Returns exactly ``0`` on the first task because
        ``Omega`` is all zeros until the first consolidation.
        """
        device = self._device()
        loss = torch.zeros(1, device=device)
        for name in self._tracked_names:
            param = self._tracked_params[name]
            key = self._param_to_key[name]
            omega = getattr(self, f"{key}_woe_omega")
            prev = getattr(self, f"{key}_woe_prev")
            loss = loss + (omega * (param - prev).pow(2)).sum()
        return loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_proximal_anchor(self) -> None:
        """Apply the quadratic anchor as a closed-form post-step update.

        The loss form takes an explicit gradient step on ``lambda*Omega*(theta -
        theta*)^2``, whose curvature is ``k = 2*lambda*Omega``. Explicit descent on
        a quadratic is stable only while ``lr*k < 2``; the path integral is heavy
        tailed enough that a few parameters land far outside that window, diverge,
        and -- because ``clip_grad_norm_`` rescales every gradient by one global
        scalar -- drag the whole network's effective learning rate down with them.

        The proximal (backward-Euler) form evaluates the anchor gradient at the
        *new* point, ``theta_new = theta+ - lr*2*lambda*Omega*(theta_new -
        theta*)``, which solves in closed form to a convex combination::

            b = 2 * lr * lambda * Omega
            theta_new = (theta+ + b * theta*) / (1 + b)
                      = (1 - a) * theta+ + a * theta*,   a = b/(1+b) in [0, 1)

        Because ``a`` saturates at 1 for any ``Omega``, the update can never
        overshoot the anchor: ``Omega -> 0`` leaves the parameter free and
        ``Omega -> inf`` pins it exactly to ``theta*``. It is also a no-op on the
        first task, where ``Omega`` is still all zeros.
        """
        learning_rate = float(self.opt.param_groups[0]["lr"])
        scale = 2.0 * learning_rate * self.woe_lambda
        if scale == 0.0:
            return
        for name in self._tracked_names:
            param = self._tracked_params[name]
            key = self._param_to_key[name]
            omega = getattr(self, f"{key}_woe_omega")
            prev = getattr(self, f"{key}_woe_prev")
            b = omega * scale
            param.copy_((param + b * prev) / (1.0 + b))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _winsorise_omega(self) -> None:
        """Cap cumulative ``Omega`` at a global quantile across all buffers.

        The per-parameter path integral is heavy tailed -- in practice a handful
        of weights accumulate importance orders of magnitude above the 99th
        percentile, which is what pushes the loss-form anchor outside its
        stability window. Capping at ``woe_omega_winsorise`` bounds the curvature
        while leaving the relative ordering of every other parameter untouched.

        The quantile is taken over the concatenation of all tracked buffers, not
        per tensor, because the tail is concentrated in one layer -- a per-tensor
        cap would simply rescale that layer's own outliers against each other.
        ``kthvalue`` is used rather than ``torch.quantile`` so the computation is
        exact regardless of parameter count (``quantile`` caps out around 2**24).
        """
        if self.omega_winsorise <= 0.0:
            return
        buffers = [
            getattr(self, f"{self._param_to_key[name]}_woe_omega")
            for name in self._tracked_names
        ]
        if not buffers:
            return
        flat = torch.cat([buf.reshape(-1) for buf in buffers])
        index = max(
            1, min(flat.numel(), int(round(self.omega_winsorise * flat.numel())))
        )
        cap = torch.kthvalue(flat.float(), index).values
        for buf in buffers:
            buf.clamp_(max=cap.to(buf.dtype))

    # ------------------------------------------------------------------
    def _evidence_distillation_loss(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Output-level penalty: drift of the DS evidence vs. a frozen teacher.

        Penalises how far the current net's per-class total evidence
        ``(w_plus, w_minus)`` on previously-seen classes has moved from the frozen
        end-of-task teacher's, on the current batch (and, when the detector replay
        buffer is enabled and non-empty, on a replay batch too)::

            L = mean_b  sum_{k in old}  (w+_s - w+_t)^2 + (w-_s - w-_t)^2

        Student and teacher are centred with the **same** feature mean -- the
        teacher's snapshot. Centring is a choice of reference point for the
        Least-Commitment decomposition (``w_jk = beta_kj (phi_j - mu_j) +
        beta_0k/J``), so scoring the two networks against different ``mu`` leaves a
        reference shift inside the measured "drift": an *unchanged* network then
        scores a non-zero penalty. Measured on the 10-task TIL run, distilling a
        network against an exact copy of itself accounted for 22-70% of the total
        penalty before this was fixed. Using the teacher's ``mu`` for both also
        keeps the target fixed for the duration of the task, where the live EMA
        would drift under the student even though the teacher is frozen.

        ``J^2``-normalised to match the ``I_2`` importance signal's scale (see
        ``_compute_information_content``). Returns exactly ``0`` before the first
        teacher exists, mirroring the SI penalty being 0 on the first task. The
        running feature mean is EMA-updated here so output mode (which skips the
        importance path) still tracks ``mu`` -- not to centre this comparison, but
        so the *next* ``_snapshot_teacher`` inherits the statistics of the task it
        was frozen on.
        """
        features = self.net.forward_features(x, bn_training=False)
        self._update_feature_mean(features.detach())
        if self.teacher is None:
            return torch.zeros(1, device=features.device)
        active = self._previous_class_indices(t, features.device)
        if active.numel() == 0:
            return torch.zeros(1, device=features.device)

        distill_x = x

        # Both networks are centred with the teacher's mu: a shared reference is
        # what makes the difference measure evidence drift rather than a shift in
        # centring statistics. See the docstring.
        centring_mean = self.teacher_feature_mean
        student_w = self._weights_of_evidence(
            features, self.net.model.fc, active, centring_mean
        )
        w_plus_s, w_minus_s = per_class_total_evidence(student_w)
        with torch.no_grad():
            teacher_features = self.teacher.forward_features(
                distill_x, bn_training=False
            )
            teacher_w = self._weights_of_evidence(
                teacher_features,
                self.teacher.model.fc,
                active,
                centring_mean,
            )
            w_plus_t, w_minus_t = per_class_total_evidence(teacher_w)

        student = self._to_penalty_scale(w_plus_s, w_minus_s)
        teacher = self._to_penalty_scale(w_plus_t, w_minus_t)
        drift = self._drift_terms(student, teacher)
        return drift.sum(dim=1).mean() / self._evidence_normaliser(features.shape[1])

    # ------------------------------------------------------------------
    def _drift_terms(
        self,
        student: Tuple[torch.Tensor, torch.Tensor],
        reference: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Per-class penalty between a student and a reference evidence pair.

        Symmetric by default -- any movement away from the reference is charged.
        Under ``woe_evidence_asymmetric`` only *deterioration* is charged: support
        for the class falling, or evidence against it rising. Improvement is then
        free, so the term never competes for capacity the old tasks do not need.

        The squared hinge is C^1 (its derivative ``2*relu(.)`` is continuous at the
        kink), so the asymmetric form is no harder to optimise than the symmetric
        one. Note it also removes the upper arm that pinned the evidence scale --
        pair it with ``woe_evidence_scale='belief'``, which is bounded, or the
        constraint becomes satisfiable by inflating the readout.

        WARNING -- asymmetry is only sound where the reference was recorded on the
        *same* inputs it is scored on. That holds for ``woe_si_replay``'s decay
        penalty, which re-evaluates each stored item against its own snapshot. It
        does **not** hold in output mode, where the frozen teacher is scored on
        *current-task* data: there the second arm reads "evidence against an old
        class must not rise on new-task inputs", which forbids exactly what the
        model should be learning, since new-task samples genuinely are not members
        of the old classes. Measured on the 10-task TIL run, output mode with
        ``woe_evidence_asymmetric`` at lambda=1 gives BWT -0.4383 against naive
        fine-tuning's ~-0.37, i.e. worse retention than no penalty at all, while
        the diagonal stays healthy at 0.6221. Use the symmetric form in output
        mode; "match the teacher" is direction-neutral and carries no such
        assumption.

        Args:
            student: ``(w_plus, w_minus)`` of the live network, shape ``(batch, K)``.
            reference: ``(w_plus, w_minus)`` of the frozen teacher or snapshot.

        Returns:
            Per-class penalty with shape ``(batch, K)``.
        """
        student_plus, student_minus = student
        reference_plus, reference_minus = reference
        if self.evidence_asymmetric:
            return torch.relu(reference_plus - student_plus).pow(2) + torch.relu(
                student_minus - reference_minus
            ).pow(2)
        return (student_plus - reference_plus).pow(2) + (
            student_minus - reference_minus
        ).pow(2)

    # ------------------------------------------------------------------
    def _to_penalty_scale(
        self, w_plus: torch.Tensor, w_minus: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map a ``(w_plus, w_minus)`` pair onto the configured penalty scale.

        Identity under ``woe_evidence_scale='weight'``. Under ``'belief'`` both
        channels go through :func:`evidence_to_belief`, which is monotone -- so a
        stored snapshot can stay in weight space and be converted here.

        The two channels are transformed *separately* rather than combined into
        ``Bel({theta_k})``. That is deliberate: the asymmetric penalty makes two
        independent one-sided statements (support must not fall, counter-evidence
        must not rise), and combining the channels would let a drop in support be
        repaired by suppressing counter-evidence instead.

        Args:
            w_plus: Positive total evidence ``(batch, K)``.
            w_minus: Negative total evidence ``(batch, K)``.

        Returns:
            The pair mapped onto the penalty scale, shapes unchanged.
        """
        if self.evidence_scale != "belief":
            return w_plus, w_minus
        tau = self.evidence_belief_tau
        return evidence_to_belief(w_plus, tau), evidence_to_belief(w_minus, tau)

    # ------------------------------------------------------------------
    def _evidence_normaliser(self, feature_count: int) -> float:
        """Scale divisor for the functional penalties, which differs per scale.

        A squared difference of weights was assumed to be ``O(J^2)``, since
        ``w_plus`` sums up to ``J`` non-negative terms. Measured on the 10-task TIL
        run it is not: ``w_plus`` reaches a median of 3.5 and a max of 11.7, not
        ~512, so the ``J^2`` divisor over-normalises by roughly 4000x and forces a
        correspondingly large ``lambda``. It is kept for the weight scale so
        existing tuned values stay valid. Beliefs are ``O(1)`` and take no divisor.

        Consequence: ``lambda`` does **not** transfer between the two scales.

        Args:
            feature_count: Readout input width ``J``.

        Returns:
            Divisor applied after averaging over the batch.
        """
        if self.evidence_scale == "belief":
            return 1.0
        return float(feature_count * feature_count)

    # ------------------------------------------------------------------
    def _update_feature_mean(self, batch_features: torch.Tensor) -> None:
        """EMA-update the per-task running feature mean ``mu_j``."""
        batch_mean = batch_features.mean(dim=0)
        if not bool(self.woe_feature_mean_initialised.item()):
            self.woe_feature_mean.copy_(batch_mean)
            self.woe_feature_mean_initialised.fill_(True)
        else:
            self.woe_feature_mean.mul_(self.mu_momentum).add_(
                batch_mean, alpha=1.0 - self.mu_momentum
            )

    # ------------------------------------------------------------------
    def _mu_for_evidence(self) -> torch.Tensor:
        """The ``mu`` that centres the weights of evidence, per ``woe_mu_mode``.

        Every site that builds ``w_jk`` must read mu through here, including the
        *diagnostics*. Reading ``woe_feature_mean`` directly in a diagnostic while
        the objective reads the frozen mu would silently report the split of a
        quantity the run is not optimising -- a wrong-number-shaped bug rather
        than a crash, which is the kind this project can least afford.

        Falls back to the EMA if the frozen mu has not been computed yet (only
        reachable before the first pre-pass, e.g. a unit test that calls a
        penalty without stepping ``observe``).
        """
        if self.mu_mode == "frozen_pretask" and bool(self.woe_mu_frozen_set.item()):
            return self.woe_mu_frozen
        return self.woe_feature_mean

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_pretask_mu(self, t: int) -> None:
        """Unweighted mean of ``phi`` over task ``t``'s full training set.

        Run before task ``t``'s first gradient step, with the model exactly as
        task ``t-1`` left it, and held fixed for the whole task.

        The pass must leave the run otherwise untouched, because the whole design
        rests on the two arms being the same trajectory with one knob changed.
        Structurally it already is:

        * ``torch.no_grad()`` -- no graph, no parameter touch.
        * ``bn_training=False`` -- matches every existing WoE call site
          (``_compute_information_content`` et al). Disables dropout *and* stops
          BatchNorm updating its running statistics, which the CE path owns.
        * the loader from ``get_tasks("train")`` is built ``shuffle=False,
          num_workers=0`` (``task_incremental_loader.py:559-563``), so it uses a
          ``SequentialSampler``; ``IQDataGenerator.__getitem__`` does no
          augmentation. Nothing on the path *should* draw randomness.
        * ``get_tasks`` rebuilds loaders from the retained per-task arrays; it does
          **not** touch the loader's one-way ``new_task()`` cursor
          (``task_incremental_loader.py:414``) and does not consume the iterator
          the training loop is using.

        **"Should" was not good enough.** The first full-scale gate ran this pass
        in the `ema` arm and missed the recorded control (0.5224 / 0.4984 /
        -0.0240 against 0.5206 / 0.5008 / -0.0198), while two runs *without* the
        pass agreed with each other bit-identically -- so the pass, not
        nondeterminism and not checkpoint writing, was moving the trajectory. The
        CPU unit tests could not see it: they drive a stub loader and never touch
        CUDA, AMP, or ``IQDataGenerator``.

        So the RNG state is now saved and restored explicitly rather than argued
        to be untouched. That closes the entire RNG-mediated class of divergence
        whatever the specific draw turned out to be, and costs one tensor copy per
        task. The train/eval flag is restored the same way, for the same reason.
        """
        if self._task_loader_fn is None:
            return
        loaders = self._task_loader_fn("train")
        if t < 0 or t >= len(loaders):
            raise ValueError(
                f"pre-task mu: task {t} out of range for {len(loaders)} train loaders"
            )
        device = self._device()
        was_training = self.net.training
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        # Unweighted over *samples*, not over batches: the last batch is short, so
        # averaging batch means would silently up-weight it.
        total = torch.zeros(self.feature_dim, device=device, dtype=torch.float64)
        count = 0
        try:
            for batch in loaders[t]:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                xb = xb.to(device, non_blocking=True)
                features = self.net.forward_features(xb, bn_training=False)
                total += features.detach().double().sum(dim=0)
                count += int(features.shape[0])
        finally:
            self.net.train(was_training)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        if count == 0:
            raise ValueError(f"pre-task mu: task {t} yielded no samples")
        self.woe_mu_frozen.copy_((total / count).to(self.woe_mu_frozen.dtype))
        self.woe_mu_frozen_set.fill_(True)

    # ------------------------------------------------------------------
    def _mu_divergence_debug(self) -> None:
        """How far the EMA reference sits from the task's true mean (PR-3).

        Reports the ratio *and* both raw norms: the June checkpoints show ``||mu||``
        varying 2.5x across tasks (42.6 at task 1 against ~17 elsewhere), so a
        ratio alone would hide which end of that range a task sits at.
        """
        if os.environ.get("WOE_LC_DEBUG") != "1":
            return
        if not bool(self.woe_mu_frozen_set.item()):
            return
        frozen_norm = float(self.woe_mu_frozen.norm().item())
        ema_norm = float(self.woe_feature_mean.norm().item())
        gap = float((self.woe_feature_mean - self.woe_mu_frozen).norm().item())
        ratio = gap / frozen_norm if frozen_norm > 0.0 else float("nan")
        print(
            f"[WOE_MU] task={self.current_task} mode={self.mu_mode} "
            f"rel_gap={ratio:.6f} gap={gap:.6f} "
            f"norm_ema={ema_norm:.6f} norm_frozen={frozen_norm:.6f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _active_class_indices(self, t: int, device: torch.device) -> torch.Tensor:
        """Active output columns: cumulative seen classes (CIL) or task slice (TIL).

        Matches ``utils.misc_utils.apply_task_incremental_logit_mask``, so the DS
        frame ``Theta`` spans the same classes the CE head is actually predicting
        over.
        """
        offset1, offset2 = misc_utils.compute_offsets(t, self.classes_per_task)
        offset2 = min(self.n_outputs, offset2)
        if self.is_cil:
            indices = list(range(0, offset2))
        else:
            indices = list(range(offset1, offset2))
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    def _current_task_class_indices(self, t: int, device: torch.device) -> torch.Tensor:
        """Output columns belonging to task ``t`` itself (plus noise).

        The complement of :meth:`_previous_class_indices` within the active set.
        Identical to :meth:`_active_class_indices` under TIL, where the head is
        masked to the current task anyway; under CIL the active set is
        cumulative and this is the part of it the current task is responsible
        for.
        """
        return misc_utils.current_task_class_indices(
            t,
            self.classes_per_task,
            self.n_outputs,
            device=device,
        )

    # ------------------------------------------------------------------
    def _previous_class_indices(self, t: int, device: torch.device) -> torch.Tensor:
        """Output columns of classes from *completed* tasks ``< t``.

        Used by the output-mode distillation to penalise evidence drift only on
        previously-seen classes (analogous to ``model.lwf``'s previous-class ids).
        The cumulative prior-class span is ``[0, offset1)`` in both TIL and CIL,
        where ``offset1`` is the first column of the current task. Returns an empty
        tensor on the first task.
        """
        offset1, _ = misc_utils.compute_offsets(t, self.classes_per_task)
        offset1 = min(self.n_outputs, offset1)
        indices = list(range(0, offset1))
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    def omega_summary(self) -> Dict[str, Dict[str, float]]:
        """Per-group summary of cumulative ``Omega`` (readout vs backbone).

        Returns a dict with ``"readout"`` (the ``fc`` head) and ``"backbone"``
        (everything else) sub-dicts holding mean / max / sum of ``Omega`` so a
        caller can eyeball that importance concentrates sensibly.
        """
        groups: Dict[str, List[torch.Tensor]] = {"readout": [], "backbone": []}
        for name in self._tracked_names:
            key = self._param_to_key[name]
            omega = getattr(self, f"{key}_woe_omega")
            bucket = "readout" if name.startswith("fc.") else "backbone"
            groups[bucket].append(omega.reshape(-1))
        summary: Dict[str, Dict[str, float]] = {}
        for bucket, tensors in groups.items():
            if not tensors:
                summary[bucket] = {"mean": 0.0, "max": 0.0, "sum": 0.0, "count": 0}
                continue
            flat = torch.cat(tensors)
            summary[bucket] = {
                "mean": float(flat.mean().item()),
                "max": float(flat.max().item()),
                "sum": float(flat.sum().item()),
                "count": int(flat.numel()),
            }
        return summary

    # ------------------------------------------------------------------
    def _compute_offsets(self, task: int) -> Tuple[int, int]:
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return offset1, min(self.n_outputs, offset2)

    # ------------------------------------------------------------------
    def _classification_replay_loss(self, t: int) -> torch.Tensor:
        """Extra classification loss from a rehearsal buffer (hook).

        Base WoE-SI is a pure regularisation method and keeps no rehearsal
        buffer, so this returns exactly ``0``. The ``woe_si_replay`` subclass
        overrides it to add an experience-replay cross-entropy term over a
        reservoir sample of previously-seen batches.

        Args:
            t: Current task index (unused in the base no-op).

        Returns:
            A scalar loss contribution; zero in the base learner.
        """
        del t
        return torch.zeros(1, device=self._device())

    # ------------------------------------------------------------------
    def _store_classification_replay(
        self, x: torch.Tensor, y: torch.Tensor, t: int
    ) -> None:
        """Write the current batch into a rehearsal buffer (hook).

        No-op in base WoE-SI; overridden by ``woe_si_replay`` to feed its
        reservoir buffer.

        Args:
            x: Current input batch.
            y: Current (possibly packed) labels.
            t: Current task index.
        """
        del x, y, t
        return None

    # ------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device


__all__ = [
    "Net",
    "WoeSiConfig",
    "compute_weights_of_evidence",
    "per_class_total_evidence",
    "information_content",
    "least_commitment_penalty",
    "evidential_classification_loss",
]
