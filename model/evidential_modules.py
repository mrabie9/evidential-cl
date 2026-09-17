"""Dempster-Shafer evidential neural modules for the EUCR learner.

This is a self-contained port of the Dempster-Shafer machinery used by the
EUCR-Evidential project, adapted for the La-MAML harness. It provides:

* :class:`Dempster_Shafer_module` -- maps a feature vector to normalised
  Dempster-Shafer mass functions ``[B, n_classes + 1]`` (the last column is the
  ignorance mass ``omega``).
* :class:`DM` -- a decision-making layer turning masses into per-class expected
  utilities (last column kept as ``nu``-scaled ``omega``).
* :class:`EvidentialLoss` -- the BCE-style evidential classification loss with a
  cosine KL warm-up, used to train both the final head and the backbone probes.

The maths matches the original implementation; the only behavioural changes are
(1) :class:`Distance_layer` is vectorised with ``torch.cdist`` instead of a
Python loop over prototypes (numerically identical, much faster) and (2)
:class:`DM` derives its device from the input tensor so the module works after
``model.cuda()`` without storing a fixed device.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_NUMERICAL_EPS = 1e-6
# Floor for logs inside the Dempster combination. Smaller than _NUMERICAL_EPS so
# a genuinely zero prototype mass still vetoes its class rather than being
# rounded up into a visible belief.
_LOG_EPS = 1e-12


def _safe_normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize along ``dim`` with a small epsilon to avoid NaNs from zero sums."""
    denom = tensor.sum(dim=dim, keepdim=True).clamp_min(_NUMERICAL_EPS)
    return tensor / denom


class Distance_layer(nn.Module):
    """Distance from each input to ``n_prototypes`` prototypes.

    ``metric="cosine"`` returns the directional distance ``1 - cos(f, p)`` in
    ``[0, 2]``; ``metric="euclidean"`` returns the squared Euclidean distance.
    Cosine is the default because the backbone feeds ``LayerNorm``-normalised
    features whose L2 norm is ~constant across samples, so squared-Euclidean
    distance is dominated by that constant norm and collapses the input signal
    (feature coefficient-of-variation ~0.44 -> distance ~0.01); cosine reads the
    directional signal LayerNorm preserves and keeps the prototype gradient
    informative.
    """

    def __init__(
        self, n_prototypes: int, n_feature_maps: int, metric: str = "cosine"
    ) -> None:
        super().__init__()
        self.w = nn.Linear(
            in_features=n_feature_maps, out_features=n_prototypes, bias=False
        ).weight
        self.n_prototypes = n_prototypes
        self.metric = str(metric).lower()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, F], self.w: [P, F] -> [B, P] distances.
        if self.metric == "euclidean":
            return torch.cdist(inputs, self.w, p=2).pow(2)
        normalized_inputs = F.normalize(inputs, dim=-1)
        normalized_prototypes = F.normalize(self.w, dim=-1)
        return 1.0 - normalized_inputs @ normalized_prototypes.t()


class DistanceActivation_layer(nn.Module):
    """Turn distances into per-prototype activations in ``[0, 1]``.

    ``activation_norm="max"`` divides by the per-sample maximum activation. That
    is the shipped behaviour and it is the first of three independent reasons the
    fused ignorance mass is dead: it pins the best-matching prototype at
    ``s_p ~ 1``, so its own ignorance ``1 - s_p`` is the ``1e-4/max`` rounding
    artefact (measured: 4.2e-4) no matter how far the input actually is from every
    prototype. It also makes the fusion approximately a nearest-prototype rule,
    since only the argmax term ``log u_{c,argmax}`` is unbounded below.
    ``activation_norm="none"`` leaves ``s_p = alpha_p exp(-gamma_p d_p)``, which is
    already in (0, 1) because ``alpha = sigmoid(xi) < 1``.
    """

    def __init__(
        self,
        n_prototypes: int,
        init_alpha: float = 0.0,
        init_gamma: float = 0.1,
        activation_norm: str = "max",
    ) -> None:
        super().__init__()
        self.eta = nn.Linear(in_features=n_prototypes, out_features=1, bias=False)
        self.xi = nn.Linear(in_features=n_prototypes, out_features=1, bias=False)
        nn.init.constant_(self.eta.weight, init_gamma)
        nn.init.constant_(self.xi.weight, init_alpha)
        self.n_prototypes = n_prototypes
        self.alpha = None
        self.activation_norm = str(activation_norm).lower()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gamma = torch.square(self.eta.weight)
        alpha = torch.div(1.0, torch.exp(torch.neg(self.xi.weight)) + 1.0)
        self.alpha = alpha
        si = torch.mul(torch.exp(torch.neg(torch.mul(gamma, inputs))), alpha)
        if self.activation_norm == "max":
            max_val, _ = torch.max(si, dim=-1, keepdim=True)
            si = si / (max_val + 1e-4)
        return si


class Belief_layer(nn.Module):
    """Distribute each prototype's activation as belief mass over classes.

    ``belief_init="class"`` gives prototype ``p`` a head start on class
    ``p % num_class``. The default random ``beta`` makes every prototype's belief
    split ``u`` near-uniform, so the fused score
    ``log q(c) = sum_p log(1 - s_p (1 - u_cp))`` starts almost identical for
    every class and the head spends its first few hundred steps just breaking
    that symmetry -- the main reason EUCR needs several times as many steps as a
    linear+CE head to reach the same accuracy. Round-robin assignment is the
    usual initialisation for prototype-based evidential classifiers and costs
    nothing.
    """

    def __init__(
        self,
        n_prototypes: int,
        num_class: int,
        belief_init: str = "random",
        init_peak: float = 2.0,
    ) -> None:
        super().__init__()
        self.beta = nn.Linear(
            in_features=n_prototypes, out_features=num_class, bias=False
        ).weight
        self.num_class = num_class
        if str(belief_init).lower() == "class" and num_class > 1:
            with torch.no_grad():
                assignment = torch.arange(n_prototypes) % num_class
                self.beta.copy_(self.beta.abs().clamp(0.5, 1.0))
                self.beta[assignment, torch.arange(n_prototypes)] *= float(init_peak)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True).clamp_min(_NUMERICAL_EPS)
        u = beta / beta_sum
        mass_prototype = torch.einsum("cp,b...p->b...pc", u, inputs)
        return mass_prototype


class Omega_layer(nn.Module):
    """Append the per-prototype ignorance mass ``omega = 1 - sum(beliefs)``."""

    def __init__(self, n_prototypes: int, num_class: int) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mass_omega_sum = 1 - torch.sum(inputs, -1, keepdim=True)
        return torch.cat([inputs, mass_omega_sum], -1)


class Dempster_layer(nn.Module):
    """Combine the prototype mass functions with Dempster's rule, in closed form.

    For singleton-plus-ignorance masses, Dempster's rule for two sources sends

        class c : m1(c)m2(c) + m1(c)omega2 + omega1 m2(c)
        ignorance: omega1 * omega2   (both sources ignorant)

    (combine2 / combine3 are belief-transfer terms and must NOT add to the
    ignorance column; summing all three there triple-counts ignorance and biases
    the head toward maximum ignorance.)

    Adding the ignorance column to the class column gives the commonality
    ``q_k(c) = m_k(c) + omega_k``, and the rule above is exactly
    ``q(c) = q1(c) q2(c)`` with ``omega = omega1 omega2``. Both are products, so
    fusing ``P`` prototypes is

        log q(c) = sum_k log(m_k(c) + omega_k),   log omega = sum_k log omega_k
        m(c)     = q(c) - omega

    i.e. two reductions rather than ``P - 1`` sequential fusions. The per-step
    renormalisation the loop applied is a positive per-sample scalar that the
    recursion carries through linearly, so it only rescales the result and drops
    out of the final :class:`DempsterNormalize_layer`. Output matches the loop to
    ~2e-6 and is 40-75x faster at the prototype counts EUCR actually uses.
    """

    def __init__(self, n_prototypes: int, num_class: int, temper: float = 0.0) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class
        self.temper = float(temper)
        # Divide both log-sums by P**temper. temper=0 is Dempster's rule; temper=1
        # is the geometric mean of the per-prototype commonalities, i.e. a
        # cautious/idempotent combination that stops P prototypes reading ONE
        # feature vector from being multiply-counted as P independent sources.
        # q_p(c) >= omega_p holds for every p, and geometric means preserve it,
        # so m(c) = q(c) - omega stays non-negative for any temper.
        self._divisor = float(n_prototypes) ** self.temper

    def forward(self, inputs: torch.Tensor, return_conflict: bool = False):
        # inputs: [..., P, C + 1]; reduce over the prototype axis -2.
        mass = inputs[..., :-1]
        omega = inputs[..., -1:].clamp_min(0.0)
        log_q = (mass + omega).clamp_min(_LOG_EPS).log().sum(dim=-2)
        log_omega = omega.clamp_min(_LOG_EPS).log().sum(dim=-2)
        if self._divisor != 1.0:
            log_q = log_q / self._divisor
            log_omega = log_omega / self._divisor
        # Work relative to the largest log-mass so the exponentials stay in range
        # (both sums run over P terms and would otherwise underflow for large P).
        ref = torch.maximum(log_q.max(dim=-1, keepdim=True).values, log_omega)
        q = (log_q - ref).exp()
        omega_out = (log_omega - ref).exp()
        combined = torch.cat([(q - omega_out).clamp_min(0.0), omega_out], dim=-1)
        if return_conflict:
            # Every per-prototype mass function sums to one, so the unnormalised
            # total K = sum_c m(c) + omega is at most one and 1 - K is exactly the
            # mass Dempster's rule sends to the empty set -- the conflict between
            # prototypes, which _safe_normalize otherwise discards. Undo the
            # stabilising shift to recover it on the true scale. (Exact only at
            # temper=0; for temper>0 it is the same quantity for the tempered
            # combination, and still monotone in prototype disagreement.)
            log_k = ref.squeeze(-1) + combined.sum(dim=-1).clamp_min(_LOG_EPS).log()
            # The loop renormalised after every fusion, so its output was already a
            # unit-mass function; keep that contract (the following
            # DempsterNormalize_layer is then idempotent, as it was before).
            return _safe_normalize(combined, dim=-1), log_k, log_omega.squeeze(-1)
        return _safe_normalize(combined, dim=-1)


class DempsterNormalize_layer(nn.Module):
    """Normalise a combined mass function so it sums to one."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _safe_normalize(inputs, dim=-1)


class Dempster_Shafer_module(nn.Module):
    """Feature vector -> normalised Dempster-Shafer masses ``[B, n_classes + 1]``."""

    def __init__(
        self,
        n_feature_maps: int,
        n_classes: int,
        n_prototypes: int,
        metric: str = "cosine",
        belief_init: str = "random",
        temper: float = 0.0,
        activation_norm: str = "max",
    ) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.n_feature_maps = n_feature_maps
        self.metric = str(metric).lower()
        self.ds1 = Distance_layer(
            n_prototypes=n_prototypes,
            n_feature_maps=n_feature_maps,
            metric=self.metric,
        )
        # gamma = eta**2 sets the activation temperature exp(-gamma * distance).
        # Euclidean distances are O(feat_dim) so they need a tiny gamma (~0.01);
        # cosine distances live in [0, 2] and need gamma ~1 to span the activation
        # range. Pick the init that matches the metric's scale.
        init_gamma = 0.1 if self.metric == "euclidean" else 1.0
        self.ds1_activate = DistanceActivation_layer(
            n_prototypes=n_prototypes,
            init_gamma=init_gamma,
            activation_norm=activation_norm,
        )
        self.ds2 = Belief_layer(
            n_prototypes=n_prototypes, num_class=n_classes, belief_init=belief_init
        )
        self.ds2_omega = Omega_layer(n_prototypes=n_prototypes, num_class=n_classes)
        self.ds3_dempster = Dempster_layer(
            n_prototypes=n_prototypes, num_class=n_classes, temper=temper
        )
        self.ds3_normalize = DempsterNormalize_layer()

    def forward(self, inputs: torch.Tensor, return_conflict: bool = False):
        """Masses ``[B, C+1]``; with ``return_conflict`` also ``(log K, log omega)``.

        ``log K`` is the log of the unnormalised combined mass, so ``1 - exp(log K)``
        is the conflict between prototypes. Unlike ``omega`` it does not degenerate
        as the prototype count grows: it rises as the evidence sources disagree.
        """
        ed = self.ds1(inputs)
        ed_ac = self.ds1_activate(ed)
        mass_prototypes = self.ds2(ed_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        if return_conflict:
            mass_dempster, log_k, log_omega = self.ds3_dempster(
                mass_prototypes_omega, return_conflict=True
            )
            return self.ds3_normalize(mass_dempster), log_k, log_omega
        mass_dempster = self.ds3_dempster(mass_prototypes_omega)
        return self.ds3_normalize(mass_dempster)


def _tile(a: torch.Tensor, dim: int, n_tile: int) -> torch.Tensor:
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    ).to(a.device)
    return torch.index_select(a, dim, order_index)


class DM(nn.Module):
    """Decision-making layer: masses -> per-class expected utilities.

    The first ``num_class`` columns are ``beliefs + (1 - nu) * omega`` (the
    pignistic-style redistribution of the ignorance mass) and the last column is
    the retained ``nu * omega`` ignorance term.
    """

    def __init__(
        self, num_class: int, nu: float = 0.9, device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.nu = nu
        self.num_class = num_class

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upper = torch.unsqueeze((1 - self.nu) * inputs[..., -1], -1)
        upper_tiled = _tile(upper, dim=-1, n_tile=self.num_class)
        beliefs = inputs[..., :-1] + upper_tiled
        omega = self.nu * inputs[..., -1:]
        return torch.cat([beliefs, omega], dim=-1)


def pignistic_probability(
    mass: torch.Tensor, eps: float = _NUMERICAL_EPS, scale: float = 1.0
) -> torch.Tensor:
    """Smets pignistic transform of a singleton+ignorance mass function.

    Splits the ignorance mass equally over the classes:
    ``BetP_c = m({c}) + omega / C``. Unlike the :class:`DM` expected-utility
    vector, the result is a genuine probability distribution (sums to one), so
    cross-entropy / NLL on it is well defined.

    Usage:
        >>> betp = pignistic_probability(mass)  # mass: [B, C+1] -> [B, C]
    """
    beliefs = mass[..., :-1]
    omega = mass[..., -1:]
    num_classes = beliefs.size(-1)
    betp = beliefs + omega / num_classes
    betp = betp / betp.sum(dim=-1, keepdim=True).clamp_min(eps)
    if scale != 1.0:
        # Tempering divides log q(c) by P**temper, which shrinks the decision
        # logits by the same factor -- a temperature collapse that leaves the
        # readout near-uniform and stops the head learning at all (measured:
        # macro recall 0.000 at temper=1 without this, 0.710 with it). Undo it
        # on the DECISION only; the mass function stays tempered, so omega and
        # the conflict readout keep their cautious values.
        return torch.softmax(scale * betp.clamp_min(eps).log(), dim=-1)
    return betp


class PignisticNLLLoss(nn.Module):
    """Negative log-likelihood on the pignistic probability.

    Drop-in replacement for :class:`EvidentialLoss` when the head emits a
    pignistic distribution: same call signature ``(probs, targets, beliefs,
    epoch)`` so callers need not branch, but ``beliefs`` / ``epoch`` are unused
    (no BCE-style per-class term, no KL warm-up -- the uniform-attractor KL is
    exactly what destabilised the expected-utility head).
    """

    def __init__(self, num_classes: int, eps: float = _NUMERICAL_EPS) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, probs, targets, beliefs=None, epoch=None):
        log_probs = probs.clamp_min(self.eps).log()
        return F.nll_loss(log_probs, targets)


class EvidentialLoss(nn.Module):
    """BCE-style evidential loss on expected utilities with a KL warm-up gate."""

    def __init__(
        self, num_classes: int, kl_warmup_epochs: int = 35, lmda: float = 10.0
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lmda = lmda
        self.kl_warmup_epochs = kl_warmup_epochs

    def forward(self, E_preds, targets, beliefs=None, epoch=None):
        E = E_preds.float().clamp(_NUMERICAL_EPS, 1.0 - _NUMERICAL_EPS)
        yk = F.one_hot(targets, num_classes=self.num_classes).float().to(E.device)

        base = F.binary_cross_entropy(E, yk, reduction="none").sum(dim=1)

        U = E
        p = U / (U.sum(dim=1, keepdim=True) + 1e-8)
        K = U.size(1)
        kl_vals = (p * (p.add(1e-8).log())).sum(dim=1) + math.log(K)

        u_max, _ = torch.max(E, dim=1)
        u_true = E.gather(1, targets.view(-1, 1)).squeeze(1)
        gate = u_max * (1.0 - u_true)
        kl = (kl_vals * gate).mean()

        kl_weight = self._kl_warmup_weight(epoch)
        return (base + kl_weight * kl).mean()

    def _kl_warmup_weight(self, epoch):
        # When the epoch is unknown (e.g. ``observe`` is exercised outside the
        # ``life_experience`` loop that sets ``model.real_epoch``), treat the KL
        # warm-up as not started. Applying the KL term at full strength from the
        # first step rewards the trivial uniform / maximum-ignorance solution and
        # traps the Dempster-Shafer head before any belief structure can form.
        if epoch is None:
            return 0.0
        if self.kl_warmup_epochs <= 0:
            return self.lmda
        if epoch <= 5:
            return 0.0
        if epoch >= self.kl_warmup_epochs:
            return self.lmda
        progress = float(epoch - 5) / float(self.kl_warmup_epochs)
        return self.lmda * 0.5 * (1.0 - math.cos(math.pi * progress))


__all__ = [
    "Dempster_Shafer_module",
    "DM",
    "EvidentialLoss",
    "PignisticNLLLoss",
    "pignistic_probability",
]
