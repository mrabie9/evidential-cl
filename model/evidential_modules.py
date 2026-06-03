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


class Distance_layer(nn.Module):
    """Squared Euclidean distance from each input to ``n_prototypes`` prototypes."""

    def __init__(self, n_prototypes: int, n_feature_maps: int) -> None:
        super().__init__()
        self.w = nn.Linear(
            in_features=n_feature_maps, out_features=n_prototypes, bias=False
        ).weight
        self.n_prototypes = n_prototypes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, F], self.w: [P, F] -> [B, P] squared distances.
        return torch.cdist(inputs, self.w, p=2).pow(2)


class DistanceActivation_layer(nn.Module):
    """Turn distances into per-prototype activations in ``[0, 1]``."""

    def __init__(self, n_prototypes: int, init_alpha: float = 0.0, init_gamma: float = 0.1) -> None:
        super().__init__()
        self.eta = nn.Linear(in_features=n_prototypes, out_features=1, bias=False)
        self.xi = nn.Linear(in_features=n_prototypes, out_features=1, bias=False)
        nn.init.constant_(self.eta.weight, init_gamma)
        nn.init.constant_(self.xi.weight, init_alpha)
        self.n_prototypes = n_prototypes
        self.alpha = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gamma = torch.square(self.eta.weight)
        alpha = torch.div(1.0, torch.exp(torch.neg(self.xi.weight)) + 1.0)
        self.alpha = alpha
        si = torch.mul(torch.exp(torch.neg(torch.mul(gamma, inputs))), alpha)
        max_val, _ = torch.max(si, dim=-1, keepdim=True)
        si = si / (max_val + 1e-4)
        return si


class Belief_layer(nn.Module):
    """Distribute each prototype's activation as belief mass over classes."""

    def __init__(self, n_prototypes: int, num_class: int) -> None:
        super().__init__()
        self.beta = nn.Linear(
            in_features=n_prototypes, out_features=num_class, bias=False
        ).weight
        self.num_class = num_class

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        u = torch.div(beta, beta_sum)
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
    """Sequentially combine prototype mass functions with Dempster's rule."""

    def __init__(self, n_prototypes: int, num_class: int) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / torch.sum(combine2_3, dim=-1, keepdim=True)
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1


class DempsterNormalize_layer(nn.Module):
    """Normalise a combined mass function so it sums to one."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / torch.sum(inputs, dim=-1, keepdim=True)


class Dempster_Shafer_module(nn.Module):
    """Feature vector -> normalised Dempster-Shafer masses ``[B, n_classes + 1]``."""

    def __init__(self, n_feature_maps: int, n_classes: int, n_prototypes: int) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.n_feature_maps = n_feature_maps
        self.ds1 = Distance_layer(n_prototypes=n_prototypes, n_feature_maps=n_feature_maps)
        self.ds1_activate = DistanceActivation_layer(n_prototypes=n_prototypes)
        self.ds2 = Belief_layer(n_prototypes=n_prototypes, num_class=n_classes)
        self.ds2_omega = Omega_layer(n_prototypes=n_prototypes, num_class=n_classes)
        self.ds3_dempster = Dempster_layer(n_prototypes=n_prototypes, num_class=n_classes)
        self.ds3_normalize = DempsterNormalize_layer()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ed = self.ds1(inputs)
        ed_ac = self.ds1_activate(ed)
        mass_prototypes = self.ds2(ed_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
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

    def __init__(self, num_class: int, nu: float = 0.9, device: torch.device | None = None) -> None:
        super().__init__()
        self.nu = nu
        self.num_class = num_class

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upper = torch.unsqueeze((1 - self.nu) * inputs[..., -1], -1)
        upper_tiled = _tile(upper, dim=-1, n_tile=self.num_class)
        beliefs = inputs[..., :-1] + upper_tiled
        omega = self.nu * inputs[..., -1:]
        return torch.cat([beliefs, omega], dim=-1)


class EvidentialLoss(nn.Module):
    """BCE-style evidential loss on expected utilities with a KL warm-up gate."""

    def __init__(self, num_classes: int, kl_warmup_epochs: int = 35, lmda: float = 10.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lmda = lmda
        self.kl_warmup_epochs = kl_warmup_epochs

    def forward(self, E_preds, targets, beliefs=None, epoch=None):
        E = E_preds.clamp(1e-6, 1 - 1e-6)
        yk = F.one_hot(targets, num_classes=self.num_classes).float().to(E.device)

        log_probs = yk * torch.log(E) + (1 - yk) * torch.log(1 - E)
        base = -torch.sum(log_probs, dim=1)

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
]
