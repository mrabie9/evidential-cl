"""EUCR consolidation: evidential per-channel / per-parameter importance.

This is the anti-forgetting machinery for EUCR. It replaces the Fisher-information
diagonal of EWC with an *evidential* importance signal read out from the
Dempster-Shafer uncertainty the backbone probes assign to each stage.

Pipeline per task ``t``:
  1. :func:`compute_importance` -- run over the task's data and accumulate the
     squared gradient of the mean backbone DS uncertainty w.r.t. each shared
     backbone parameter. The uncertainty readout (``uncertainty_mode``) selects
     the DS component: ``nonspecificity`` (the ignorance mass ``omega``),
     ``discord`` (entropy of the pignistic probability), or ``both`` (their sum,
     the DS total). Parameters whose perturbation most changes the backbone's
     uncertainty are deemed important (evidential analogue of the Fisher
     diagonal). Note ``nonspecificity`` alone weakens as the long Dempster chain
     drives ``omega`` -> 0; ``discord`` / ``both`` stay informative.
  2. :func:`to_channel` (optional) -- collapse per-weight importance to one score
     per convolution output channel, then broadcast it back across the filter.
  3. :func:`accumulate` -- online (running-sum) accumulation across tasks.
  4. :func:`snapshot` -- store the post-task backbone weights ``theta_star``.

During training of task ``t+1`` the penalty
``lambda * sum_i Omega_i (theta_i - theta_star_i)^2`` is added to the loss.
No pruning and no binary masks are used.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

_UNCERTAINTY_MODES = ("nonspecificity", "discord", "both")

# Per-task evidential heads / probes are NOT shared across tasks, so they are
# excluded from consolidation. Everything else (conv1, layer1..layer4, the
# feature LayerNorm, BatchNorm params, input adapter) is the shared backbone.
_EXCLUDE_SUBSTRINGS = ("ds_head", "dm_head", "probes")


def is_consolidatable(name: str) -> bool:
    """Return True for shared backbone parameters that should be regularised."""
    return not any(token in name for token in _EXCLUDE_SUBSTRINGS)


def _named_consolidatable_params(model: nn.Module) -> Iterable:
    for name, param in model.named_parameters():
        if param.requires_grad and is_consolidatable(name):
            yield name, param


def _stage_uncertainty(
    beliefs: torch.Tensor, omega: torch.Tensor, mode: str, eps: float = 1e-8
) -> torch.Tensor:
    """Per-sample Dempster-Shafer uncertainty for one probe stage. Shape ``[B]``.

    Decomposes total DS uncertainty into its two additive components:

    * ``nonspecificity`` -- the ignorance mass ``omega`` (normalised; "how vague").
    * ``discord`` -- the Shannon entropy of the pignistic probability
      ``BetP_c = beliefs_c + omega / C`` divided by ``log C`` ("how conflicted
      across classes"). Entropy is taken on ``BetP`` -- a genuine probability
      distribution -- not on the raw masses, on which Shannon entropy is undefined.
    * ``both`` -- their sum, the (normalised) DS total uncertainty.

    Using ``discord`` / ``both`` keeps the importance signal alive even when the
    long Dempster chain drives ``omega`` toward zero (where ``nonspecificity``
    alone vanishes).
    """
    if mode == "nonspecificity":
        return omega
    num_classes = beliefs.size(-1)
    log_c = math.log(num_classes) if num_classes > 1 else 1.0
    betp = beliefs + omega.unsqueeze(-1) / num_classes
    betp = betp / betp.sum(dim=-1, keepdim=True).clamp_min(eps)
    discord = -(betp * (betp + eps).log()).sum(dim=-1) / log_c
    if mode == "discord":
        return discord
    return omega + discord


def _backbone_uncertainty(probe_outs, mode: str) -> torch.Tensor:
    """Mean over stages of the chosen per-sample DS uncertainty. Shape ``[B]``."""
    if mode not in _UNCERTAINTY_MODES:
        raise ValueError(
            f"Unknown eucr_uncertainty mode {mode!r}; expected one of {_UNCERTAINTY_MODES}."
        )
    per_stage = [
        _stage_uncertainty(beliefs, omega, mode) for _eu, beliefs, omega in probe_outs
    ]
    return torch.stack(per_stage, dim=0).mean(dim=0)


@torch.enable_grad()
def compute_importance(
    backbone: nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
    normalize: bool = True,
    uncertainty_mode: str = "both",
) -> Dict[str, torch.Tensor]:
    """Estimate evidential importance (Fisher-style diagonal of uncertainty).

    For each batch we differentiate the mean backbone DS uncertainty
    (``uncertainty_mode`` selects nonspecificity / discord / both) and accumulate
    ``grad ** 2`` weighted by the batch size, then normalise by the number of
    samples seen. Differentiating uncertainty rather than confidence yields the
    same importance (the squared gradient is invariant to the sign flip).
    Returns a dict ``{param_name: importance_tensor}`` over shared backbone
    parameters only. With ``normalize`` the per-task importance is rescaled to
    unit mean so ``lambda`` stays interpretable across datasets and uncertainty
    modes.
    """
    was_training = backbone.training
    backbone.eval()
    importance: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=param.device)
        for name, param in _named_consolidatable_params(backbone)
    }
    if not importance:
        if was_training:
            backbone.train()
        return importance

    n_seen = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = batch[0]
        if not torch.is_tensor(inputs):
            inputs = torch.as_tensor(inputs)
        inputs = inputs.float().to(device)
        if inputs.numel() == 0:
            continue
        bsz = inputs.size(0)

        out = backbone(inputs, return_probes=True)
        probe_outs = out[-1]
        if not probe_outs:
            break
        uncertainty = _backbone_uncertainty(probe_outs, uncertainty_mode)
        scalar = uncertainty.mean()

        backbone.zero_grad(set_to_none=True)
        scalar.backward()

        for name, param in _named_consolidatable_params(backbone):
            if param.grad is not None:
                importance[name] += (param.grad.detach() ** 2) * bsz
        n_seen += bsz

    backbone.zero_grad(set_to_none=True)
    if n_seen > 0:
        for name in importance:
            importance[name] /= float(n_seen)

    if normalize and importance:
        total = sum(float(v.sum()) for v in importance.values())
        count = sum(int(v.numel()) for v in importance.values())
        mean = total / count if count > 0 else 0.0
        if mean > 0:
            for name in importance:
                importance[name] = importance[name] / mean

    if was_training:
        backbone.train()
    return importance


def to_channel(
    backbone: nn.Module,
    importance: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Collapse per-weight importance to per-output-channel, then broadcast."""
    channel_imp: Dict[str, torch.Tensor] = {}
    for name, imp in importance.items():
        if imp.dim() >= 2:
            reduce_dims = tuple(range(1, imp.dim()))
            per_channel = imp.mean(dim=reduce_dims, keepdim=True)
            channel_imp[name] = per_channel.expand_as(imp).contiguous()
        else:
            channel_imp[name] = imp.clone()
    return channel_imp


def accumulate(
    old: Optional[Dict[str, torch.Tensor]],
    new: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Online (running-sum) accumulation of importance across tasks."""
    if old is None:
        return {k: v.clone() for k, v in new.items()}
    merged: Dict[str, torch.Tensor] = {}
    for k in set(old) | set(new):
        if k in old and k in new:
            merged[k] = old[k] + new[k]
        elif k in old:
            merged[k] = old[k].clone()
        else:
            merged[k] = new[k].clone()
    return merged


def snapshot(backbone: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone the current shared backbone weights as the consolidation anchor."""
    return {
        name: param.detach().clone()
        for name, param in _named_consolidatable_params(backbone)
    }


def penalty(
    backbone: nn.Module,
    importance: Optional[Dict[str, torch.Tensor]],
    theta_star: Optional[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Quadratic EUCR consolidation penalty (unweighted by lambda)."""
    device = next(backbone.parameters()).device
    loss = torch.zeros((), device=device)
    if not importance or not theta_star:
        return loss
    for name, param in _named_consolidatable_params(backbone):
        if name in importance and name in theta_star:
            omega = importance[name].to(param.device)
            anchor = theta_star[name].to(param.device)
            loss = loss + (omega * (param - anchor) ** 2).sum()
    return loss


__all__ = [
    "is_consolidatable",
    "compute_importance",
    "to_channel",
    "accumulate",
    "snapshot",
    "penalty",
]
