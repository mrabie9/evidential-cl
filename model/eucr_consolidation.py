"""EUCR consolidation: evidential per-channel / per-parameter importance.

This is the anti-forgetting machinery for EUCR. It replaces the Fisher-information
diagonal of EWC with an *evidential* importance signal read out from the
Dempster-Shafer uncertainty the backbone probes assign to each stage.

Note on lineage: despite the shape of the formula this is **not** a Fisher
approximation -- there is no Laplace / log-likelihood-curvature story behind it.
Squaring the gradient of a scalar *output functional*, accumulated over
(effectively unlabelled) data, is Memory Aware Synapses (Aljundi et al., ECCV
2018), which uses ``grad ||F(x)||^2``. EUCR substitutes a Dempster-Shafer
uncertainty for MAS's output norm. Call it a MAS variant, not a Fisher analogue,
and benchmark it against MAS.

Pipeline per task ``t``:
  1. :func:`compute_importance` -- run over the task's data and accumulate the
     squared gradient of the mean backbone DS uncertainty w.r.t. each shared
     backbone parameter. The uncertainty readout (``uncertainty_mode``) selects
     the DS component: ``nonspecificity`` (the ignorance mass ``omega``),
     ``discord`` (entropy of the pignistic probability), or ``both`` (their sum,
     the DS total). Parameters whose perturbation most changes the backbone's
     uncertainty are deemed important (a MAS-style sensitivity measure, see the
     lineage note above). ``nonspecificity`` is a dead readout: ``omega`` is ~0 by
     construction (see ``DistanceActivation_layer``), which also makes ``both``
     *bit-identical* to ``discord`` -- measured max|both - discord| = 0.0. There
     are two live settings here, not three.
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

_UNCERTAINTY_MODES = ("nonspecificity", "discord", "both", "uniform", "random_proj")

# Per-task evidential heads / probes are NOT shared across tasks, so they are
# excluded from consolidation. Everything else (conv1, layer1..layer4, the
# feature LayerNorm, BatchNorm params, input adapter) is the shared backbone.
_EXCLUDE_SUBSTRINGS = ("ds_head", "dm_head", "ce_head", "probes")


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


def _random_projection_scalar(backbone, inputs, generator) -> torch.Tensor:
    """MAS with a random head: grad of ||W f_s||^2 on the same stage features.

    The tightest control on whether the Dempster-Shafer path contributes anything
    to the importance signal. Everything the evidential probes do -- which stages,
    global average pooling, normalisation, a readout of ``num_classes`` width -- is
    kept; only the DS fusion and its learned parameters are replaced by a FIXED
    random linear map. If this recovers the same importance, Omega is a measure of
    layer and channel sensitivity that any random readout finds, and the
    evidential machinery is decorative.
    """
    feats: Dict[int, torch.Tensor] = {}
    handles = [
        getattr(backbone, f"layer{s}").register_forward_hook(
            lambda _m, _i, out, s=s: feats.__setitem__(s, out)
        )
        for s in backbone.probe_stages
    ]
    try:
        backbone(inputs)
    finally:
        for h in handles:
            h.remove()

    if not hasattr(backbone, "_mas_random_heads"):
        backbone._mas_random_heads = {}
    per_stage = []
    for stage, feat in feats.items():
        pooled = torch.nn.functional.adaptive_avg_pool1d(feat, 1).flatten(1)
        pooled = torch.nn.functional.layer_norm(pooled, (pooled.size(-1),))
        key = (stage, pooled.size(-1))
        if key not in backbone._mas_random_heads:
            w = torch.empty(
                backbone.num_classes, pooled.size(-1), device=pooled.device
            )
            torch.nn.init.normal_(w, std=pooled.size(-1) ** -0.5, generator=generator)
            backbone._mas_random_heads[key] = w
        readout = pooled @ backbone._mas_random_heads[key].t()
        per_stage.append(readout.pow(2).sum(dim=-1))
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
    """Estimate evidential importance (a MAS-style sensitivity diagonal).

    Two known weaknesses, both unfixed:

    * This runs under ``backbone.eval()``, i.e. with BatchNorm running statistics
      -- the exact regime the cosine-distance head is least stable in (pooled
      feature directional coherence swings 0.16-0.42 under running stats against
      a steady 0.061-0.080 under batch stats). The importance is measured in the
      one regime we have shown we cannot trust.
    * For ``nonspecificity`` the *estimator* degenerates as well as the value:
      ``grad E[omega] = E[omega grad log omega]``, so when omega spans orders of
      magnitude the effective sample size collapses onto a few inputs.
      Differentiating ``log omega`` is scale-free; the closed-form
      ``Dempster_layer`` already computes it.

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
    if uncertainty_mode == "uniform":
        # Ablation control: Omega_i = 1 for every coordinate, so the penalty
        # becomes plain L2 pull toward theta_star with no evidential weighting.
        # If this matches the evidential readouts, the DS uncertainty is
        # contributing nothing and only the quadratic anchor is doing work --
        # which is what PR-E1 found at the per-coordinate level, where a
        # displacement control beat every importance measure.
        return {
            name: torch.ones_like(param)
            for name, param in _named_consolidatable_params(backbone)
        }

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
    generator = None
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

        if uncertainty_mode == "random_proj":
            if generator is None:
                generator = torch.Generator(device=device)
                generator.manual_seed(0)
            scalar = _random_projection_scalar(backbone, inputs, generator).mean()
        else:
            out = backbone(inputs, return_probes=True)
            probe_outs = out[-1]
            if not probe_outs:
                break
            scalar = _backbone_uncertainty(probe_outs, uncertainty_mode).mean()

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
