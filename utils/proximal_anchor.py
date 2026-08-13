"""Closed-form (proximal) application of a diagonal quadratic anchor.

Every regularisation-based continual learner in this repository ends up adding
the same shape of term to its training loss: a diagonal quadratic anchor that
pulls each parameter back toward the value it held at the end of the previous
task, weighted by a per-parameter importance::

    P(theta) = (k / 2) * sum_i Omega_i * (theta_i - theta_i^*)^2

Only ``k`` and the meaning of ``Omega`` differ between methods (see
:func:`anchor_curvature` for the per-method values). The gradient of ``P`` is
``k * Omega_i * (theta_i - theta_i^*)``, so the anchor is a spring of stiffness
``k * Omega_i``.

Adding ``P`` to the loss makes the optimiser take an **explicit** gradient step
on that spring, and explicit descent on a quadratic is stable only while
``lr * k * Omega_i < 2``. Every importance estimator used here is heavy tailed
-- SI's and RWalk's path integrals divide by a squared parameter displacement,
EWC's Fisher is a squared gradient, UCL's strength is an inverse posterior
standard deviation -- so a handful of parameters reliably land outside that
window. Those parameters oscillate with growing amplitude, and because
``clip_grad_norm_`` rescales *every* gradient by one global scalar, their
divergence drags the whole network's effective learning rate down with them.

The proximal (backward-Euler) form evaluates the anchor gradient at the *new*
point instead of the old one::

    theta_new = theta_plus - lr * k * Omega_i * (theta_new - theta_i^*)

which solves in closed form to a convex combination::

    b_i       = lr * k * Omega_i
    theta_new = (theta_plus + b_i * theta^*) / (1 + b_i)
              = (1 - a_i) * theta_plus + a_i * theta^*,   a_i = b_i / (1 + b_i)

Because ``a_i`` saturates at 1 for any importance, the update can never
overshoot the anchor: ``Omega -> 0`` leaves a parameter free, ``Omega -> inf``
pins it exactly to ``theta^*``. It is unconditionally stable at any ``lr`` and
any importance scale, and it costs nothing in the backward pass -- so it also
stops consuming the gradient-norm clip budget that the task loss needs.

Two consequences are worth stating because they change how the methods behave,
not just how they are computed:

* **Negative importances are clamped to zero.** ``model.si`` and ``model.rwalk``
  can both produce negative importance (neither applies a ``relu`` to its path
  integral). In the loss form a negative ``Omega_i`` is an *inverted* parabola
  that actively pushes the parameter away from its anchor; in the proximal form
  it would make ``1 + b_i`` approach or cross zero and blow up. Clamping is the
  conservative reading -- "this parameter is not protected" -- and it is applied
  here rather than in each caller so the treatment is identical everywhere.

* **``lambda`` does not transfer between the two modes.** In the loss form the
  anchor competes with the task gradient and is then globally renormalised by
  the clip; in the proximal form ``b_i = lr * k * Omega_i`` is applied exactly.
  The useful range typically moves by several orders of magnitude, so a mode
  switch requires its own sweep.

Usage:
    >>> coefficient = proximal_anchor_coefficient(
    ...     learning_rate=0.003, curvature=anchor_curvature("si", si_c)
    ... )
    >>> apply_proximal_anchor(param, omega, param_star, coefficient)
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch

# How the diagonal quadratic anchor is applied to the parameters:
#   "loss"     -- add it to the training loss and let the optimiser descend it
#                 (the original behaviour of every method here).
#   "proximal" -- keep it out of the backward pass and apply its closed-form
#                 minimiser as a post-step update (this module).
ANCHOR_MODES = ("loss", "proximal")

# Curvature ``k`` in ``P(theta) = (k/2) * sum_i Omega_i (theta_i - theta_i^*)^2``
# for each method, expressed as the multiplier on that method's own penalty
# strength. It is read straight off how each learner writes its loss term:
#
#   si     : loss += si_c * sum Omega (theta - prev)^2        -> k = 2 * si_c
#   woe_si : loss += woe_lambda * sum Omega (theta - prev)^2  -> k = 2 * woe_lambda
#   ewc    : loss += 0.5 * lamb * sum F (theta - star)^2      -> k = 1 * lamb
#   rwalk  : loss += lamb * sum (F + s) (theta - star)^2      -> k = 2 * lamb
#
# UCL is not in this table: its ``k`` depends on the minibatch size and the
# regularised-parameter count, so it is computed per step (see
# ``model.ucl_bresnet.Net._apply_proximal_mu_anchor``).
_PENALTY_STRENGTH_MULTIPLIER = {
    "si": 2.0,
    "woe_si": 2.0,
    "ewc": 1.0,
    "rwalk": 2.0,
}


def anchor_curvature(method: str, penalty_strength: float) -> float:
    """Return the anchor curvature ``k`` for a method's penalty strength.

    ``k`` is defined by writing the method's penalty in the canonical form
    ``(k/2) * sum_i Omega_i (theta_i - theta_i^*)^2``, so that the anchor
    gradient is ``k * Omega_i * (theta_i - theta_i^*)``. The factor differs
    between methods only because they write the same penalty with different
    constants folded into their ``lambda``; see ``_PENALTY_STRENGTH_MULTIPLIER``.

    Args:
        method: One of ``{"si", "woe_si", "ewc", "rwalk"}``.
        penalty_strength: The method's own penalty weight (``si_c`` for SI,
            ``lamb`` for EWC and RWalk, ``woe_lambda`` for WoE-SI).

    Returns:
        The curvature ``k``.

    Raises:
        ValueError: If ``method`` has no registered multiplier.

    Usage:
        >>> anchor_curvature("ewc", 100.0)
        100.0
        >>> anchor_curvature("si", 0.4)
        0.8
    """
    if method not in _PENALTY_STRENGTH_MULTIPLIER:
        raise ValueError(
            f"method must be one of {tuple(_PENALTY_STRENGTH_MULTIPLIER)}, "
            f"got {method!r}"
        )
    return _PENALTY_STRENGTH_MULTIPLIER[method] * float(penalty_strength)


def validate_anchor_mode(anchor_mode: object) -> str:
    """Normalise and validate an ``anchor_mode`` value.

    Args:
        anchor_mode: Candidate mode, typically straight off ``args``.

    Returns:
        The validated mode string.

    Raises:
        ValueError: If the mode is not in :data:`ANCHOR_MODES`.

    Usage:
        >>> validate_anchor_mode("proximal")
        'proximal'
    """
    mode = str(anchor_mode)
    if mode not in ANCHOR_MODES:
        raise ValueError(f"anchor_mode must be one of {ANCHOR_MODES}, got {mode!r}")
    return mode


def proximal_anchor_coefficient(learning_rate: float, curvature: float) -> float:
    """Compute the scalar multiplying the importance in the proximal update.

    The per-parameter blend weight is ``b_i = coefficient * Omega_i``, so this is
    ``lr * k``. Splitting it out keeps the per-tensor update free of any
    optimiser or method bookkeeping.

    Args:
        learning_rate: The optimiser's current learning rate for the parameters
            being anchored.
        curvature: The anchor curvature ``k`` from :func:`anchor_curvature`.

    Returns:
        The coefficient ``lr * k``; ``0.0`` disables the anchor.

    Usage:
        >>> proximal_anchor_coefficient(0.003, 0.8)
        0.0024
    """
    return float(learning_rate) * float(curvature)


@torch.no_grad()
def apply_proximal_anchor(
    param: torch.Tensor,
    importance: torch.Tensor,
    anchor: torch.Tensor,
    coefficient: float,
) -> None:
    """Pull one parameter tensor toward its anchor, in place.

    Applies ``theta <- (theta + b * theta^*) / (1 + b)`` with
    ``b = coefficient * relu(Omega)``, the closed-form minimiser derived in the
    module docstring. Negative importance is clamped to zero (see the module
    docstring for why). A no-op while ``Omega`` is all zeros, which is the state
    before the first consolidation -- so the first task trains unregularised
    exactly as it does in the loss form.

    Args:
        param: Live parameter tensor, modified in place.
        importance: Per-parameter importance ``Omega``, broadcastable to
            ``param``.
        anchor: Per-parameter anchor ``theta^*``, broadcastable to ``param``.
        coefficient: ``lr * k`` from :func:`proximal_anchor_coefficient`.

    Usage:
        >>> apply_proximal_anchor(param, omega, param_star, 0.0024)
    """
    if coefficient == 0.0:
        return
    blend = torch.clamp(importance, min=0.0) * coefficient
    param.copy_((param + blend * anchor) / (1.0 + blend))


def optimizer_learning_rate(
    optimizer: torch.optim.Optimizer, group_index: int = 0
) -> float:
    """Read the current learning rate off an optimiser parameter group.

    The proximal step is derived for plain gradient descent, so it needs the step
    size the optimiser is actually using *now* -- not the configured initial
    value, which any scheduler may have since changed.

    Args:
        optimizer: The optimiser driving the anchored parameters.
        group_index: Index of the parameter group holding them (default ``0``).

    Returns:
        The group's learning rate.

    Usage:
        >>> optimizer_learning_rate(self.opt)
        0.003
    """
    return float(optimizer.param_groups[group_index]["lr"])


def log_importance_summary(
    method: str, task_index: int, importances: Iterable[torch.Tensor]
) -> None:
    """Print quantiles of a consolidated importance buffer.

    The proximal blend weight is ``b_i = lr * k * Omega_i``, so the penalty
    strength that actually does anything is set entirely by where ``Omega``
    sits -- and that scale is method- and dataset-specific, spanning several
    orders of magnitude between the median parameter and the tail. Printing it
    once per consolidation turns choosing ``lambda`` into arithmetic instead of
    a blind sweep, and makes an inert regulariser (every ``b_i`` far below 1)
    visible rather than silent.

    Args:
        method: Method name, used only in the log line.
        task_index: Index of the task that was just consolidated.
        importances: The per-parameter importance tensors.

    Usage:
        >>> log_importance_summary("si", 0, [omega for omega in buffers])
        [si] task 0 importance: n=3851033 neg=0.0000 q50=1.29e-05 ...
    """
    flat = [tensor.detach().reshape(-1).float() for tensor in importances]
    if not flat:
        return
    values = torch.cat(flat)
    negative_fraction = float((values < 0).float().mean())
    positive = values[values > 0].double()
    if positive.numel() == 0:
        print(f"[{method}] task {task_index} importance: all zero")
        return
    quantiles = {
        f"q{int(q * 100)}": float(torch.quantile(positive, q)) for q in (0.5, 0.9, 0.99)
    }
    print(
        f"[{method}] task {task_index} importance: n={values.numel()} "
        f"neg={negative_fraction:.4f} "
        + " ".join(f"{key}={value:.3e}" for key, value in quantiles.items())
        + f" max={float(positive.max()):.3e}"
    )


def resolve_anchor_mode(args: object, default: str = "loss") -> str:
    """Read ``anchor_mode`` off an ``args`` namespace, tolerating absence/None.

    Args:
        args: Parsed experiment arguments, or any object.
        default: Mode to use when the attribute is missing or ``None``.

    Returns:
        A validated mode string.

    Usage:
        >>> resolve_anchor_mode(args)
        'loss'
    """
    value: Optional[object] = getattr(args, "anchor_mode", None)
    if value is None:
        value = default
    return validate_anchor_mode(value)


__all__ = [
    "ANCHOR_MODES",
    "anchor_curvature",
    "apply_proximal_anchor",
    "log_importance_summary",
    "optimizer_learning_rate",
    "proximal_anchor_coefficient",
    "resolve_anchor_mode",
    "validate_anchor_mode",
]
