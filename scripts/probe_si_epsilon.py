"""Is SI's ``delta^2 + epsilon`` denominator dominated by epsilon?

SI consolidates the path integral as

    omega += W / (delta^2 + epsilon)

where ``delta`` is the *whole-task* displacement of each parameter and
``epsilon`` is 0.01 by default. The division is what makes SI's importance a
*normalised* path integral: parameters that moved a long way get their credit
discounted, so importance measures work-per-unit-distance rather than raw work.
If ``delta^2`` sits far below ``epsilon`` for most parameters, the denominator
is effectively the constant ``epsilon`` and the normalisation never happens --
omega degenerates to ``(1/epsilon) * sum(-grad * step)``, an unnormalised
Euclidean path integral, which is a different (and weaker) quantity than the
one SI is supposed to compute.

This monkeypatches consolidation to report, per task, the distribution of
``delta^2 / epsilon`` and the fraction of parameters for which ``delta^2``
actually contributes. Nothing here changes training; the patch only observes.

Companion to scripts/probe_rwalk_riemannian.py, which asks the same question of
RWalk's per-step ``0.5 * F * delta^2 + eps``.

Usage:
    la-maml_env/bin/python scripts/probe_si_epsilon.py
"""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model import si  # noqa: E402

ROWS: list[tuple[int, float, float, float, float, float, float]] = []

_original_consolidate = si.Net._consolidate_current_task


def _patched_consolidate(self) -> None:
    """Record the delta^2-vs-epsilon distribution, then run the real thing."""
    task = self.current_task
    if task is None:
        _original_consolidate(self)
        return
    deltas: list[torch.Tensor] = []
    for name, param in self.net.named_parameters():
        if not param.requires_grad or name.startswith("det_head"):
            continue
        key = self._param_to_key[name]
        prev = getattr(self, f"{key}_si_prev")
        deltas.append((param.detach() - prev).reshape(-1).float().pow(2))
    _original_consolidate(self)
    if not deltas:
        return
    d2 = torch.cat(deltas)
    ratio = d2 / self.epsilon
    q = torch.quantile(
        ratio, torch.tensor([0.5, 0.9, 0.99, 1.0], device=ratio.device)
    )
    ROWS.append(
        (
            task,
            float(ratio.mean().item()),
            float(q[0].item()),
            float(q[1].item()),
            float(q[2].item()),
            float(q[3].item()),
            float((ratio > 1.0).float().mean().item()),
        )
    )


si.Net._consolidate_current_task = _patched_consolidate

sys.argv = [
    "main.py",
    "--config",
    os.path.join(REPO, "configs/base.yaml"),
    "--config",
    os.path.join(REPO, "configs/models/til/si_lwf.yaml"),
    "--si_lwf_lambda",
    "0.0",
    "--n_epochs",
    "1",
    "--inner_steps",
    "2",
    "--samples_per_task",
    "-1",
    "--no-amp",
    "--no-save_checkpoints",
    "--single-seed",
    "--expt_name",
    "probe_si_epsilon",
]

import main  # noqa: E402

os.chdir(REPO)
main.main()

print("\n" + "=" * 78)
print("SI epsilon probe:  omega += W / (delta^2 + epsilon),  epsilon = 0.01")
print("=" * 78)
print("\ndelta^2 / epsilon at each consolidation (1.0 = delta^2 equals epsilon):")
print(
    f"{'task':>5s} {'mean':>11s} {'q50':>11s} {'q90':>11s} "
    f"{'q99':>11s} {'max':>11s} {'frac>1':>8s}"
)
for task, mean, q50, q90, q99, mx, frac in ROWS:
    print(
        f"{task:5d} {mean:11.3e} {q50:11.3e} {q90:11.3e} "
        f"{q99:11.3e} {mx:11.3e} {frac:8.4f}"
    )
print(
    "\nIf q99 is well below 1, the denominator is epsilon for essentially every\n"
    "parameter and SI's per-parameter normalisation is inert: omega becomes an\n"
    "unnormalised path integral scaled by 1/epsilon."
)
