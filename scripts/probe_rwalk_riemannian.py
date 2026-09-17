"""Is RWalk's Riemannian normalisation numerically live, and does Fisher matter?

RWalk's importance is ``F + s``, where ``s`` accumulates

    s_update = -grad * delta / (0.5 * F * delta^2 + eps)

The ``0.5 * F * delta^2`` denominator is the Riemannian part: it measures each
step in the Fisher metric rather than the Euclidean one, and it is the whole
difference between ``s`` and Synaptic Intelligence's path integral. But it is
damped by ``eps`` (0.01 by default), and ``F * delta^2`` is a product of two
already-tiny quantities. If it sits far below ``eps``, the denominator is
constant, ``s`` degenerates to ``(1/eps) * sum(-grad * delta)`` -- a plain
Euclidean path integral -- and RWalk's Riemannian claim is inert *at this
operating point*, whatever it does in principle.

This monkeypatches the update to record the ratio ``F * delta^2 / eps`` and the
share of the consolidated importance contributed by ``F`` versus ``s``, then
drives a short real-data run. Nothing here changes training; the patch only
observes.

Usage:
    la-maml_env/bin/python scripts/probe_rwalk_riemannian.py
"""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model import rwalk  # noqa: E402

RATIOS: list[tuple[float, float, float, float]] = []
SHARES: list[tuple[int, float, float, float, float]] = []

_original_update = rwalk.Net._update_running_statistics
_original_consolidate = rwalk.Net._consolidate_current_task


def _patched_update(self) -> None:
    """Record the Riemannian-vs-eps ratio, then run the real update."""
    numerators: list[torch.Tensor] = []
    for name, param in self.net.named_parameters():
        if not param.requires_grad or name.startswith("det_head"):
            continue
        if param.grad is None or name not in self.p_old:
            continue
        self._ensure_state_device(name, param)
        delta = param.detach() - self.p_old[name]
        numerators.append(
            (0.5 * self.fisher_running[name] * delta.pow(2)).reshape(-1).float()
        )
    if numerators:
        values = torch.cat(numerators) / self.eps
        # The max decides whether the metric term is ever live; the quantiles
        # decide how far eps must fall to make it live for a typical parameter,
        # which is what sets the range of an eps sweep.
        q = torch.quantile(
            values, torch.tensor([0.5, 0.9, 0.99], device=values.device)
        )
        RATIOS.append(
            (
                float(values.max().item()),
                float(q[0].item()),
                float(q[1].item()),
                float(q[2].item()),
            )
        )
    _original_update(self)


def _patched_consolidate(self) -> None:
    """Run the real consolidation, then decompose ``F + s``."""
    task = self.current_task
    _original_consolidate(self)
    if task is None:
        return
    fisher = torch.cat([t.reshape(-1).float() for t in self.fisher.values()])
    s_term = torch.cat([t.reshape(-1).float() for t in self.s.values()])
    total = (fisher + s_term).clamp(min=0)
    SHARES.append(
        (
            task,
            float(fisher.abs().sum().item()),
            float(s_term.abs().sum().item()),
            float(total.sum().item()),
            float((s_term < 0).float().mean().item()),
        )
    )


rwalk.Net._update_running_statistics = _patched_update
rwalk.Net._consolidate_current_task = _patched_consolidate

sys.argv = [
    "main.py",
    "--config",
    os.path.join(REPO, "configs/base.yaml"),
    "--config",
    os.path.join(REPO, "configs/models/til/rwalk_lwf.yaml"),
    "--rwalk_lwf_lambda",
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
    "probe_rwalk_riemannian",
]

import main  # noqa: E402

os.chdir(REPO)
main.main()

print("\n" + "=" * 72)
print("RWalk Riemannian probe")
print("=" * 72)
if RATIOS:
    print(f"\n0.5 * F_i * delta_i^2 / eps,  eps = 0.01, steps observed = {len(RATIOS)}")
    print("(per-step statistics over parameters, then aggregated over steps)\n")
    print(f"{'':14s}{'max':>12s}{'q50':>12s}{'q90':>12s}{'q99':>12s}")
    peak = [max(r[i] for r in RATIOS) for i in range(4)]
    typical = [sorted(r[i] for r in RATIOS)[len(RATIOS) // 2] for i in range(4)]
    print(
        f"{'worst step':14s}"
        + "".join(f"{v:12.3e}" for v in peak)
    )
    print(
        f"{'median step':14s}"
        + "".join(f"{v:12.3e}" for v in typical)
    )
    print(
        "\nThe q50 column is what sets the range of an eps sweep: eps must fall to\n"
        "roughly q50 * 0.01 before the metric term is live for a typical\n"
        "parameter. Until then s is a plain Euclidean path integral scaled 1/eps."
    )
print("\nconsolidated importance decomposition (sum of |values| over all params):")
print(f"{'task':>5s} {'sum|F|':>12s} {'sum|s|':>12s} {'F share':>9s} {'s<0 frac':>9s}")
for task, f_sum, s_sum, _total, neg in SHARES:
    share = f_sum / (f_sum + s_sum) if (f_sum + s_sum) > 0 else float("nan")
    print(f"{task:5d} {f_sum:12.4e} {s_sum:12.4e} {share:9.4f} {neg:9.4f}")
print(
    "\nF share is how much of the penalty's magnitude the Fisher term supplies.\n"
    "'s<0 frac' matters because the proximal path clamps negative importance at\n"
    "zero, so on those parameters F is the only thing left anchoring them."
)
