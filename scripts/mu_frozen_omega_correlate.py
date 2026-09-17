"""Spearman between the two mu arms' cumulative ``Omega``, at matched lambda (PR-3).

Reported as a **mechanism descriptor only**. Relating ``Omega`` agreement to
accuracy gaps is inadmissible on this project
(``omega-agreement-does-not-predict-accuracy``: the halves of ``I_2`` are
collinear at cos 0.996, so agreement between ``Omega`` fields carries no
implication about the accuracy difference between the arms). That ruling is not
suspended because the arms here differ by ``mu`` rather than by tracked scalar.

What it *does* answer: whether freezing ``mu`` changes which parameters the
anchor selects at all. A Spearman near 1.0 says the intervention is inert at the
level of the ingredient, which is the mechanism half of the reading -- it does
not license any statement about final F1.

Usage:
    python scripts/mu_frozen_omega_correlate.py \\
        scripts/logs/mu_frozen/omega_dumps/mufz_ema_lam240000.0_s0 \\
        scripts/logs/mu_frozen/omega_dumps/mufz_frozen_pretask_lam240000.0_s0
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def _load_task(dump_dir: Path, task: int) -> tuple[torch.Tensor, torch.Tensor]:
    path = dump_dir / f"omega_task{task:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["omega"], payload["task_omega"]


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rho via Pearson on ranks; average ranks for ties."""
    if a.numel() != b.numel():
        raise ValueError(f"length mismatch: {a.numel()} vs {b.numel()}")

    def _rank(x: torch.Tensor) -> torch.Tensor:
        """Average ranks, tie-corrected, vectorised (Omega is ~4M elements).

        Ties must be averaged rather than broken by index: Omega carries large
        exact-zero blocks, and index-breaking would manufacture agreement out of
        the parameter ordering rather than out of the values.
        """
        n = x.numel()
        order = torch.argsort(x)
        sorted_x = x[order]
        ranks_sorted = torch.arange(n, dtype=torch.float64)
        # Group boundaries: a new group starts wherever the sorted value changes.
        new_group = torch.ones(n, dtype=torch.bool)
        new_group[1:] = sorted_x[1:] != sorted_x[:-1]
        group_id = torch.cumsum(new_group.long(), 0) - 1
        n_groups = int(group_id[-1].item()) + 1
        sums = torch.zeros(n_groups, dtype=torch.float64).index_add_(
            0, group_id, ranks_sorted
        )
        counts = torch.zeros(n_groups, dtype=torch.float64).index_add_(
            0, group_id, torch.ones(n, dtype=torch.float64)
        )
        averaged = (sums / counts)[group_id]
        ranks = torch.empty(n, dtype=torch.float64)
        ranks[order] = averaged
        return ranks

    ra, rb = _rank(a.double()), _rank(b.double())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((ra @ rb) / denom)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    dir_a, dir_b = Path(argv[1]), Path(argv[2])
    print(f"A = {dir_a.name}")
    print(f"B = {dir_b.name}")
    print()
    print(
        f"{'task':>4}  {'rho(cumulative)':>16}  {'rho(per-task)':>14}  {'massA':>11}  {'massB':>11}"
    )
    for task in range(10):
        if not (dir_a / f"omega_task{task:02d}.pt").exists():
            break
        if not (dir_b / f"omega_task{task:02d}.pt").exists():
            break
        cum_a, per_a = _load_task(dir_a, task)
        cum_b, per_b = _load_task(dir_b, task)
        rho_cum = _spearman(cum_a, cum_b)
        rho_per = _spearman(per_a, per_b)
        print(
            f"{task:>4}  {rho_cum:>16.6f}  {rho_per:>14.6f}  "
            f"{float(cum_a.sum()):>11.4e}  {float(cum_b.sum()):>11.4e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
