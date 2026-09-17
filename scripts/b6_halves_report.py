"""Collect the B6-halves grids: lambda response, Omega totals, peak per arm.

Reads scripts/logs/b6_halves/*.log and reports, per tracked scalar, the final
cls_f1 against lambda plus the measured cumulative Omega -- the latter because
the grid was placed from *shadow* Omegas measured on the i2 trajectory
(logit 42.23, conflict 91.98), and this project has mispredicted the optimal
lambda from an Omega ratio more than once. A peak at a grid edge means the
bracket failed and the arm needs extending, not reporting.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

LOGS = Path("/home/lunet/wsmr11/repos/La-MAML-EUCR/scripts/logs/b6_halves")
NAME = re.compile(r"b6h_(?P<scalar>[a-z0-9]+)_lam(?P<lam>[0-9.e+]+)_s(?P<seed>\d+)")
FINAL = re.compile(r"SUMMARY_TE .*?cls_f1=(?P<f1>[0-9.]+)")
OMEGA = re.compile(r"total_omega=(?P<tot>[0-9.e+-]+)")

# i2 control on this host (abs/proximal/sum, lambda 2.4e5), seeds 0/39/55.
I2 = {0: 0.5008, 39: 0.5039, 55: 0.4977}
SHADOW = {"logit": 42.23, "conflict": 91.98, "i2": 122.58}


def main() -> None:
    rows = defaultdict(list)
    for log in sorted(LOGS.glob("*.log")):
        m = NAME.search(log.name)
        if not m:
            continue
        text = log.read_text(errors="replace")
        finals = FINAL.findall(text)
        omegas = OMEGA.findall(text)
        rows[m["scalar"]].append(
            {
                "lam": float(m["lam"]),
                "seed": int(m["seed"]),
                "final": float(finals[-1]) if finals else None,
                "omega": float(omegas[-1]) if omegas else None,
                "done": bool(finals),
            }
        )

    for scalar in sorted(rows):
        arms = sorted(rows[scalar], key=lambda r: (r["seed"], r["lam"]))
        print(f"\n=== {scalar}   (shadow Omega {SHADOW.get(scalar, float('nan')):.2f})")
        print(
            f"{'lambda':>12} {'seed':>5} {'final':>8} {'Omega':>12} {'lam*Omega':>12}"
        )
        for r in arms:
            f = f"{r['final']:.4f}" if r["final"] is not None else "running"
            o = f"{r['omega']:.4e}" if r["omega"] is not None else "-"
            lo = f"{r['lam'] * r['omega']:.3e}" if r["omega"] is not None else "-"
            print(f"{r['lam']:>12.0f} {r['seed']:>5} {f:>8} {o:>12} {lo:>12}")

        done = [r for r in arms if r["done"] and r["seed"] == 0]
        if not done:
            continue
        peak = max(done, key=lambda r: r["final"])
        lams = sorted(r["lam"] for r in done)
        edge = peak["lam"] in (lams[0], lams[-1])
        print(
            f"  peak lambda={peak['lam']:.0f} final={peak['final']:.4f}"
            f"  delta vs i2 s0 = {peak['final'] - I2[0]:+.4f}"
        )
        if edge:
            print("  ** PEAK AT GRID EDGE -- not bracketed, extend before reporting **")
        if peak["omega"]:
            pred = SHADOW.get(scalar)
            if pred:
                print(f"  Omega vs shadow prediction: {peak['omega'] / pred:.2f}x")


if __name__ == "__main__":
    main()
