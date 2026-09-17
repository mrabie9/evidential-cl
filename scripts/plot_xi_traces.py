#!/usr/bin/env python
"""Plot the ``[LC] consolidated`` traces across the xi sweep.

Five panels, all sharing the task axis except the last:

* per-task path integral (``task_omega``) -- what one task contributes
* cumulative ``Omega`` (``total_omega``) -- what the anchor actually holds
* ``delta_rms`` against ``sqrt(xi)`` per run. Note what that line *is*: at
  ``|Delta| = sqrt(xi)`` displacement supplies exactly **half** the denominator,
  so it marks 50% damping, not a floor.
* the fraction with ``Delta^2 < xi``, which is the same 50% line counted rather
  than drawn -- a loose upper bound on how much is genuinely floored, kept
  because it is what the logs recorded
* the honest version, computed offline from the dumped ``Delta^2``: the share of
  the denominator displacement actually supplies, ``Delta^2 / (Delta^2 + xi)``,
  as a median over parameters and weighted by ``Omega``. The gap between those
  two curves is the point -- dividing by ``Delta^2`` up-weights small-``Delta^2``
  parameters, so ``Omega`` concentrates on exactly the parameters still damped
* final F1 against lambda, one curve per xi, which is what the diagnostic
  panels have to explain

Colour encodes xi (the variable under test); line alpha encodes lambda within an
xi, so the three families separate at a glance without a 15-entry legend.

Usage:
    python scripts/plot_xi_traces.py --out /tmp/xi_traces.png
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LC = re.compile(
    r"\[LC\] consolidated task=(\d+) total_omega=(\S+) task_omega=(\S+) "
    r"accum=\S+ nonzero_frac=(\S+) delta_rms=(\S+) sqrt_xi=(\S+) "
    r"delta_floored_frac=(\S+) "
)
# Globs rather than one pattern, because the xi=1e-3 baseline predates the
# xifix naming and would otherwise be missing from the very comparison the plot
# exists to make. xi and lambda are read from training_parameters.json, which is
# the project's rule: trust the recorded config, never the expt_name.
GLOBS = (
    "logs/woe_si_lc/xifix_xi*/0",
    "logs/woe_si_lc/dsat_lam*/0",
    "logs/woe_si_lc/cn_sum_regression-*/0",
)


def collect() -> List[Dict]:
    runs = []
    seen = set()
    for pattern in GLOBS:
        for d in sorted(glob.glob(pattern)):
            cfg = Path(d) / "training_parameters.json"
            if not cfg.exists():
                continue
            import json

            params = json.loads(cfg.read_text())
            xi = float(params.get("woe_xi", 1e-3))
            lam = float(params.get("woe_lambda", 0.0))
            if (xi, lam) in seen:
                continue
            seen.add((xi, lam))
            log = Path(d) / "terminal.log"
            if not log.exists():
                continue
            rows = LC.findall(log.read_text())
            if not rows:
                continue
            res = Path(d) / "results.txt"
            final = None
            if res.exists():
                fm = re.search(r"^Final \S+: ([-\d.]+)", res.read_text(), re.M)
                final = float(fm.group(1)) if fm else None
            arr = np.array([[float(v) for v in r] for r in rows])
            runs.append(
                {
                    "xi": xi,
                    "lam": lam,
                    "task": arr[:, 0],
                    "total": arr[:, 1],
                    "task_omega": arr[:, 2],
                    "delta_rms": arr[:, 4],
                    "sqrt_xi": arr[0, 5],
                    "floored": arr[:, 6],
                    "final": final,
                }
            )
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/xi_traces.png")
    args = ap.parse_args()

    runs = collect()
    if not runs:
        raise SystemExit("no xifix runs with [LC] traces found")
    xis = sorted({r["xi"] for r in runs}, reverse=True)
    colours = {xi: c for xi, c in zip(xis, ["#1b3a5c", "#c2571a", "#7a1f4f", "#2d6a4f"])}

    fig, axgrid = plt.subplots(2, 3, figsize=(16.5, 8.6))
    axes = axgrid.ravel()
    fig.patch.set_facecolor("white")

    def alpha_for(r):
        lams = sorted({x["lam"] for x in runs if x["xi"] == r["xi"]})
        return 0.30 + 0.65 * lams.index(r["lam"]) / max(1, len(lams) - 1)

    for r in runs:
        c, a = colours[r["xi"]], alpha_for(r)
        axes[0].plot(r["task"], r["task_omega"], color=c, alpha=a, lw=1.3)
        axes[1].plot(r["task"], r["total"], color=c, alpha=a, lw=1.3)
        axes[2].plot(r["task"], r["delta_rms"], color=c, alpha=a, lw=1.3)
        axes[3].plot(r["task"], r["floored"], color=c, alpha=a, lw=1.3)

    for xi in xis:
        sq = next(r["sqrt_xi"] for r in runs if r["xi"] == xi)
        axes[2].axhline(sq, color=colours[xi], ls="--", lw=1.6, alpha=0.9)
        axes[2].annotate(
            f"50% damped ($|\\Delta|=\\sqrt{{\\xi}}$), $\\xi$={xi:g}",
            (0.02, sq),
            xycoords=("axes fraction", "data"),
            fontsize=8,
            color=colours[xi],
            va="bottom",
        )
        pts = sorted(
            [
                (r["lam"], r["final"])
                for r in runs
                if r["xi"] == xi and r["final"] and r["lam"] > 0
            ]
        )
        if pts:
            axes[5].plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                "o-",
                color=colours[xi],
                lw=1.8,
                ms=5,
                label=f"$\\xi$ = {xi:g}",
            )

    axes[5].axhline(0.5008, color="#444", ls=":", lw=1.8)
    axes[5].annotate(
        "$\\xi$=1e-3 baseline 0.5008 (n=3)",
        (0.03, 0.5008),
        xycoords=("axes fraction", "data"),
        fontsize=8,
        va="bottom",
        color="#444",
    )
    axes[5].set_xscale("log")
    axes[5].set_xlabel("woe_lambda")
    axes[5].set_ylabel("final F1")
    axes[5].set_title("outcome: final F1 vs $\\lambda$")
    axes[5].legend(frameon=False, fontsize=8, loc="lower center")

    for ax, (ylab, title, logy) in zip(
        axes[:4],
        [
            ("task_omega", "per-task path integral", True),
            ("total_omega", "cumulative $\\Omega$", True),
            ("delta_rms", "displacement vs the 50%-damping line", True),
            ("frac($\\Delta^2 < \\xi$)", "fraction at least 50% damped", False),
        ],
    ):
        ax.set_xlabel("task")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
    axes[3].set_ylim(-0.03, 1.03)

    # Panel 4: what share of the denominator displacement really supplies.
    dump = Path("scripts/logs/omega_dumps/xiparts_s0")
    files = sorted(dump.glob("omega_task*.pt"))
    if files:
        import torch

        blobs = [torch.load(f, map_location="cpu") for f in files]
        dsq = np.stack([b["delta_sq"].numpy() for b in blobs]).ravel()
        numer = np.stack([b["numerator"].numpy() for b in blobs]).ravel()
        grid = np.logspace(-12, -2, 45)
        med = [float(np.median(dsq / (dsq + x))) for x in grid]
        wtd = []
        for x in grid:
            w = numer / (dsq + x)
            wtd.append(float(((dsq / (dsq + x)) * w).sum() / w.sum()))
        axes[4].plot(grid, med, color="#1b3a5c", lw=2.2, label="median parameter")
        axes[4].plot(
            grid, wtd, color="#c2571a", lw=2.2, ls="--", label="weighted by $\\Omega$"
        )
        for xi in xis:
            axes[4].axvline(xi, color=colours[xi], lw=1.2, alpha=0.55)
        axes[4].set_xscale("log")
        axes[4].set_xlabel("$\\xi$")
        axes[4].set_ylabel("$\\Delta^2/(\\Delta^2+\\xi)$")
        axes[4].set_title("share of denominator from displacement")
        axes[4].set_ylim(-0.03, 1.03)
        axes[4].legend(frameon=False, fontsize=8, loc="upper left")
        axes[4].annotate(
            "$\\Omega$ up-weights small $\\Delta^2$,\nso the weighted share lags the\n"
            "median by ~4 decades of $\\xi$\n(+0.04 per decade; 0.48 at $10^{-11}$)",
            (0.30, 0.55),
            xycoords="axes fraction",
            fontsize=8,
            color="#c2571a",
        )
        axes[4].axvline(
            float(np.median(dsq)), color="#555", ls=":", lw=1.4
        )
        axes[4].annotate(
            "median $\\Delta^2$",
            (float(np.median(dsq)), 0.02),
            xycoords=("data", "axes fraction"),
            fontsize=8,
            color="#555",
            rotation=90,
            va="bottom",
            ha="right",
        )

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(
        "SI damping constant $\\xi$: what the anchor sees, and what it costs\n"
        "colour = $\\xi$, opacity = $\\lambda$ within each $\\xi$",
        fontsize=13,
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(runs)} runs, xi = {xis})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
