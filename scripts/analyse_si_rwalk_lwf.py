"""SI / RWalk 2x2: parameter-space anchor x LwF function-space regulariser.

Repeats the woe_si stacking test on the two other quadratic-anchor learners,
through the shared ``model/lwf_regulariser.py``. Reports each cell's pool and
the two paired contrasts that decide whether the mechanisms are additive:
LwF's effect with the anchor off vs with it on. If the two match, the anchor
and the distillation term are doing independent work; if the second collapses,
they are substitutes.

Runs come from scripts/run_si_rwalk_lwf.sh (single-epoch TIL, inner_steps 2,
seeds 0/39/55, --no-amp throughout).

Usage:
    la-maml_env/bin/python scripts/analyse_si_rwalk_lwf.py
"""

from __future__ import annotations

import glob
import itertools
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
METRICS = ["Final F1", "Diagonal F1", "Backward"]

# method -> cell -> results glob. "neither" is plain fine-tuning at that
# method's lr and comes from scripts/run_si_rwalk_lwf_naive.sh.
CELLS = {
    "si": {
        "neither": "logs/si_lwf/si_neither_noamp_se_til-*",
        "anchor only": "logs/si_lwf/si_anchoronly_noamp_se_til-*",
        "lwf only": "logs/si_lwf/si_lwfonly_noamp_se_til-*",
        "anchor + lwf": "logs/si_lwf/si_anchor_lwf_noamp_se_til-*",
    },
    "rwalk": {
        "neither": "logs/rwalk_lwf/rwalk_neither_noamp_se_til-*",
        "anchor only": "logs/rwalk_lwf/rwalk_anchoronly_noamp_se_til-*",
        "lwf only": "logs/rwalk_lwf/rwalk_lwfonly_noamp_se_til-*",
        "anchor + lwf": "logs/rwalk_lwf/rwalk_anchor_lwf_noamp_se_til-*",
    },
}


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in f1_stats(open(path).read()).items():
        if key in ('Diagonal F1', 'Final F1', 'Backward'):
            out[key] = value * 100
    return out


def collect(pattern: str) -> dict[int, dict[str, float]]:
    seeds: dict[int, dict[str, float]] = {}
    for base in sorted(glob.glob(os.path.join(REPO, pattern))):
        for dp, _d, files in os.walk(base):
            leaf = os.path.basename(dp)
            if "results.txt" in files and leaf.isdigit():
                rec = parse(os.path.join(dp, "results.txt"))
                if "Final F1" in rec:
                    seeds.setdefault(int(leaf), rec)
    return seeds


def perm_p(d: list[float]) -> float:
    """Exact two-sided sign-flip test; floor is 2^-(n-1), i.e. 0.25 at n=3."""
    n = len(d)
    if n == 0 or n > 20:
        return float("nan")
    obs = abs(sum(d) / n)
    hits = sum(
        1
        for s in itertools.product((1, -1), repeat=n)
        if abs(sum(a * b for a, b in zip(s, d)) / n) >= obs - 1e-12
    )
    return hits / 2**n


def fmt(vals: list[float]) -> str:
    if not vals:
        return "     --    "
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):6.2f}±{sd:4.2f}"


def paired(a: dict, b: dict, key: str) -> str:
    """``a`` minus ``b`` over the seeds both pools share."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return f"(n={len(common)})"
    d = [a[s][key] - b[s][key] for s in common]
    half = 1.96 * st.stdev(d) / len(d) ** 0.5
    return f"{st.mean(d):+5.2f} ±{half:4.2f} p={perm_p(d):.3f} n={len(d)}"


def main() -> None:
    data = {
        method: {cell: collect(pattern) for cell, pattern in cells.items()}
        for method, cells in CELLS.items()
    }
    print("SI / RWalk x LwF, single-epoch TIL, inner_steps 2, no AMP.")
    print("Percent, mean±sample sd.\n")

    for key in METRICS:
        print(f"--- {key} ---")
        header = f"{'method':7s}" + "".join(f"{c:>15s}" for c in CELLS["si"])
        print(header)
        for method, cells in data.items():
            row = "".join(
                f"{fmt([cells[c][s][key] for s in sorted(cells[c])]):>15s}"
                for c in CELLS[method]
            )
            print(f"{method:7s}{row}")
        print()

    print("--- is LwF additive with the anchor? (Final F1) ---")
    contrasts = [
        ("LwF alone     ", "lwf only", "neither"),
        ("LwF on anchor ", "anchor + lwf", "anchor only"),
        ("anchor alone  ", "anchor only", "neither"),
        ("anchor on LwF ", "anchor + lwf", "lwf only"),
    ]
    for method, cells in data.items():
        for label, plus, minus in contrasts:
            print(
                f"  {method:7s} {label} ({plus} minus {minus}): "
                f"{paired(cells[plus], cells[minus], 'Final F1')}"
            )
        print()

    print("  A mechanism is additive when its two rows agree: 'LwF alone' vs")
    print("  'LwF on anchor', and 'anchor alone' vs 'anchor on LwF'. A collapse")
    print("  in the second of a pair means the two are substitutes.")
    print("  p is the exact two-sided sign-flip test; its floor is 0.25 at n=3,")
    print("  so treat these as effect sizes, not significance claims.")


if __name__ == "__main__":
    main()
