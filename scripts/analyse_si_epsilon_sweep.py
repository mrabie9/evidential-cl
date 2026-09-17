"""Does SI's per-parameter normalisation buy anything once epsilon lets it run?

Reads scripts/run_si_epsilon_sweep.sh. The eps=1e-2 column is the shipped
operating point and comes from the existing anchor-only cell of the LwF 2x2
(same config, same seeds, same flags), so it is a genuine paired reference.

`si_c` is compensated along the sweep to hold `si_c/epsilon` fixed, so the
penalty on a parameter still in the epsilon-dominated regime is unchanged and
only the discounting of high-displacement parameters varies. Thresholds from
scripts/probe_si_epsilon.py: epsilon ~2e-5 engages the top 1% of parameters,
~3e-7 engages the median. So the left of the sweep should reproduce the
reference and any movement should appear at 1e-6/1e-7.

The uncompensated arm is the control. It shares epsilon with the compensated
1e-6 arm but leaves si_c at 1000, so its anchor is 1e4x stronger. If the two
agree, the sweep is measuring normalisation shape; if the compensated arms are
flat and only the uncompensated one moves, the sweep is measuring strength.

Usage:
    la-maml_env/bin/python scripts/analyse_si_epsilon_sweep.py
"""

from __future__ import annotations

import glob
import itertools
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
METRICS = ["Final F1", "Diagonal F1", "Backward"]

# label -> results glob. Order is the sweep order; the first is the reference.
CELLS = {
    "1e-2 (shipped)": "logs/si_lwf/si_anchoronly_noamp_se_til-*",
    "1e-3": "logs/si_lwf/si_eps1em3_noamp_se_til-*",
    "1e-4": "logs/si_lwf/si_eps1em4_noamp_se_til-*",
    "1e-5": "logs/si_lwf/si_eps1em5_noamp_se_til-*",
    "1e-6": "logs/si_lwf/si_eps1em6_noamp_se_til-*",
    "1e-7": "logs/si_lwf/si_eps1em7_noamp_se_til-*",
    "1e-6 uncomp": "logs/si_lwf/si_eps1em6_uncomp_noamp_se_til-*",
    # eps=1e-6 re-tune (scripts/run_si_epsilon_retune.sh): the si_c ladder at
    # fixed epsilon, to locate normalised SI's own best operating point rather
    # than only its matched-strength one. si_c 0.1 is the "1e-6" row above and
    # si_c 1000 the "1e-6 uncomp" row.
    "1e-6 si_c=1": "logs/si_lwf/si_eps1em6_c1_noamp_se_til-*",
    "1e-6 si_c=10": "logs/si_lwf/si_eps1em6_c10_noamp_se_til-*",
}
REFERENCE = "1e-2 (shipped)"


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
    data = {label: collect(pattern) for label, pattern in CELLS.items()}
    print("SI si_epsilon sweep, anchor only, single-epoch TIL, no AMP.")
    print("si_c compensated to hold si_c/epsilon fixed except where noted.")
    print("Percent, mean±sample sd.\n")

    width = max(len(c) for c in CELLS) + 2
    for key in METRICS:
        print(f"--- {key} ---")
        for label in CELLS:
            pool = data[label]
            vals = [pool[s][key] for s in sorted(pool)]
            print(f"  {label:{width}s}{fmt(vals):>14s}   (n={len(vals)})")
        print()

    print(f"--- paired vs {REFERENCE} (Final F1) ---")
    for label in CELLS:
        if label == REFERENCE:
            continue
        print(f"  {label:{width}s}{paired(data[label], data[REFERENCE], 'Final F1')}")

    print(
        "\n  Flat across 1e-3..1e-5 is expected: too few parameters engage to\n"
        "  matter. Movement at 1e-6/1e-7 is the normalisation doing real work.\n"
        "  Compare '1e-6' against '1e-6 uncomp' to separate shape from strength.\n"
        "  p is the exact two-sided sign-flip test; its floor is 0.25 at n=3,\n"
        "  so treat these as effect sizes, not significance claims."
    )


if __name__ == "__main__":
    main()
