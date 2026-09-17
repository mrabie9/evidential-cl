"""Attribute the C-MAML (M0) vs Res-ER (E0) single-epoch TIL gap.

Pools the probe rows launched by ``scripts/run_cmaml_reser_gap.sh`` and pairs
each of them, by seed, against the M0 baseline and the E0 target:

    M0          pre-fix C-MAML pool        pooled CE, opt_wt 0.01   (baseline)
    E0          pre-joint-ER eralg4 pool    eralg4 split CE          (target)
    split       --cmaml_replay_loss_mode split       (eralg4's exact reduction)
    split_norm  --cmaml_replay_loss_mode split_norm  (1:1 replay share, same scale)
    pooled_lr02 --opt_wt 0.02                        (2x step, pooled ratio kept)

Usage:
    la-maml_env/bin/python scripts/analyse_cmaml_reser_gap.py
"""

from __future__ import annotations

import glob
import itertools
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
METRICS = ["Final F1", "Diagonal F1", "Backward"]

# This probe measures the gap as it stood BEFORE the split-CE and joint-ER fixes, so its
# M0/E0 references are the pre-fix pools. Those were superseded as grid rows and returned
# to logs/<stem>/ by scripts/organise_ablations.py; they are listed run by run here.
ROWS = [
    ("M0 pooled CE (baseline)", [
        os.path.join(REPO, "logs/cmaml/cmaml_cwfix_so_se_til-*"),
        os.path.join(REPO, "logs/cmaml/cmaml_secondorder_s7-13-21-*"),
        os.path.join(REPO, "logs/cmaml/cmaml_secondorder_topup-*"),
    ]),
    ("E0 Res-ER (target)", [
        os.path.join(REPO, "logs/eralg4/eralg4_resER_s0-39-55-*"),
        os.path.join(REPO, "logs/eralg4/eralg4_resER_s7-13-21-*"),
        os.path.join(REPO, "logs/eralg4/eralg4_resER_topup-*"),
    ]),
    (
        "split  (eralg4 reduction)",
        os.path.join(REPO, "logs/cmaml/cmaml_splitloss_se_til-*"),
    ),
    (
        "split_norm (1:1, same scale)",
        os.path.join(REPO, "logs/cmaml/cmaml_splitnorm_se_til-*"),
    ),
    (
        "pooled + opt_wt 0.02",
        os.path.join(REPO, "logs/cmaml/cmaml_pooled_lr02_se_til-*"),
    ),
]


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in f1_stats(open(path).read()).items():
        if key in ('Diagonal F1', 'Final F1', 'Backward', 'Forward'):
            out[key] = value * 100
    return out


def collect(patterns: str | list[str]) -> dict[int, dict[str, float]]:
    """Seed -> metrics, walking either an ablation ID tree or globs of run dirs."""
    if isinstance(patterns, str):
        patterns = [patterns]
    seeds: dict[int, dict[str, float]] = {}
    for base in sorted(b for pat in patterns for b in (glob.glob(pat) or [pat])):
        for dirpath, _dirs, files in os.walk(base):
            leaf = os.path.basename(dirpath)
            if "results.txt" in files and leaf.isdigit():
                rec = parse(os.path.join(dirpath, "results.txt"))
                if "Final F1" in rec:
                    seeds.setdefault(int(leaf), rec)
    return seeds


def perm_p(diffs: list[float]) -> float:
    """Two-sided exact sign-flip permutation p on paired differences.

    Same test ``compare_ablation.py`` reports as ``p_perm``; with n pairs the
    smallest attainable value is 2 / 2**n (0.031 at n=6).
    """
    n = len(diffs)
    if n == 0:
        return 1.0
    observed = abs(sum(diffs) / n)
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, diffs)) / n) >= observed - 1e-12
    )
    return hits / 2**n


def fmt(values: list[float]) -> str:
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return f"{st.mean(values):5.2f}±{sd:4.2f}"


def main() -> None:
    data = {name: collect(pat) for name, pat in ROWS}
    print("Single-epoch TIL, C-MAML operating point (cmaml.yaml, --second_order,")
    print("inner_steps 2, opt_wt 0.01 unless stated). Percent, mean±sample sd.\n")
    print(f"{'row':30s} {'n':>2s}  {'F1':>11s}  {'Diag':>11s}  {'BWT':>11s}")
    print("-" * 74)
    for name, _ in ROWS:
        d = data[name]
        if not d:
            print(f"{name:30s}  0   (no results yet)")
            continue
        seeds = sorted(d)
        cols = [fmt([d[s][k] for s in seeds]) for k in METRICS]
        print(
            f"{name:30s} {len(seeds):2d}  {cols[0]:>11s}  {cols[1]:>11s}  {cols[2]:>11s}"
        )

    for ref in ["M0 pooled CE (baseline)", "E0 Res-ER (target)"]:
        print(f"\nPaired difference vs {ref} (probe minus reference):")
        base = data[ref]
        for name, _ in ROWS:
            if name == ref or not data[name]:
                continue
            common = sorted(set(data[name]) & set(base))
            if not common:
                continue
            parts = []
            for k in METRICS:
                diff = [data[name][s][k] - base[s][k] for s in common]
                sd = st.stdev(diff) if len(diff) > 1 else 0.0
                sem = sd / max(1, len(diff)) ** 0.5
                tag = f"{k.split()[0]:5s} {st.mean(diff):+5.2f} ±{1.96 * sem:4.2f}"
                if k == "Final F1" and len(diff) <= 20:
                    tag += f" (p={perm_p(diff):.3f})"
                parts.append(tag)
            print(f"  {name:30s} n={len(common)}  " + "  ".join(parts))
    print("\n  ± is a 95% normal CI on the paired mean difference; p is an exact")
    print("  two-sided sign-flip permutation test on the paired F1 differences")
    print("  (floor 2/2**n = 0.031 at n=6, 0.004 at n=9).")


if __name__ == "__main__":
    main()
