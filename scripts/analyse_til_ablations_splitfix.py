"""Compare the post-fix TIL ablation rows E0 and M0-M4 against their old pools.

Old pools are the pre-fix runs, logged before the 2026-07-25 meta-loss reduction fix
(commit 0b977d1c). They were the curated grid rows until the reruns replaced them, and
scripts/organise_ablations.py returned them to ``logs/<config-stem>/``; the new runs (now
the curated rows) come from ``scripts/run_til_ablations_splitfix.sh``.

Every row also carries the BN split (``--eralg4_joint_er`` on E0, its twin
``--cmaml_joint_er`` on the M rows), so the two families are matched on that axis.
Both old pools predate those flags, so no delta vs an old pool is a pure
reproduction: for the M rows it mixes the loss-reduction fix with the BN split,
for E0 it is the BN split alone. This grid therefore has no drift control. M2 was
dropped on request.

Usage:
    la-maml_env/bin/python scripts/analyse_til_ablations_splitfix.py
"""

from __future__ import annotations

import glob
import itertools
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
METRICS = ["Final F1", "Diagonal F1", "Backward"]

# id -> (label, pre-fix pool globs, post-fix pool (curated row or glob), is_control)
GRID = [
    (
        "E0",
        "Res-ER + joint-ER probe",
        [
            "logs/eralg4/eralg4_resER_s0-39-55-*",
            "logs/eralg4/eralg4_resER_s7-13-21-*",
            "logs/eralg4/eralg4_resER_topup-*",
        ],
        "logs/eralg4/e0_resER_joint_splitfix_se_til-*",
        False,
    ),
    (
        "M0",
        "C-MAML full (2nd order)",
        [
            "logs/cmaml/cmaml_cwfix_so_se_til-*",
            "logs/cmaml/cmaml_secondorder_s7-13-21-*",
            "logs/cmaml/cmaml_secondorder_topup-*",
        ],
        "logs/ablations/til/cmaml/M0",
        False,
    ),
    (
        "M1",
        "-- second order",
        [
            "logs/cmaml/cmaml_cwfix_se_til-*",
            "logs/cmaml/cmaml_full_s7-13-21-*",
        ],
        "logs/ablations/til/cmaml/M1",
        False,
    ),
    (
        "M3",
        "-- meta-batches",
        [
            "logs/cmaml_single_inner/cmaml_cwfix_singleinner_se_til-*",
            "logs/cmaml_single_inner/cmaml_singleinner_s7-13-21-*",
            "logs/cmaml_single_inner/cmaml_singleinner_topup-*",
        ],
        "logs/ablations/til/cmaml/M3",
        False,
    ),
    (
        "M4",
        "-- inner adaptation",
        [
            "logs/cmaml_alpha0/cmaml_cwfix_alpha0_se_til-*",
            "logs/cmaml_alpha0/cmaml_alpha0_s7-13-21-*",
            "logs/cmaml_alpha0/cmaml_alpha0_topup-*",
        ],
        "logs/ablations/til/cmaml/M4",
        False,
    ),
]


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in f1_stats(open(path).read()).items():
        if key in ('Diagonal F1', 'Final F1', 'Backward', 'Forward'):
            out[key] = value * 100
    return out


def collect(patterns: str | list[str]) -> dict[int, dict[str, float]]:
    if isinstance(patterns, str):
        patterns = [patterns]
    seeds: dict[int, dict[str, float]] = {}
    for base in sorted(b for pat in patterns
                       for b in (glob.glob(os.path.join(REPO, pat)) or [pat])):
        for dirpath, _dirs, files in os.walk(base):
            leaf = os.path.basename(dirpath)
            if "results.txt" in files and leaf.isdigit():
                rec = parse(os.path.join(dirpath, "results.txt"))
                if "Final F1" in rec:
                    seeds.setdefault(int(leaf), rec)
    return seeds


def perm_p(diffs: list[float]) -> float:
    """Two-sided exact sign-flip permutation p (the grid's usual p_perm)."""
    n = len(diffs)
    if n == 0 or n > 20:
        return float("nan")
    observed = abs(sum(diffs) / n)
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, diffs)) / n) >= observed - 1e-12
    )
    return hits / 2**n


def stat(values: list[float]) -> str:
    if not values:
        return "     --    "
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return f"{st.mean(values):5.2f}±{sd:4.2f}"


def verdict(diffs: list[float], delta: float = 1.0) -> str:
    """Same four-way rule ``compare_ablation.py`` uses on a paired difference.

    load-bearing  significant by the permutation test
    borderline    CI excludes 0 but the test is not significant
    inert         CI lies entirely within +/-delta, i.e. decisively small
    inconclusive  CI spans 0 AND exceeds +/-delta, i.e. underpowered

    Only the last two cases are ambiguous about the mechanism; TOP_UP names the
    ones that earn extra seeds.
    """
    if len(diffs) < 2:
        return "n/a"
    sd = st.stdev(diffs)
    half = 1.96 * sd / len(diffs) ** 0.5
    mean = st.mean(diffs)
    lo, hi = mean - half, mean + half
    if perm_p(diffs) < 0.05:
        return "load-bearing"
    if lo > 0 or hi < 0:
        return "borderline"
    if abs(lo) <= delta and abs(hi) <= delta:
        return "inert"
    return "inconclusive"


TOP_UP = {"inconclusive", "borderline"}


def main() -> None:
    print("TIL ablation rows re-run after the meta-loss reduction fix (0b977d1c).")
    print("Single-epoch, inner_steps 2. Percent, mean±sample sd.\n")
    header = f"{'row':4s} {'mechanism':24s} {'old F1':>12s} {'new F1':>12s} {'ΔF1 (paired)':>22s}"
    print(header)
    print("-" * len(header))

    new_pools: dict[str, dict[int, dict[str, float]]] = {}
    for rid, label, old_pat, new_pat, control in GRID:
        old, new = collect(old_pat), collect(new_pat)
        new_pools[rid] = new
        common = sorted(set(old) & set(new))
        if common:
            diff = [new[s]["Final F1"] - old[s]["Final F1"] for s in common]
            sd = st.stdev(diff) if len(diff) > 1 else 0.0
            sem = sd / len(diff) ** 0.5
            delta = f"{st.mean(diff):+5.2f} ±{1.96 * sem:4.2f} p={perm_p(diff):.3f} n={len(diff)}"
        else:
            delta = "(no new results yet)"
        tag = f"{rid}{'*' if control else ' '}"
        print(
            f"{tag:4s} {label:24s} "
            f"{stat([old[s]['Final F1'] for s in sorted(old)]):>12s} "
            f"{stat([new[s]['Final F1'] for s in sorted(new)]):>12s} {delta:>22s}"
        )
    print("\n  Every row adds the BN split (joint-ER), which both old pools predate,")
    print("  so no Δ above is a reproduction check and this grid has no drift control.")
    print("    ± is a 95% normal CI on the paired mean difference; p is the exact")
    print("    two-sided sign-flip test (floor 0.031 at n=6, 0.004 at n=9).")

    base = new_pools.get("M0", {})
    if not base:
        return
    print("\nPost-fix leave-one-out verdicts, paired against the new M0 baseline:")
    needs_top_up = []
    for rid, label, _old, _new, _control in GRID:
        if rid in ("E0", "M0") or not new_pools[rid]:
            continue
        common = sorted(set(new_pools[rid]) & set(base))
        if not common:
            continue
        parts = []
        for k in METRICS:
            diff = [new_pools[rid][s][k] - base[s][k] for s in common]
            sd = st.stdev(diff) if len(diff) > 1 else 0.0
            sem = sd / len(diff) ** 0.5
            parts.append(f"{k.split()[0]:5s} {st.mean(diff):+6.2f} ±{1.96 * sem:4.2f}")
        f1_diff = [new_pools[rid][s]["Final F1"] - base[s]["Final F1"] for s in common]
        v = verdict(f1_diff)
        if v in TOP_UP and len(common) < 9:
            needs_top_up.append(rid.lower())
        print(f"  {rid} {label:24s} n={len(common)}  " + "  ".join(parts) + f"  -> {v}")

    print("\n  Top-up rule: rows that come back load-bearing or inert are done at")
    print("  n=6; inconclusive or borderline rows earn seeds 1,2,3 for n=9.")
    if needs_top_up:
        print(f"\n  TOP UP: {', '.join(needs_top_up)} -- run")
        print(
            f"    SEED_GROUP_SPEC=\"1,2,3\" ROWS_ONLY={','.join(needs_top_up)} "
            "bash scripts/run_til_ablations_splitfix.sh"
        )
    else:
        print("\n  No row needs topping up.")


if __name__ == "__main__":
    main()
