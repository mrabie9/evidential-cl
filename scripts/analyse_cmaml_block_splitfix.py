"""C-MAML ablation block (M0-M4) recomputed after the split-CE loss-reduction fix.

The pre-fix table rows were measured when lamaml_cifar.meta_loss took ONE class-weighted CE
over the concatenated replay+current batch. Inverse-frequency weights are computed on the
pooled batch, so old-task classes -- rare there -- let replay's share of the loss escalate
0.34 -> 0.85 across the 10 tasks, starving the current task. The fix splits the reduction
(--cmaml_replay_loss_mode split, now the default) and is worth ~+3 F1, all plasticity.
Every M row therefore needed re-running.

M0/M1/M3/M4 were re-run 2026-07-25 with --cmaml_joint_er (the two-forward loop) on top of
the fix. M2 (replay removed) was NOT re-run, and does not need to be: with an empty buffer
_combine_replay_current_loss returns the pooled loss unchanged (replay_count == 0), and
joint-ER likewise only governs how replay and current are combined. Both fixes are inert
when there is no replay, so M2's absolute numbers stand; only its delta must be re-paired
against the new M0 pool. See the caveat printed at the end.

All rows here are bf16 AMP, matching the GEM/BCL/CTN blocks of tab:ablation_optim.

Usage:
    la-maml_env/bin/python scripts/analyse_cmaml_block_splitfix.py
"""

from __future__ import annotations

import glob
import itertools
import math
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
DELTA = 1.0

BASELINE = ("M0", "None (full method)", "logs/ablations/til/cmaml/M0")
ROWS = [
    ("M1", "second-order meta-gradient", "logs/ablations/til/cmaml/M1"),
    ("M2", "episodic replay (NOT re-run)", "logs/ablations/til/cmaml/M2"),
    ("M3", "mini-batch averaging K_meta=1", "logs/ablations/til/cmaml/M3"),
    ("M4", "meta-learning (alpha=0)", "logs/ablations/til/cmaml/M4"),
]


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
    n = len(d)
    obs = abs(sum(d) / n)
    hits = sum(
        1
        for s in itertools.product((1, -1), repeat=n)
        if abs(sum(a * b for a, b in zip(s, d)) / n) >= obs - 1e-12
    )
    return hits / 2**n


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m2 = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m2 * (b - m2) * x) / ((a + 2 * m2 - 1) * (a + 2 * m2))
        else:
            num = -((a + m2) * (a + b + m2) * x) / ((a + 2 * m2) * (a + 2 * m2 + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def t_p(d: list[float]) -> float:
    n = len(d)
    m, sd = st.mean(d), st.stdev(d)
    if sd == 0:
        return 0.0
    t = abs(m) / (sd / math.sqrt(n))
    df = n - 1
    return _betainc(df / 2, 0.5, df / (df + t * t))


def verdict(lo: float, hi: float, ph: float, pp: float) -> str:
    if ph < 0.05 and pp < 0.05 and not (lo <= 0 <= hi):
        return "load-bearing"
    if not (lo <= 0 <= hi):
        return "borderline"
    if lo >= -DELTA and hi <= DELTA:
        return "inert"
    return "inconclusive"


def main() -> None:
    base = collect(BASELINE[2])
    f1b = [base[s]["Final F1"] for s in sorted(base)]
    bwtb = [base[s]["Backward"] for s in sorted(base)]
    print("C-MAML block, post split-CE fix, joint-ER, bf16 AMP, single-epoch TIL.\n")
    print(
        f"  M0  F1 {st.mean(f1b):5.2f}+-{st.stdev(f1b):4.2f}({len(f1b)})   "
        f"BWT {st.mean(bwtb):5.2f}+-{st.stdev(bwtb):4.2f}   {BASELINE[1]}"
    )

    raw = {}
    stats = {}
    for rid, desc, pat in ROWS:
        s = collect(pat)
        if not s:
            print(f"  {rid}  -- no runs found -- {desc}")
            continue
        f1 = [s[k]["Final F1"] for k in sorted(s)]
        bwt = [s[k]["Backward"] for k in sorted(s)]
        common = sorted(set(base) & set(s))
        d = [s[k]["Final F1"] - base[k]["Final F1"] for k in common]
        raw[rid] = d
        stats[rid] = (f1, bwt, common, desc)
        print(
            f"  {rid}  F1 {st.mean(f1):5.2f}+-{st.stdev(f1):4.2f}({len(f1)})   "
            f"BWT {st.mean(bwt):5.2f}+-{st.stdev(bwt):4.2f}   {desc}"
        )

    order = sorted(raw, key=lambda r: t_p(raw[r]))
    holm, prev = {}, 0.0
    for i, rid in enumerate(order):
        prev = max(prev, min(t_p(raw[rid]) * (len(order) - i), 1.0))
        holm[rid] = prev

    print("\n  row  dF1      95% CI             p_Holm   p_perm  verdict        n")
    for rid, _desc, _pat in ROWS:
        if rid not in raw:
            continue
        d = raw[rid]
        m = st.mean(d)
        half = 1.96 * st.stdev(d) / math.sqrt(len(d))
        lo, hi = m - half, m + half
        pp = perm_p(d)
        ph = holm[rid]
        pht = "<0.001" if ph < 0.001 else f"{ph:6.3f}"
        print(
            f"  {rid}  {m:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]   {pht:>6s}   {pp:6.3f}  "
            f"{verdict(lo, hi, ph, pp):14s} {len(d)}"
        )

    print(
        "\n  CAVEAT M2: pre-fix runs, paired against the post-fix M0 on common seeds.\n"
        "  The loss-reduction fix and joint-ER are both inert with an empty buffer, but the\n"
        "  M2 runs also predate --second_order (M1 measures that term as ~-0.1 F1)."
    )


if __name__ == "__main__":
    main()
