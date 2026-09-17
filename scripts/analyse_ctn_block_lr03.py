"""CTN ablation block (T0-T3) at lr 0.03, CTN's native/tuned rate, for tab:ablation_optim.

The table previously reported CTN at lr 0.01, where its baseline is depressed
(fcl 53.2 vs 57.3 at 0.03) but per-seed variance is lower. This restores the native rate.

Every row reads the curated pool logs/ablations/til/ctn/<row>, which holds fp32 runs only:
each row also had a bf16 twin of the same expt name beside it, and pooling walk-order
decided which won (that is where the previously reported T2 fcl 51.5 came from; the fp32
pool gives 52.6). --lr lr0.01 reads the superseded lr replica, now back in logs/<stem>/.

Holm is corrected across the 3 ablation rows in the family, matching sec:stat_protocol.

Usage:
    la-maml_env/bin/python scripts/analyse_ctn_block_lr03.py [--lr lr0.01|lr0.03]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import math
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
DELTA = 1.0

ROWS = [
    ("T1", "FiLM task conditioning"),
    ("T2", "KL distillation term"),
    ("T3", "episodic replay"),
]

# config stem each row's runs were logged under, for the demoted lr0.01 replica
STEM = {"T0": "ctn", "T1": "ctn_nofilm", "T2": "ctn_nodistill", "T3": "ctn_noreplay"}


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in f1_stats(open(path).read()).items():
        if key in ('Diagonal F1', 'Final F1', 'Backward'):
            out[key] = value * 100
    return out


def collect(row: str, lr: str) -> dict[int, dict[str, float]]:
    """Seed -> metrics for one row. lr0.03 reads the curated pool; lr0.01 reads the
    superseded replica, which scripts/organise_ablations.py returned to logs/<stem>/."""
    if lr == "lr0.03":
        roots = [os.path.join(REPO, "logs/ablations/til/ctn", row)]
    else:
        roots = sorted(glob.glob(os.path.join(REPO, "logs", STEM[row], "*_lr01abl-*")))
    seeds: dict[int, dict[str, float]] = {}
    for root in roots:
        for dp, _d, files in os.walk(root):
            leaf = os.path.basename(dp)
            if "results.txt" in files and leaf.isdigit():
                rec = parse(os.path.join(dp, "results.txt"))
                if "Final F1" in rec:
                    seeds.setdefault(int(leaf), rec)
    return seeds


def perm_p(d: list[float]) -> float:
    n = len(d)
    obs = abs(sum(d) / n)
    return sum(
        1
        for s in itertools.product((1, -1), repeat=n)
        if abs(sum(a * b for a, b in zip(s, d)) / n) >= obs - 1e-12
    ) / 2**n


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", default="lr0.03", choices=["lr0.01", "lr0.03"])
    args = ap.parse_args()

    base = collect("T0", args.lr)
    f1b = [base[s]["Final F1"] for s in sorted(base)]
    bwtb = [base[s]["Backward"] for s in sorted(base)]
    print(f"CTN block at {args.lr}, single-epoch TIL, no AMP.\n")
    print(
        f"  T0  F1 {st.mean(f1b):5.2f}+-{st.stdev(f1b):4.2f}({len(f1b)})  "
        f"BWT {st.mean(bwtb):6.2f}+-{st.stdev(bwtb):4.2f}  None (full method)"
    )

    raw = {}
    for rid, desc in ROWS:
        s = collect(rid, args.lr)
        if not s:
            print(f"  {rid}  -- no runs -- {desc}")
            continue
        f1 = [s[k]["Final F1"] for k in sorted(s)]
        bwt = [s[k]["Backward"] for k in sorted(s)]
        common = sorted(set(base) & set(s))
        raw[rid] = [s[k]["Final F1"] - base[k]["Final F1"] for k in common]
        print(
            f"  {rid}  F1 {st.mean(f1):5.2f}+-{st.stdev(f1):4.2f}({len(f1)})  "
            f"BWT {st.mean(bwt):6.2f}+-{st.stdev(bwt):4.2f}  {desc}"
        )

    order = sorted(raw, key=lambda r: t_p(raw[r]))
    holm, prev = {}, 0.0
    for i, rid in enumerate(order):
        prev = max(prev, min(t_p(raw[rid]) * (len(order) - i), 1.0))
        holm[rid] = prev

    print("\n  row  dF1      95% CI             p_Holm   p_perm  verdict        n")
    for rid, _desc in ROWS:
        if rid not in raw:
            continue
        d = raw[rid]
        m = st.mean(d)
        half = 1.96 * st.stdev(d) / math.sqrt(len(d))
        lo, hi = m - half, m + half
        pp, ph = perm_p(d), holm[rid]
        pht = "<0.001" if ph < 0.001 else f"{ph:6.3f}"
        print(
            f"  {rid}  {m:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]   {pht:>6s}   {pp:6.3f}  "
            f"{verdict(lo, hi, ph, pp):14s} {len(d)}"
        )


if __name__ == "__main__":
    main()
