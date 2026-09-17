#!/usr/bin/env python
"""Paired significance tests + pooled mean/std for the TIL ablation grid.

Reads the curated tree logs/ablations/til/<algo>/<ID>/[lrX/]<run>/<seed>/results.txt
(see scripts/organise_ablations.py). Each ID folder holds exactly the runs backing that
row of tab:ablation_optim, so pooling is just "gather every seed under this ID, newest
run wins per seed" -- no config signature matching needed.

Precision and learning rate per family are properties of the curated rows, not options
here: GEM and Res-ER are fp32 at lr 0.01, CTN is fp32 at its native lr 0.03, C-MAML and
BCL-Dual are bf16 at their tuned configs. The two rows that were deliberately not re-run
under the current settings -- G2 (bf16, lr 0.03) and M2 (pre-split-CE) -- keep their old
pools because their effects are ~-35 F1, orders larger than either correction; their
deltas therefore pair across that difference, as the table notes state.

Reported per row (canonical metric = results.txt "Final F1" = f1_total):
  * F1  mean +/- sample std (n)  and  BWT mean +/- sample std   -- pooled
  * dF1 vs family baseline, paired over common seeds: 95% CI, paired-t p,
    exact sign-flip permutation p, Cohen's d_z
  * Holm-Bonferroni adjusted p across each family's ablations

Usage:  la-maml_env/bin/python scripts/compare_ablation.py [--verbose]
"""
import argparse
import glob
import math
import os
from itertools import product

from scipy import stats
from results_txt import f1_stats

LOG = "/home/lunet/wsmr11/repos/evidential-cl/logs"
ABL = f"{LOG}/ablations/til"

DELTA = 1.0  # smallest effect of interest, F1 points


def verdict(ph, pp, lo, hi, delta=DELTA):
    """Combined verdict from Holm p, permutation p, and 95% CI (lo/hi in F1 pts)."""
    if ph == ph and pp == pp and ph < 0.05 and pp < 0.05:
        return "load-bearing"
    if lo > 0 or hi < 0:
        return "borderline"
    if lo >= -delta and hi <= delta:
        return "inert"
    return "inconclusive"


FAMILIES = {
    "GEM": {"baseline": ("G0", "gem/G0/lr0.01"),
            "ablations": [("G1", "gem/G1/lr0.01"), ("G2", "gem/G2/lr0.03")]},
    "Res-ER": {"baseline": ("E0", "res-er/E0"),
               "ablations": [("E1", "res-er/E1"), ("E2", "res-er/E2")]},
    "C-MAML": {"baseline": ("M0", "cmaml/M0"),
               "ablations": [("M1", "cmaml/M1"), ("M2", "cmaml/M2"),
                             ("M3", "cmaml/M3"), ("M4", "cmaml/M4")]},
    "BCL-Dual": {"baseline": ("B0", "bcl-dual/B0"),
                 "ablations": [("B1", "bcl-dual/B1"), ("B3", "bcl-dual/B3"),
                               ("B4", "bcl-dual/B4"), ("B5a", "bcl-dual/B5a"),
                               ("B5b", "bcl-dual/B5b")]},
    "CTN": {"baseline": ("T0", "ctn/T0"),
            "ablations": [("T1", "ctn/T1"), ("T2", "ctn/T2"), ("T3", "ctn/T3")]},
}


def parse_results(seeddir):
    text = open(os.path.join(seeddir, "results.txt"), errors="ignore").read()
    f1_values = f1_stats(text)
    return f1_values.get("Final F1"), f1_values.get("Backward")


def arm_pool(relpath):
    """Pool per-seed (f1, bwt) across every run under logs/ablations/<relpath>.
    Newest run wins per seed. Returns (pool, None, contributors) to match the old
    signature (pool: seed -> (f1, bwt, mtime, dir))."""
    pool = {}
    for sd in glob.glob(f"{ABL}/{relpath}/*/*"):
        seed = os.path.basename(sd)
        if not (seed.isdigit() and os.path.isdir(sd)
                and os.path.exists(os.path.join(sd, "results.txt"))):
            continue
        f1, bwt = parse_results(sd)
        if f1 is None:
            continue
        mt = os.path.getmtime(sd)
        s = int(seed)
        if s not in pool or mt > pool[s][2]:
            pool[s] = (f1, bwt, mt, sd)
    contributors = {}
    for s, (_, _, _, sd) in pool.items():
        contributors.setdefault(os.path.basename(os.path.dirname(sd)), []).append(s)
    return pool, None, sorted(contributors.items())


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return mu, sd, n


def paired(base_pool, abl_pool, key=0):
    seeds = sorted(set(base_pool) & set(abl_pool))
    d = [abl_pool[s][key] - base_pool[s][key] for s in seeds]
    n = len(d)
    if n < 2:
        return dict(n=n, seeds=seeds, mean=(d[0] if d else float("nan")),
                    ci=(float("nan"),) * 2, p_t=float("nan"),
                    p_perm=float("nan"), dz=float("nan"))
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    sem = sd / math.sqrt(n) if sd else 0.0
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mu - tcrit * sem, mu + tcrit * sem)
    p_t = float(stats.ttest_rel([abl_pool[s][key] for s in seeds],
                                [base_pool[s][key] for s in seeds]).pvalue)
    obs = abs(mu)
    perm = [abs(sum(sgn * x for sgn, x in zip(signs, d)) / n)
            for signs in product((1, -1), repeat=n)]
    p_perm = sum(1 for v in perm if v >= obs - 1e-12) / len(perm)
    dz = mu / sd if sd else float("inf")
    return dict(n=n, seeds=seeds, mean=mu, ci=ci, p_t=p_t, p_perm=p_perm, dz=dz)


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        a = min(1.0, (m - rank) * pvals[i])
        running = max(running, a)
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print contributing run dirs")
    args = ap.parse_args()
    fams = FAMILIES

    print(f"tab:ablation_optim, single-epoch TIL. Each row reads its curated pool under "
          f"{os.path.relpath(ABL, LOG)}/. delta={DELTA} F1 pt.\n")
    print(f"{'row':4s} {'ablation':16s} {'F1 mean±std (n)':20s} {'ΔF1':7s} {'95% CI':16s} "
          f"{'p_holm':7s} {'p_perm':7s} {'verdict':13s}")
    print("-" * 96)

    for fam, spec in fams.items():
        bid, bpath = spec["baseline"]
        bpool, _, bcontrib = arm_pool(bpath)
        f1m, f1s, n = mean_std([v[0] for v in bpool.values()])
        print(f"\n== {fam} ==")
        print(f"{bid:4s} {bpath:16s} {f1m*100:5.1f}±{f1s*100:3.1f} (n={n})     "
              f"{'—':7s} {'—':16s} {'—':7s} {'—':7s} baseline")
        if args.verbose:
            print(f"        pooled from: {bcontrib}")

        rows, p_ts = [], []
        for aid, apath in spec["ablations"]:
            apool, _, acontrib = arm_pool(apath)
            f1m, f1s, n = mean_std([v[0] for v in apool.values()])
            st = paired(bpool, apool, key=0)
            rows.append((aid, apath, f1m, f1s, n, st, acontrib))
            p_ts.append(st["p_t"])

        adj = holm([p if p == p else 1.0 for p in p_ts])
        for (aid, apath, f1m, f1s, n, st, acontrib), ph in zip(rows, adj):
            lo, hi = st["ci"]
            v = verdict(ph, st["p_perm"], lo * 100, hi * 100) if st["n"] >= 2 else "n<2"
            print(f"{aid:4s} {apath:16s} {f1m*100:5.1f}±{f1s*100:3.1f} (n={n})     "
                  f"{st['mean']*100:+6.1f}  [{lo*100:+5.1f},{hi*100:+5.1f}]  "
                  f"{ph:6.3f}  {st['p_perm']:6.3f}  {v:13s}")
            if args.verbose:
                print(f"        pooled from: {acontrib}")

    print(f"\n  verdict: load-bearing (p_holm&p_perm<0.05) / borderline (CI excludes 0) "
          f"/ inert (CI within ±{DELTA}) / inconclusive (CI spans 0 & exceeds ±{DELTA}).")


if __name__ == "__main__":
    main()
