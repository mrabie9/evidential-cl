#!/usr/bin/env python3
"""Reader for the post-noise-removal ablation grids (TIL + CIL, 5-epoch and single-pass).

Pools every seed under logs/ablations_noise_removed/<mode>/<stem>/<ts>_<expt>/<seed>/ by
(stem, expt_name) and prints one block per family: Final F1 and BWT as mean +/- sample
stdev, plus the within-seed paired delta against the family baseline over the seeds the
two rows share. With --stats it adds the paired test protocol once a row has n >= 5.

Regimes are separate pools, selected with --tag: "nrm" is the 5-epoch grid that matches
the main experiments, "nrm1e" the single-pass replicate. Both run the same rows.

ROW -> POOL. An expt_name always denotes exactly one configuration, so pools are never
re-pointed and the table carries the mapping instead. C-MAML is anchored on the FIRST-ORDER
method (what the La-MAML paper recommends and what configs/models/*/cmaml.yaml ships), but
the driver's row ids were fixed when the anchor was second-order, so table row M0 reads
pool "m1" and row M1 -- which ADDS the second-order term rather than removing one -- reads
pool "m0". The "[ADDED]" tag marks every row whose delta is an addition, not a removal.

NOT IN THE GRID: rows that remove a memory -- G2 gem_noqp, M2 cmaml_no_replay, B3
bcl_nodualmem, B4 bcl_noreplay, T3 ctn_noreplay. Also unreferenced: the superseded
second-order m3/m4 pools, and every BCL run made before commit 7d1028c7 re-tuned the
BCL-Dual baselines (those are quarantined outside this tree, in
logs/ablations_noise_removed_stale_bclcfg/, and all ten BCL rows were rerun).

Metric: the HEADLINE macro F1 (macro over every class seen so far) from the results.txt
metric block -- identical to the row mean in TIL, ~7 points lower in CIL. The row mean
("Final F1") is printed beside it as a secondary column. Deltas and tests use the
headline. Seed protocol: n=6 in both modes, topped up once to n=12 for INC rows.

Usage:
  la-maml_env/bin/python scripts/analyse_ablations_noise_removed.py [--tag nrm|nrm1e] [--stats] [--csv]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_txt import f1_stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs", "ablations_noise_removed")
DELTA = 1.0  # smallest effect of interest, F1 points

# mode -> family -> [(row_id, mechanism, config stem, driver row id = pool key), ...]
# First entry of each family is its baseline. G1 has no job of its own: it is the same
# er_ring configuration as E1, so both rows read the E1 pool.
GRIDS = {
    "til": {
        "GEM": [
            ("G0", "None (full method)", "gem", "g0"),
            ("G1", "Gradient projection -> episodic replay (ring)", "er_ring", "e1"),
        ],
        "Res-ER": [
            ("E0", "None (full method)", "eralg4", "e0"),
            ("E1", "Reservoir -> static ring buffer", "er_ring", "e1"),
            ("E2", "Reservoir -> fully-utilised ring", "er_ring", "e2"),
        ],
        "C-MAML": [
            ("M0", "None (full method, first-order)", "cmaml", "m1"),
            ("M1", "[ADDED] Second-order term (Hessian)", "cmaml", "m0"),
            ("M3", "Meta-batch averaging (3 -> 1)", "cmaml_single_inner", "m3fo"),
            ("M4", "Inner-loop adaptation (alpha -> 0)", "cmaml_alpha0", "m4fo"),
        ],
        "BCL-Dual": [
            ("B0", "None (full method)", "bcl_dual", "b0"),
            ("B1", "KL distillation", "bcl_nodistill", "b1"),
            ("B2", "Bilevel interpolation (beta -> 1)", "bcl_nobilevel", "b2"),
            ("B5a", "Bilevel loop (half budget)", "bcl_singlelevel", "b5a"),
            ("B5b", "Bilevel loop (matched budget)", "bcl_singlelevel", "b5b"),
        ],
        "CTN": [
            ("T0", "None (full method)", "ctn", "t0"),
            ("T1", "FiLM task conditioning", "ctn_nofilm", "t1"),
            ("T2", "KL distillation", "ctn_nodistill", "t2"),
        ],
        "GEM-BOB": [
            ("A0", "None (all gates off; reproduces G0)", "gem_bob", "a0"),
            ("A1", "[ADDED] Fully-utilised (dynamic) ring", "gem_bob_dynring", "a1"),
            ("A2", "[ADDED] KL distillation", "gem_bob_distill", "a2"),
            ("C1", "[ADDED] Dynamic ring + KL distillation", "gem_bob_c1", "c1"),
        ],
    },
    "cil": {
        "Res-ER": [
            ("E0", "None (full method)", "eralg4", "e0"),
            ("E1", "Reservoir -> static ring buffer", "er_ring", "e1"),
            ("E2", "Reservoir -> fully-utilised ring", "er_ring", "e2"),
            ("C1", "[ADDED] KL distillation (Res-ER + distill)", "eralg4_distill", "c1"),
            ("C1b", "[ADDED] Same, HPT-tuned weight 0.5", "eralg4_distill_ms05", "c1b"),
        ],
        "C-MAML": [
            ("M0", "None (full method, first-order)", "cmaml", "m1"),
            ("M1", "[ADDED] Second-order term (Hessian)", "cmaml", "m0"),
            ("M3", "Meta-batch averaging (3 -> 1)", "cmaml_single_inner", "m3fo"),
            ("M4", "Inner-loop adaptation (alpha -> 0)", "cmaml_alpha0", "m4fo"),
        ],
        "BCL-Dual": [
            ("B0", "None (full method)", "bcl_dual", "b0"),
            ("B1", "KL distillation", "bcl_nodistill", "b1"),
            ("B2", "Bilevel interpolation (beta -> 1)", "bcl_nobilevel", "b2"),
            ("B5a", "Bilevel loop (half budget)", "bcl_singlelevel", "b5a"),
            ("B5b", "Bilevel loop (matched budget)", "bcl_singlelevel", "b5b"),
        ],
    },
}


_F1_ROW = re.compile(
    r"^f1\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$", re.M
)


def read_metrics(text):
    """(headline_f1, row_mean_f1, bwt) in points from one results.txt.

    PRIMARY METRIC IS THE HEADLINE: the macro F1 over every class seen so far, which is
    what the metric block's last column records. The row mean ("Final F1", the mean of the
    task matrix's last row) is kept as a secondary column. The two are identical in TIL,
    where each task is scored in its own label space, but in CIL the headline runs ~7
    points lower because it scores one macro average across the whole seen label space
    instead of averaging per-task scores -- so a CIL table built on the row mean flatters
    every method. BWT stays as recorded (final - diagonal), so it remains a row-mean
    quantity in both modes.
    """
    m = _F1_ROW.search(text)
    if m:
        _diag, final, bwt, _fwt, headline = (100.0 * float(g) for g in m.groups())
        return headline, final, bwt
    stats = f1_stats(text)  # legacy layout: no metric block, no headline
    if "Final F1" not in stats:
        return None
    final = 100.0 * stats["Final F1"]
    return final, final, 100.0 * stats.get("Backward", float("nan"))


def pool(mode, stem, pool_id, tag):
    """Return {seed: (headline_f1, row_mean_f1, bwt)}, newest run dir winning a duplicate."""
    out = {}
    expt = "{}_{}_{}".format(pool_id, tag, mode)
    pattern = os.path.join(LOGS, mode, stem, "*_" + expt, "*", "results.txt")
    for path in sorted(glob.glob(pattern)):  # timestamped dirs sort oldest first
        seed = int(os.path.basename(os.path.dirname(path)))
        with open(path) as handle:
            metrics = read_metrics(handle.read())
        if metrics is None:
            continue
        out[seed] = metrics
    return out


def fmt(values):
    """mean +/- sample stdev, or the bare value when n == 1."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return "{:5.2f}       ".format(arr[0])
    return "{:5.2f} +/- {:4.2f}".format(arr.mean(), arr.std(ddof=1))


def paired_stats(diffs):
    """Mean, 95% t CI, paired-t p and exact sign-flip permutation p for one row."""
    from itertools import product

    from scipy import stats as sps

    arr = np.asarray(diffs, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    sem = arr.std(ddof=1) / np.sqrt(n)
    half = sps.t.ppf(0.975, n - 1) * sem if sem > 0 else 0.0
    p_t = float(sps.ttest_1samp(arr, 0.0).pvalue) if sem > 0 else 1.0
    signs = np.array(list(product([1, -1], repeat=n)))
    null = np.abs((signs * arr).mean(axis=1))
    p_perm = float((null >= abs(mean) - 1e-12).mean())
    return mean, mean - half, mean + half, p_t, p_perm


def verdict(lo, hi, p_holm, p_perm):
    """Paper protocol (sec:stat_protocol) against the smallest effect of interest DELTA.

    SS: CI excludes 0 and p_Holm, p_perm < 0.05. MS: CI excludes 0 but a test misses 0.05.
    NS: CI inside +/-DELTA. INC: CI spans 0 and reaches past +/-DELTA.
    """
    if lo > 0 or hi < 0:
        return "SS" if (p_holm < 0.05 and p_perm < 0.05) else "MS"
    if lo > -DELTA and hi < DELTA:
        return "NS"
    return "INC"


def family_stats(rows, pools, min_n=5):
    """{row_id: (n, mean, lo, hi, p_holm, p_perm, verdict)} for every tested row.

    Paired against the family baseline (first row) over the seeds the two share; Holm is
    applied across the family's tested rows only.
    """
    base = pools[rows[0][0]]
    raw = {}
    for rid, _label, _stem, _pid in rows[1:]:
        data = pools[rid]
        common = [s_ for s_ in sorted(data) if s_ in base]
        if len(common) < min_n:
            continue
        diffs = [data[s_][0] - base[s_][0] for s_ in common]
        raw[rid] = (len(common),) + paired_stats(diffs)
    order = sorted(raw, key=lambda r: raw[r][4])  # ascending paired-t p
    m = len(order)
    holm, running = {}, 0.0
    for i, rid in enumerate(order):
        running = max(running, min(1.0, (m - i) * raw[rid][4]))
        holm[rid] = running
    out = {}
    for rid, (n, mean, lo, hi, _p_t, p_perm) in raw.items():
        out[rid] = (n, mean, lo, hi, holm[rid], p_perm, verdict(lo, hi, holm[rid], p_perm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nrm", help="regime pool tag: nrm (5 epochs) or nrm1e")
    ap.add_argument("--stats", action="store_true", help="paired tests for rows with n >= 5")
    ap.add_argument("--csv", action="store_true", help="machine-readable rows")
    ap.add_argument("--seeds", default=None,
                    help="restrict the report to these seeds (e.g. 0,39,55,100,390,550); "
                         "pools keep every seed on disk, this only filters the view")
    ap.add_argument("--topup", action="store_true",
                    help="list the driver row ids needing the n=12 top-up (INC rows plus "
                         "the family baselines their paired deltas are measured against)")
    ap.add_argument("--modes", default="til,cil", help="comma list of modes to report")
    args = ap.parse_args()

    modes = [m_ for m_ in args.modes.split(",") if m_ in GRIDS]

    if args.topup:
        for mode in modes:
            need = []
            for family, rows in GRIDS[mode].items():
                pools = {rid: pool(mode, stem, pid, args.tag) for rid, _l, stem, pid in rows}
                fstats = family_stats(rows, pools)
                hits = [pid for rid, _l, _s, pid in rows[1:]
                        if rid in fstats and fstats[rid][-1] == "INC"]
                if hits:
                    need.extend([rows[0][3]] + hits)
            uniq = sorted(set(need))
            print("{} {}: ROWS_ONLY={}".format(args.tag, mode, ",".join(uniq)))
        return

    if args.csv:
        print("mode,family,row,mechanism,n,seeds,headline_mean,headline_sd,rowmean_mean,bwt_mean,delta_headline,delta_n")

    for mode in modes:
        families = GRIDS[mode]
        if not args.csv:
            print("\n=== {} [{}] ===".format(mode.upper(), args.tag))
        for family, rows in families.items():
            pools = {rid: pool(mode, stem, pid, args.tag) for rid, _l, stem, pid in rows}
            if args.seeds:
                keep = {int(x) for x in args.seeds.split(",")}
                pools = {rid: {s_: v for s_, v in d.items() if s_ in keep}
                         for rid, d in pools.items()}
            base_id = rows[0][0]
            base = pools[base_id]
            fstats = family_stats(rows, pools)
            if not args.csv:
                print("\n{}  (baseline {}; F1 and BWT in points; Delta = paired vs {})".format(
                    family, base_id, base_id))
                header = ("  row  mechanism                                   n  seeds        "
                          "Headline F1       row-mean     BWT        Delta F1")
                if args.stats:
                    header += "      95% CI            p_Holm  p_perm  verdict"
                print(header)
            for rid, label, _stem, _pid in rows:
                data = pools[rid]
                if not data:
                    if not args.csv:
                        print("  {:4s} {:42s}  -  pending".format(rid, label[:42]))
                    continue
                seeds = sorted(data)
                f1 = [data[s][0] for s in seeds]        # headline (primary)
                rowmean = [data[s][1] for s in seeds]
                bwt = [data[s][2] for s in seeds]
                common = [s for s in seeds if s in base]
                diffs = [data[s][0] - base[s][0] for s in common]
                if rid == base_id or not common:
                    delta_txt, delta_val, delta_n = "  --", "", 0
                else:
                    delta_val = float(np.mean(diffs))
                    delta_n = len(common)
                    delta_txt = "{:+6.2f} (n={})".format(delta_val, delta_n)
                if args.csv:
                    arr = np.asarray(f1)
                    print("{},{},{},{},{},{},{:.3f},{},{:.3f},{:.3f},{},{}".format(
                        mode, family, rid, label.replace(",", ";"), len(seeds),
                        "|".join(str(s) for s in seeds), arr.mean(),
                        "{:.3f}".format(arr.std(ddof=1)) if arr.size > 1 else "",
                        float(np.mean(rowmean)), float(np.mean(bwt)),
                        "{:.3f}".format(delta_val) if delta_val != "" else "", delta_n))
                    continue
                line = "  {:4s} {:42s} {:2d}  {:12s} {}  {:6.2f}  {:7.2f}  {}".format(
                    rid, label[:42], len(seeds), ",".join(str(s) for s in seeds),
                    fmt(f1), float(np.mean(rowmean)), float(np.mean(bwt)), delta_txt)
                if args.stats and rid in fstats:
                    _n, _mean, lo, hi, p_holm, p_perm, verd = fstats[rid]
                    line += "   [{:+5.2f},{:+5.2f}]  {:6.3f}  {:6.3f}  {}".format(
                        lo, hi, p_holm, p_perm, verd)
                elif args.stats and delta_n:
                    line += "   (n={} < 5: no test)".format(delta_n)
                print(line)


if __name__ == "__main__":
    main()
