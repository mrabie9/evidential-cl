#!/usr/bin/env python3
"""Headroom sweep for every curated ablation block, TIL and CIL, computed from existing
runs by truncating their task x task matrix -- no rerun, no checkpoints.

THE QUESTION. A leave-one-out row that reads "inert" may be genuinely inert, or it may be
an artefact of an operating point pressed against the floor (acute in CIL, where the full
10-task baselines sit at 13-16 F1). Evaluating the SAME run after K tasks instead of 10
raises the operating point without touching the method, so sweeping K separates the two: a
mechanism whose effect is real but compressed grows as K falls; a mechanism that is nil
stays nil at every horizon.

HOW. results.txt stores the full matrix R (row r = after training task r, col c = F1 on
task c); metrics/metrics.py defines Final F1 = mean(R[T-1]) and Backward = mean(R[T-1] -
diag R). Restricting to R[K-1, :K] and diag(R)[:K] is exactly the run's state after K
tasks. See scripts/analyse_cil_cmaml_prefix.py for the C-MAML-only version.

WHAT IS HELD FIXED. Truncation keeps the label space (the head still spans all 10 tasks'
classes) and the replay budget (n_memories/10 per task) at their 10-task values, so the
only knob varied across K is the forgetting horizon. It is NOT a native K-task experiment.

CONFOUND REPORTING. Rows are paired against their family baseline on common seeds only.
Before the sweep, each row's training_parameters.json is diffed against the baseline's on
the knobs that move results (lr, amp, inner_steps, beta, ...) and any difference is printed.
Some differences ARE the ablation (M3 sets meta_batches 1; B5b sets inner_steps 4); others
are curation accidents that make a contrast multi-knob (til/gem G2 sits at lr 0.03 + AMP
against G0 at lr 0.01 + no-AMP). The script cannot tell them apart -- it prints them so a
reader can.

Usage:  la-maml_env/bin/python scripts/analyse_horizon_sweep.py [--ks 2,3,4,6,10] [--only cil/bcl-dual]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_cil_ablations import holm, paired_stats, verdict  # noqa: E402
from analyse_cil_cmaml_prefix import metrics_at_k, parse_matrix  # noqa: E402

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
LOGS = os.path.join(REPO, "logs")
ABL = os.path.join(LOGS, "ablations")

# Family baseline row (the "None (full method)" arm every other row is paired against).
BASELINE = {
    "res-er": "E0",
    "cmaml": "M0",
    "bcl-dual": "B0",
    "ctn": "T0",
    "gem": "G0",
    "gem_bob": "A0",
}

# Rows whose curated copy is superseded. The curated cil/cmaml M0/M1/M3/M4 predate commit
# 0b977d1c and ran the legacy pooled meta-loss CE, worth ~2.7 F1 of plasticity; the
# split-era reruns live outside the curated tree. Pairing curated M2 (memories: 0, so the
# flag is a no-op and the row is split-era by construction) against them is correct.
OVERRIDES = {
    ("cil", "cmaml", "M0"): "cmaml/m0_split_se_cil-*",
    ("cil", "cmaml", "M1"): "cmaml/m1_secondorder_split_se_cil-*",
    ("cil", "cmaml", "M3"): "cmaml_single_inner/m3_singleinner_split_se_cil-*",
    ("cil", "cmaml", "M4"): "cmaml_alpha0/m4_alpha0_split_se_cil-*",
}

# Knobs that change results; diffed against the baseline row and reported.
CFG_KEYS = [
    "lr",
    "opt_wt",
    "beta",
    "alpha_init",
    "inner_steps",
    "n_epochs",
    "amp",
    "meta_batches",
    "memories",
]

MARK = {"load-bearing": "*", "borderline": "~", "inert": "o", "inconclusive": "?"}


def seed_dirs(mode, fam, row):
    """Every <run>/<seed> directory for a row, following the optional lr* layer and
    pooling sibling run dirs (base run plus topups)."""
    key = (mode, fam, row)
    if key in OVERRIDES:
        pats = [os.path.join(LOGS, OVERRIDES[key], "*")]
    else:
        base = os.path.join(ABL, mode, fam, row)
        pats = [os.path.join(base, "*", "*"), os.path.join(base, "lr*", "*", "*")]
    out = []
    for pat in pats:
        for d in glob.glob(pat):
            if os.path.isdir(d) and os.path.basename(d).isdigit():
                if os.path.exists(os.path.join(d, "results.txt")):
                    out.append(d)
    return sorted(out)


def load_row(mode, fam, row, k):
    """{seed: metrics} at horizon k. Later run dirs win on a duplicate seed."""
    seedmap = {}
    for d in seed_dirs(mode, fam, row):
        seed = int(os.path.basename(d))
        _baseline, R = parse_matrix(os.path.join(d, "results.txt"))
        if R.shape[0] < k:
            continue
        seedmap[seed] = metrics_at_k(R, k, d)
    return seedmap


def row_config(mode, fam, row):
    """Union of CFG_KEYS values across a row's seeds, as {key: set(str)}."""
    vals = {}
    for d in seed_dirs(mode, fam, row):
        p = os.path.join(d, "training_parameters.json")
        if not os.path.exists(p):
            continue
        cfg = json.load(open(p))
        for kk in CFG_KEYS:
            vals.setdefault(kk, set()).add(str(cfg.get(kk)))
    return vals


def _norm(vals):
    """1 vs 1.0 vs '1' are the same setting; compare as floats where possible."""
    out = set()
    for v in vals:
        try:
            out.add(f"{float(v):g}")
        except (TypeError, ValueError):
            out.add(str(v))
    return out


def config_diff(base_cfg, row_cfg):
    diffs = []
    for kk in CFG_KEYS:
        b, r = _norm(base_cfg.get(kk, set())), _norm(row_cfg.get(kk, set()))
        if b and r and b != r:
            diffs.append(f"{kk} {'|'.join(sorted(b))}->{'|'.join(sorted(r))}")
    return diffs


def sweep_family(mode, fam, rows, ks):
    base_row = BASELINE[fam]
    base_cfg = row_config(mode, fam, base_row)
    others = [r for r in rows if r != base_row]

    per_k = {}
    for k in ks:
        data = {r: load_row(mode, fam, r, k) for r in rows}
        base = data[base_row]
        stats_ = []
        for r in others:
            common = sorted(set(base) & set(data[r]))
            if not common:
                continue
            sf = paired_stats(
                np.array([base[s]["f1"] for s in common]),
                np.array([data[r][s]["f1"] for s in common]),
            )
            stats_.append(dict(row=r, n=len(common), sf=sf))
        if stats_:
            adj = holm(np.array([a["sf"]["t_p"] for a in stats_]))
            for i, a in enumerate(stats_):
                a["verdict"] = verdict(adj[i], a["sf"]["p_perm"], a["sf"]["ci"])
        per_k[k] = (data, {a["row"]: a for a in stats_})

    print(f"\n===== {mode.upper()} / {fam} =====")
    hdr = "".join(f"{'K=' + str(k):>13}" for k in ks)
    print(f"{'row':<7}{'n':>4}  {'':<0}{hdr}")
    bcells = ""
    for k in ks:
        m = per_k[k][0][base_row]
        bcells += f"{np.mean([m[s]['f1'] for s in m]):>13.2f}"
    print(
        f"{base_row:<7}{len(per_k[ks[0]][0][base_row]):>4}  {bcells}   <- operating point (F1)"
    )
    for r in others:
        cells, n = "", 0
        for k in ks:
            a = per_k[k][1].get(r)
            if a is None:
                cells += f"{'--':>13}"
                continue
            n = a["n"]
            cells += f"{a['sf']['mean']:>+11.2f}{MARK[a['verdict']]:>2}"
        print(f"{r:<7}{n:>4}  {cells}")

    for r in others:
        d = config_diff(base_cfg, row_config(mode, fam, r))
        if d:
            print(f"    [cfg] {r} vs {base_row}: {'; '.join(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,3,4,6,10")
    ap.add_argument("--only", default=None, help="restrict to e.g. cil/bcl-dual")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    for mode in ["cil", "til"]:
        md = os.path.join(ABL, mode)
        for fam in sorted(os.listdir(md)):
            if not os.path.isdir(os.path.join(md, fam)):
                continue
            if args.only and args.only != f"{mode}/{fam}":
                continue
            rows = sorted(
                r
                for r in os.listdir(os.path.join(md, fam))
                if os.path.isdir(os.path.join(md, fam, r))
            )
            if BASELINE.get(fam) not in rows:
                print(
                    f"\n===== {mode}/{fam} ===== SKIPPED: no baseline {BASELINE.get(fam)}"
                )
                continue
            sweep_family(mode, fam, rows, ks)

    print(
        f"\nlegend: {MARK['load-bearing']} load-bearing   {MARK['borderline']} borderline"
        f"   {MARK['inert']} inert (CI inside ±1)   {MARK['inconclusive']} inconclusive"
    )
    print("values are paired dF1 (points) vs the family baseline, on common seeds.")
    print(
        "[cfg] lines list every knob differing from the baseline: some ARE the ablation,"
    )
    print(
        "others make the contrast multi-knob. Truncation holds label space and replay"
    )
    print("budget at their 10-task values, so K varies the forgetting horizon alone.")


if __name__ == "__main__":
    main()
