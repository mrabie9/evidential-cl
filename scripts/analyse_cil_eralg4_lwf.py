#!/usr/bin/env python3
"""Paired analysis of Res-ER + LwF distillation vs the pinned CIL Res-ER baseline (E0).

Same protocol as scripts/analyse_cil_ablations.py -- per-seed paired deltas, mean effect
with 95% Student-t CI, and an exact sign-flip permutation p -- but for an *addition*
rather than a leave-one-out: the LwF arm differs from E0 only by --eralg4_lwf_lambda.
Metric is the canonical macro f1_total (results.txt "Final F1"); BWT and the zero-shot
forward-transfer term are reported alongside.

At n=3 no verdict is claimed: the per-seed deltas and their sign are the output, and the
repo's own rule is that paired effects here need n=9 before they are reported.
"""

import glob
import os
import sys
from itertools import product

import numpy as np
from scipy import stats
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
BASE_DIR = os.path.join(REPO, "logs", "ablations", "cil", "res-er", "E0")
LWF_GLOB = os.path.join(REPO, "logs", "eralg4_lwf", "eralg4_resER_joint_lwf_se_cil-*")


def _forward_zs(sd):
    """Mean pre-train zero-shot total_f1 over tasks 1..9 (the trained_zs FWT term)."""
    vals = []
    for t in range(1, 10):
        p = os.path.join(sd, "metrics", f"task{t}.npz")
        if not os.path.exists(p):
            return np.nan
        z = np.load(p, allow_pickle=True)
        if "zero_shot_total_f1" not in z.files:
            return np.nan
        vals.append(float(z["zero_shot_total_f1"]))
    return float(np.mean(vals)) if vals else np.nan


def parse_seed(sd):
    txt = open(os.path.join(sd, "results.txt")).read()
    return (
        f1_stats(txt)["Final F1"],
        f1_stats(txt)["Backward"],
        _forward_zs(sd),
    )


def load(pattern):
    """Pool seed subdirs over every matching run dir; later dirs win on a duplicate."""
    out = {}
    for d in sorted(glob.glob(pattern)):
        if not os.path.isdir(d):
            continue
        for sd in sorted(os.listdir(d)):
            path = os.path.join(d, sd)
            if sd.isdigit() and os.path.exists(os.path.join(path, "results.txt")):
                out[int(sd)] = parse_seed(path)
    return out


def perm_pvalue(delta):
    obs = abs(delta.mean())
    hits = sum(
        1
        for signs in product([1, -1], repeat=len(delta))
        if abs((np.array(signs) * delta).mean()) >= obs - 1e-12
    )
    return hits / (2 ** len(delta))


def main() -> int:
    base = load(os.path.join(BASE_DIR, "*"))
    lwf = load(LWF_GLOB)
    seeds = sorted(set(base) & set(lwf))
    if not seeds:
        print(f"no paired seeds (baseline {sorted(base)}, lwf {sorted(lwf)})")
        return 1

    print(f"Res-ER + LwF vs E0 (Res-ER), CIL, paired on seeds {seeds}\n")
    for i, name in enumerate(("Final F1", "BWT", "FWT (zero-shot)")):
        b = np.array([base[s][i] for s in seeds]) * 100
        a = np.array([lwf[s][i] for s in seeds]) * 100
        d = a - b
        print(f"{name}")
        for s, bv, av in zip(seeds, b, a):
            print(
                f"  seed {s:>2}: E0 {bv:6.2f}   +LwF {av:6.2f}   delta {av - bv:+6.2f}"
            )
        line = f"  mean:   E0 {b.mean():6.2f}   +LwF {a.mean():6.2f}   delta {d.mean():+6.2f}"
        if len(d) > 2 and d.std(ddof=1) > 0:
            se = d.std(ddof=1) / np.sqrt(len(d))
            half = stats.t.ppf(0.975, len(d) - 1) * se
            line += (
                f"  [{d.mean() - half:+.2f},{d.mean() + half:+.2f}]"
                f"  p_perm {perm_pvalue(d):.3f}"
            )
        print(line + "\n")
    print(
        f"n = {len(seeds)}; the repo's reporting threshold for a paired effect is n = 9."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
