#!/usr/bin/env python3
"""Headroom test for the CIL C-MAML M block, computed from the EXISTING 10-task runs by
truncating their task x task matrix -- no rerun, no checkpoints.

results.txt already stores the full lower-triangular matrix R (row r = after training task
r, column c = F1 on task c) plus the pre-training zero-shot baseline row. metrics/metrics.py
defines the published scalars as
    Diagonal F1 = mean(diag R)
    Final F1    = mean(R[T-1])
    Backward    = mean(R[T-1] - diag R)
    Forward     = mean_{t>=1}(R[t-1, t] - baseline[t])
Each restricts to the first K tasks by taking R[K-1, :K] and diag(R)[:K], which is exactly
the state of the run after it had seen K tasks. So the K=4 readout of an existing 10-task
run is free and already at n=9.

WHAT THIS DOES AND DOES NOT MEASURE. Truncating holds the label space and the replay budget
fixed at their 10-task values -- the head still spans all 10 tasks' classes at task 3, and
each task still gets n_memories/10 = 512 slots. So this isolates ONE knob, the forgetting
horizon: same problem, measured earlier, at a higher operating point. A native 4-task run
also shrinks the label space (fewer wrong classes to leak mass into) and enlarges the
per-task buffer to 1280; those raise the operating point further but are extra knobs. This
script is the cleaner single-knob test of "are the meta verdicts floor artefacts"; a native
4-task run answers the different question "what does the 4-task CIL problem look like".

Usage:  la-maml_env/bin/python scripts/analyse_cil_cmaml_prefix.py [K ...]
        (default K = 2 3 4 6 10)
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_cil_ablations import holm, paired_stats, verdict  # noqa: E402
from results_txt import f1_matrix

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
LOGS = os.path.join(REPO, "logs")
ABL = os.path.join(LOGS, "ablations", "cil")
SEEDS = [0, 1, 2, 3, 39, 55, 7, 13, 21]

LABELS = {
    "M0": "None (full method)",
    "M1": "Second-order meta-gradient",
    "M2": "Episodic replay buffer",
    "M3": "Meta-batch averaging (K_meta=1)",
    "M4": "Inner-loop adaptation (alpha=0)",
}
ROWS = ["M0", "M1", "M2", "M3", "M4"]

# Split-era reruns in logs/<stem>/ for M0/M1/M3/M4 (the curated pre-split rows would
# confound the horizon effect with the ~2.7 F1 the legacy pooled CE costs). M2 needs no
# split-era twin: with memories: 0 both flags are no-ops, so the curated row IS split-era.
GLOBS = {
    "M0": "cmaml/m0_split_se_cil-*",
    "M1": "cmaml/m1_secondorder_split_se_cil-*",
    "M3": "cmaml_single_inner/m3_singleinner_split_se_cil-*",
    "M4": "cmaml_alpha0/m4_alpha0_split_se_cil-*",
}
CURATED = {"M2": "cmaml/M2"}


def parse_matrix(path):
    """Return (baseline, R) from a results.txt. baseline is the pre-training zero-shot row
    printed above the '|' separator; R is the (T, T) matrix printed below it."""
    try:
        return f1_matrix(open(path).read())
    except ValueError as exc:
        raise ValueError(f"{exc} in {path}") from exc


def forward_zs(seed_dir, k):
    """FWT trained_zs term: mean pre-train zero-shot total_f1 over tasks 1..k-1, read from
    metrics/task{t}.npz. results.txt's own Forward line cannot be used -- unseen tasks are
    recorded as 0 in R and the baseline row is 0 too, so it is identically zero by
    construction. The untrained-model term of FWT is a per-seed constant common to every
    row, so it cancels in the paired delta."""
    md = os.path.join(seed_dir, "metrics")
    vals = []
    for t in range(1, k):
        p = os.path.join(md, f"task{t}.npz")
        if not os.path.exists(p):
            return np.nan
        z = np.load(p, allow_pickle=True)
        if "zero_shot_total_f1" not in z.files:
            return np.nan
        vals.append(float(z["zero_shot_total_f1"]))
    return float(np.mean(vals)) if vals else np.nan


def metrics_at_k(R, k, seed_dir):
    """The published scalars restricted to the first k tasks, in points."""
    diag = np.diag(R)[:k]
    fin = R[k - 1, :k]
    return dict(
        f1=100 * fin.mean(),
        diag=100 * diag.mean(),
        bwt=100 * (fin - diag).mean(),
        fwt=100 * forward_zs(seed_dir, k),
    )


def _pool(dirs, what, k):
    if not dirs:
        raise FileNotFoundError(f"no run dir matching {what}")
    seedmap = {}
    for d in dirs:
        for sd in sorted(os.listdir(d)):
            if not sd.isdigit() or int(sd) not in SEEDS:
                continue
            p = os.path.join(d, sd, "results.txt")
            if not os.path.exists(p):
                continue
            _baseline, R = parse_matrix(p)
            if R.shape[0] < k:
                continue
            seedmap[int(sd)] = metrics_at_k(R, k, os.path.join(d, sd))
    return seedmap


def load(rid, k):
    if rid in GLOBS:
        dirs = sorted(
            d for d in glob.glob(os.path.join(LOGS, GLOBS[rid])) if os.path.isdir(d)
        )
        return _pool(dirs, GLOBS[rid], k)
    dirs = sorted(
        d for d in glob.glob(os.path.join(ABL, CURATED[rid], "*")) if os.path.isdir(d)
    )
    return _pool(dirs, CURATED[rid], k)


def ser(seedmap, seeds, key):
    return np.array([seedmap[s][key] for s in seeds])


def block(k):
    data = {rid: load(rid, k) for rid in ROWS}
    base = data["M0"]
    out = []
    for rid in ROWS[1:]:
        common = sorted(set(base) & set(data[rid]))
        sf = paired_stats(ser(base, common, "f1"), ser(data[rid], common, "f1"))
        sb = paired_stats(ser(base, common, "bwt"), ser(data[rid], common, "bwt"))
        out.append(dict(rid=rid, n=len(common), sf=sf, sb=sb))
    adj = holm(np.array([a["sf"]["t_p"] for a in out]))
    for i, a in enumerate(out):
        a["holm"] = adj[i]
        a["verdict"] = verdict(a["holm"], a["sf"]["p_perm"], a["sf"]["ci"])
    return data, out


def print_block(k, data, stats_):
    allseeds = sorted(data["M0"])
    print(
        f"\n===== CIL C-MAML block, evaluated after K={k} tasks  (n={len(allseeds)}) ====="
    )
    print(
        f"{'ID':<4} {'F1':>13} {'BWT':>13} {'FWT':>13} | "
        f"{'dF1 [95% CI]':>24} {'p_holm':>7} {'p_perm':>7} {'verdict':>13}"
    )
    for rid in ROWS:
        m = data[rid]
        s = sorted(m)
        f1, bwt, fwt = ser(m, s, "f1"), ser(m, s, "bwt"), ser(m, s, "fwt")
        cell = (
            f"{rid:<4} {f1.mean():>7.2f}±{f1.std(ddof=1):<5.2f} "
            f"{bwt.mean():>7.2f}±{bwt.std(ddof=1):<5.2f} "
            f"{np.nanmean(fwt):>7.2f}±{np.nanstd(fwt, ddof=1):<5.2f} | "
        )
        if rid == "M0":
            print(cell + f"{'(baseline)':>24}")
            continue
        a = next(x for x in stats_ if x["rid"] == rid)
        ci = a["sf"]["ci"]
        d = f"{a['sf']['mean']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]"
        print(
            cell
            + f"{d:>24} {a['holm']:>7.3f} {a['sf']['p_perm']:>7.3f} {a['verdict']:>13}"
        )


def main():
    ks = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 6, 10]
    blocks = {}
    for k in ks:
        data, stats_ = block(k)
        blocks[k] = (data, stats_)
        print_block(k, data, stats_)

    print(
        "\n===== Does headroom magnify the mechanisms? paired dF1 vs M0 at each horizon ====="
    )
    hdr = "".join(f"{'K=' + str(k):>18}" for k in ks)
    print(f"{'ID':<4} {'mechanism':<34}{hdr}")
    base_cells = ""
    for k in ks:
        m0 = blocks[k][0]["M0"]
        base_cells += f"{np.mean([m0[s]['f1'] for s in m0]):>18.2f}"
    print(f"{'M0':<4} {'operating point (F1)':<34}{base_cells}")
    for rid in ROWS[1:]:
        cells = ""
        for k in ks:
            a = next(x for x in blocks[k][1] if x["rid"] == rid)
            mark = {
                "load-bearing": "*",
                "borderline": "~",
                "inert": "o",
                "inconclusive": "?",
            }[a["verdict"]]
            cells += f"{a['sf']['mean']:>+16.2f}{mark:>2}"
        print(f"{rid:<4} {LABELS[rid]:<34}{cells}")
    print(
        "\nlegend: * load-bearing   ~ borderline   o inert (CI inside ±1)   ? inconclusive"
    )
    print(
        "Truncation holds the label space (10-task head) and replay budget (512 slots/task)"
    )
    print("fixed, so the only knob varied across K is the forgetting horizon.")


if __name__ == "__main__":
    main()
