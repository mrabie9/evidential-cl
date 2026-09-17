#!/usr/bin/env python3
"""Does headroom magnify the C-MAML mechanisms? Paired leave-one-out analysis of the M
block (M0-M4) under CIL on the 4-task stream (slots 0-3 of the canonical order), set
side by side with the same block on the full 10-task stream.

The 10-task CIL M block sits near the floor (M0 ~ 12.6 F1) and reads "only replay is
load-bearing". If the meta mechanisms are genuinely nil, their deltas stay ~0 when the
stream is shortened and the operating point rises. If instead they were compressed against
the floor, the deltas grow. This script reports both blocks and their difference.

Statistical protocol is the one in docs/ablation_studies.tex (sec:stat_protocol), reused
from scripts/analyse_cil_ablations.py: per-seed paired deltas vs the family baseline (M0),
mean with 95% Student-t CI, paired t-test with Holm-Bonferroni within the block, exact
sign-flip permutation test, verdict against a smallest-effect-of-interest of 1 F1 point.

Metric: canonical macro f1_total = results.txt "Final F1". BWT = results.txt "Backward".
FWT = mean pre-train zero-shot total_f1 over tasks 1..T-1 (the trained_zs term; the
untrained baseline is a per-seed constant that cancels in the paired delta).

Usage:  la-maml_env/bin/python scripts/analyse_cil_cmaml_t03.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_cil_ablations import (  # noqa: E402
    DELTA,
    IDX,
    allser,
    holm,
    paired_stats,
    parse_seed,
    series,
    verdict,
)

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
LOGS = os.path.join(REPO, "logs")
ABL = os.path.join(LOGS, "ablations", "cil")
SEEDS = [0, 1, 2, 3, 39, 55, 7, 13, 21]

LABELS = {
    "M0": "None (full method)",
    "M1": "Second-order meta-gradient (Hessian term)",
    "M2": "Episodic replay buffer",
    "M3": "Meta-batch averaging (K_meta = 1)",
    "M4": "Inner-loop adaptation (alpha_init = 0)",
}

# 4-task block: everything launched by scripts/run_cil_cmaml_t03.sh, three sibling run dirs
# per row (one per seed shard) pooled by expt_name glob.
T03 = {
    "M0": "cmaml/m0_split_t03_se_cil-*",
    "M1": "cmaml/m1_secondorder_split_t03_se_cil-*",
    "M2": "cmaml_no_replay/m2_noreplay_split_t03_se_cil-*",
    "M3": "cmaml_single_inner/m3_singleinner_split_t03_se_cil-*",
    "M4": "cmaml_alpha0/m4_alpha0_split_t03_se_cil-*",
}

# 10-task block. M0/M1/M3/M4 are the split-era reruns (scripts/run_cil_cmaml_split_rerun.sh)
# in logs/<stem>/, NOT the curated pre-split rows -- the 4-task block runs split, so the
# comparison has to be split-vs-split or the headroom effect is confounded with the ~2.7 F1
# the pooled CE costs. M2 has no split-era twin, and needs none: with memories: 0 both
# --cmaml_joint_er and --cmaml_replay_loss_mode are no-ops (meta_loss falls through when
# replay_count is 0), so the curated pre-split M2 IS the split-era configuration.
T10 = {
    "M0": "cmaml/m0_split_se_cil-*",
    "M1": "cmaml/m1_secondorder_split_se_cil-*",
    "M2": None,  # curated, see CURATED_T10
    "M3": "cmaml_single_inner/m3_singleinner_split_se_cil-*",
    "M4": "cmaml_alpha0/m4_alpha0_split_se_cil-*",
}
CURATED_T10 = {"M2": "cmaml/M2"}

ROWS = ["M0", "M1", "M2", "M3", "M4"]


def _forward_zs_ntasks(sd, n_tasks):
    """FWT trained_zs term over tasks 1..n_tasks-1 (analyse_cil_ablations hardcodes 1..9)."""
    md = os.path.join(sd, "metrics")
    vals = []
    for t in range(1, n_tasks):
        p = os.path.join(md, f"task{t}.npz")
        if not os.path.exists(p):
            return np.nan
        z = np.load(p, allow_pickle=True)
        if "zero_shot_total_f1" not in z.files:
            return np.nan
        vals.append(float(z["zero_shot_total_f1"]))
    return float(np.mean(vals)) if vals else np.nan


def load_glob(pattern, n_tasks, seeds=None):
    """Pool seed subdirs across every run dir matching logs/<pattern>. Later dirs win on a
    duplicate seed. Restricted to `seeds` when given, so a 29-seed arm can be cut down to
    the canonical 9 for a like-for-like contrast."""
    dirs = sorted(d for d in glob.glob(os.path.join(LOGS, pattern)) if os.path.isdir(d))
    return _pool(dirs, pattern, n_tasks, seeds)


def load_curated(relpath, n_tasks, seeds=None):
    dirs = sorted(
        d for d in glob.glob(os.path.join(ABL, relpath, "*")) if os.path.isdir(d)
    )
    return _pool(dirs, relpath, n_tasks, seeds)


def _pool(dirs, what, n_tasks, seeds):
    if not dirs:
        raise FileNotFoundError(f"no run dir matching {what}")
    seedmap = {}
    for d in dirs:
        for sd in sorted(os.listdir(d)):
            if not sd.isdigit():
                continue
            if seeds is not None and int(sd) not in seeds:
                continue
            if not os.path.exists(os.path.join(d, sd, "results.txt")):
                continue
            vals = list(parse_seed(d, int(sd)))
            vals[IDX["fwt"]] = _forward_zs_ntasks(os.path.join(d, str(sd)), n_tasks)
            seedmap[int(sd)] = tuple(vals)
    return seedmap


def block_stats(data):
    """Paired stats for M1-M4 vs M0 on their common seeds, Holm-corrected within block."""
    base = data["M0"]
    out = []
    for rid in ROWS[1:]:
        if rid not in data:
            continue
        amap = data[rid]
        common = sorted(set(base) & set(amap))
        if not common:
            continue
        sf = paired_stats(series(base, common, "f1"), series(amap, common, "f1"))
        sb = paired_stats(series(base, common, "bwt"), series(amap, common, "bwt"))
        out.append(dict(rid=rid, n=len(common), sf=sf, sb=sb, common=common))
    if out:
        adj = holm(np.array([a["sf"]["t_p"] for a in out]))
        for i, a in enumerate(out):
            a["holm"] = adj[i]
            a["verdict"] = verdict(a["holm"], a["sf"]["p_perm"], a["sf"]["ci"])
    return out


def print_block(title, data, stats_):
    print(f"\n===== {title} =====")
    print(
        f"{'ID':<4} {'F1':>12} {'BWT':>12} {'FWT':>12} {'n':>3} | "
        f"{'dF1 [95% CI]':>24} {'p_holm':>7} {'p_perm':>7} {'verdict':>13}"
    )
    b = data["M0"]
    print(
        f"{'M0':<4} {allser(b,'f1').mean():>7.2f}±{allser(b,'f1').std(ddof=1):<4.2f} "
        f"{allser(b,'bwt').mean():>7.2f}±{allser(b,'bwt').std(ddof=1):<4.2f} "
        f"{np.nanmean(allser(b,'fwt')):>7.2f}±{np.nanstd(allser(b,'fwt'),ddof=1):<4.2f} "
        f"{len(b):>3} | {'(baseline)':>24}"
    )
    for a in stats_:
        m = data[a["rid"]]
        ci = a["sf"]["ci"]
        d = f"{a['sf']['mean']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]"
        print(
            f"{a['rid']:<4} {allser(m,'f1').mean():>7.2f}±{allser(m,'f1').std(ddof=1):<4.2f} "
            f"{allser(m,'bwt').mean():>7.2f}±{allser(m,'bwt').std(ddof=1):<4.2f} "
            f"{np.nanmean(allser(m,'fwt')):>7.2f}±{np.nanstd(allser(m,'fwt'),ddof=1):<4.2f} "
            f"{a['n']:>3} | {d:>24} {a['holm']:>7.3f} {a['sf']['p_perm']:>7.3f} "
            f"{a['verdict']:>13}"
        )


def main():
    d4 = {rid: load_glob(p, 4, SEEDS) for rid, p in T03.items()}
    d10 = {}
    for rid in ROWS:
        if T10.get(rid):
            d10[rid] = load_glob(T10[rid], 10, SEEDS)
        elif rid in CURATED_T10:
            d10[rid] = load_curated(CURATED_T10[rid], 10, SEEDS)

    s4 = block_stats(d4)
    s10 = block_stats(d10)

    print_block("CIL, 4-task stream (slots 0-3)   F1/BWT/FWT in points", d4, s4)
    print_block("CIL, 10-task stream (full)       F1/BWT/FWT in points", d10, s10)

    # Headroom contrast. The two blocks are independent runs (different streams), so the
    # difference of deltas is compared with an unpaired se built from the two paired-delta
    # sds -- there is no per-seed pairing ACROSS streams beyond the shared seed labels, and
    # a seed does not mean the same thing once the task sequence changes.
    m4 = {a["rid"]: a for a in s4}
    m10 = {a["rid"]: a for a in s10}
    print(
        "\n===== Headroom contrast: does a higher operating point magnify each mechanism? ====="
    )
    print(f"M0 operating point: 4-task {allser(d4['M0'],'f1').mean():.2f} F1  vs  "
          f"10-task {allser(d10['M0'],'f1').mean():.2f} F1")
    print(
        f"\n{'ID':<4} {'mechanism':<42} {'d(4-task)':>11} {'d(10-task)':>11} "
        f"{'change':>9} {'verdict 4':>13} {'verdict 10':>13}"
    )
    for rid in ROWS[1:]:
        if rid not in m4 or rid not in m10:
            continue
        a, b = m4[rid], m10[rid]
        print(
            f"{rid:<4} {LABELS[rid]:<42} {a['sf']['mean']:>+11.2f} {b['sf']['mean']:>+11.2f} "
            f"{a['sf']['mean'] - b['sf']['mean']:>+9.2f} {a['verdict']:>13} {b['verdict']:>13}"
        )
    print(
        f"\nsmallest effect of interest delta = {DELTA:.1f} F1 point(s); "
        "'change' is d(4-task) - d(10-task), positive = the ablation costs LESS with headroom."
    )
    print(
        "NOTE: the replay budget is n_memories 5120 total in both blocks, so the 4-task "
        "stream gives each task 1280 slots vs 512. M2's cross-stream change therefore mixes "
        "headroom with a better-fed baseline; within-block verdicts are unaffected."
    )


if __name__ == "__main__":
    main()
