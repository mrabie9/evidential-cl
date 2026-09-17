#!/usr/bin/env python
"""Paired report on ResNet1D trunk dropout: none / flat 0.2 / 0.5-0.4-0.3-0.2.

The control arm is scripts/logs/i2_recheck (run 2026-08-31 at the then-current
flat p=0.2); the treatment arms are scripts/logs/dropout_schedule (decreasing)
and scripts/logs/dropout_none (off). The arms differ only in the four trunk
`nn.Dropout` probabilities -- model/resnet1d.py is the only tracked file touched
between the launches, and RESNET1D_DROPOUT pins that difference -- so the seeds
pair against the shared control.

Statistic: SUMMARY_TE cls_f1 (the `final` column every table in
docs/woe-cl/README.md reports), plus the diagonal and BWT off the `# val:` line.
"""

from __future__ import annotations

import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 39, 55]
CONTROL = "control(p=0.2)"
ARMS = {
    CONTROL: (os.path.join(HERE, "logs", "i2_recheck"), "i2recheck_lam240000_s{s}"),
    "sched(0.5-0.2)": (os.path.join(HERE, "logs", "dropout_schedule"), "dropsched_lam240000_s{s}"),
    "none(p=0)": (os.path.join(HERE, "logs", "dropout_none"), "dropnone_lam240000_s{s}"),
}

VAL = re.compile(r"# val: ([\d.\-]+) ([\d.\-]+) ([\d.\-]+)")
SUMMARY = re.compile(r"SUMMARY_TE .*?cls_f1=([\d.\-]+)")


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        text = fh.read()
    hits = VAL.findall(text)
    if not hits:
        return None
    diag, final, bwt = (float(v) for v in hits[-1])
    precise = SUMMARY.findall(text)
    if precise:
        final = float(precise[-1])
    return diag, final, bwt


LC = re.compile(
    r"\[LC\] consolidated task=(\d+) total_omega=([\d.e+\-]+) "
    r"task_omega=([\d.e+\-]+) accum=\w+ nonzero_frac=([\d.]+) "
    r"delta_rms=([\d.e+\-]+)"
)


def read_lc(path):
    """Per-task Omega bookkeeping the run already prints. Not a concentration
    statistic -- nonzero_frac only counts exact zeros -- but it does say whether
    the *scale* of the path integral moved, which is the first thing a dropout
    change would do to the anchor's operating point b = 2 lr lambda Omega."""
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        hits = LC.findall(fh.read())
    if not hits:
        return None
    return [(int(t), float(tot), float(tk), float(nz), float(dr)) for t, tot, tk, nz, dr in hits]


def paired_t(deltas):
    n = len(deltas)
    if n < 2:
        return float("nan")
    sd = st.stdev(deltas)
    if sd == 0.0:
        return float("inf") if st.mean(deltas) != 0 else 0.0
    return st.mean(deltas) / (sd / (n ** 0.5))


def main() -> int:
    rows = {}
    missing = []
    for arm, (logdir, pattern) in ARMS.items():
        rows[arm] = {}
        for seed in SEEDS:
            got = read(os.path.join(logdir, pattern.format(s=seed) + ".log"))
            if got is None:
                missing.append(f"{arm} s{seed}")
            else:
                rows[arm][seed] = got

    print(f"{'arm':<16} {'seed':>5} {'diag':>8} {'final':>8} {'bwt':>8}")
    for arm in ARMS:
        for seed in SEEDS:
            if seed in rows[arm]:
                d, f, b = rows[arm][seed]
                print(f"{arm:<16} {seed:>5} {d:>8.4f} {f:>8.4f} {b:>8.4f}")
        vals = [rows[arm][s][1] for s in SEEDS if s in rows[arm]]
        if len(vals) > 1:
            print(f"{arm:<16} {'mean':>5} {'':>8} {st.mean(vals):>8.4f} "
                  f"(sd {st.stdev(vals):.4f})")
    if missing:
        print("\nmissing:", ", ".join(missing))

    for arm in ARMS:
        if arm == CONTROL or len(rows[arm]) < len(SEEDS):
            continue
        print(f"\npaired deltas ({arm} - {CONTROL}), by seed:")
        for idx, label in ((0, "diag"), (1, "final"), (2, "bwt")):
            deltas = [rows[arm][s][idx] - rows[CONTROL][s][idx] for s in SEEDS]
            per_seed = "  ".join(f"s{s}{d:+.4f}" for s, d in zip(SEEDS, deltas))
            print(f"  {label:<6} {per_seed}   mean {st.mean(deltas):+.4f}  "
                  f"t={paired_t(deltas):+.2f}  (|t|>4.303 is p<0.05, df=2)")

    print("\nOmega bookkeeping (mean over tasks, mean over seeds):")
    print(f"  {'arm':<16} {'task_omega':>11} {'final_total':>12} "
          f"{'nonzero_frac':>13} {'delta_rms':>11}")
    for arm, (logdir, pattern) in ARMS.items():
        per_seed = [read_lc(os.path.join(logdir, pattern.format(s=s) + ".log"))
                    for s in SEEDS]
        per_seed = [r for r in per_seed if r]
        if not per_seed:
            continue
        tk = st.mean([st.mean([r[2] for r in run]) for run in per_seed])
        tot = st.mean([run[-1][1] for run in per_seed])
        nz = st.mean([st.mean([r[3] for r in run]) for run in per_seed])
        dr = st.mean([st.mean([r[4] for r in run]) for run in per_seed])
        print(f"  {arm:<16} {tk:>11.4f} {tot:>12.4f} {nz:>13.4f} {dr:>11.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
