"""Collect the --woe_teacher_dropout 2x2x3 and run the paired tests.

Reads the `# val: <diag> <final> <bwt>` line each run writes into its terminal
log and pairs `keep` against `disable` by seed within each cell. `final` is the
macro F1 including noise (SUMMARY_TE cls_f1) -- the statistic every table in
docs/woe-cl/README.md reports, not cls_rec.
"""

from __future__ import annotations

import glob
import os
import re
import statistics as st
import sys

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "teacher_dropout")
SEEDS = [0, 39, 55]
CELLS = ["lwfonly", "both"]
ARMS = ["keep", "disable"]

VAL = re.compile(r"# val: ([\d.\-]+) ([\d.\-]+) ([\d.\-]+)")
# The `# val:` line rounds to 3dp, which is coarser than the effects this
# campaign resolves (B4's whole anchor gain is 0.0413). SUMMARY_TE carries the
# same `final` at 4dp, so prefer it and keep the val line only for diag/BWT.
SUMMARY = re.compile(r"SUMMARY_TE .*?cls_f1=([\d.\-]+)")


def read(name: str):
    """Return (diag, final, bwt) for one run, or None if it did not finish."""
    path = os.path.join(LOGS, f"{name}.log")
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


def paired_t(deltas):
    """Two-sided paired t on n=3; critical value at p<0.05 is 4.303 (df=2)."""
    n = len(deltas)
    if n < 2:
        return float("nan")
    sd = st.stdev(deltas)
    if sd == 0.0:
        return float("inf") if st.mean(deltas) != 0 else 0.0
    return st.mean(deltas) / (sd / (n**0.5))


def main() -> int:
    missing = []
    for cell in CELLS:
        rows = {}
        for arm in ARMS:
            vals = []
            for seed in SEEDS:
                name = f"td-{cell}-{arm}-s{seed}"
                got = read(name)
                if got is None:
                    missing.append(name)
                vals.append(got)
            rows[arm] = vals

        print(f"\n### cell: {cell}")
        print(f"{'seed':>6} | {'keep diag/final/bwt':>28} | {'disable diag/final/bwt':>28} | {'d final':>8}")
        print("-" * 84)
        deltas = {"diag": [], "final": [], "bwt": []}
        for i, seed in enumerate(SEEDS):
            k, d = rows["keep"][i], rows["disable"][i]
            if k is None or d is None:
                print(f"{seed:>6} | {'(incomplete)':>28} | {'(incomplete)':>28} |")
                continue
            for j, key in enumerate(("diag", "final", "bwt")):
                deltas[key].append(d[j] - k[j])
            ks = f"{k[0]:.4f} / {k[1]:.4f} / {k[2]:+.4f}"
            ds = f"{d[0]:.4f} / {d[1]:.4f} / {d[2]:+.4f}"
            print(f"{seed:>6} | {ks:>28} | {ds:>28} | {d[1]-k[1]:+.4f}")

        for key in ("diag", "final", "bwt"):
            ks = [v[{"diag": 0, "final": 1, "bwt": 2}[key]] for v in rows["keep"] if v]
            ds = [v[{"diag": 0, "final": 1, "bwt": 2}[key]] for v in rows["disable"] if v]
            if len(ks) < 2 or len(ds) < 2:
                continue
            wins = sum(1 for x in deltas[key] if x > 0)
            print(
                f"  {key:>5}: keep {st.mean(ks):.4f} +/- {st.stdev(ks):.4f} | "
                f"disable {st.mean(ds):.4f} +/- {st.stdev(ds):.4f} | "
                f"paired D {st.mean(deltas[key]):+.4f} +/- {st.stdev(deltas[key]):.4f} | "
                f"t = {paired_t(deltas[key]):+.2f} | disable wins {wins}/{len(deltas[key])}"
            )

    if missing:
        print("\nincomplete runs:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
