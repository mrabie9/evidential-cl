#!/usr/bin/env python
"""Is the DS content effective at p=0, or still inert?

Pairs, by seed, the no-dropout `i2` arm (scripts/logs/dropout_none) against the
two scalars that B6 leaves open once trunk dropout is removed:

  ce        -- no DS content at all. A tie means the DS-specific content is inert.
  conflict  -- the DS-specific half of I_2. Its Omega field is ~collinear with
               i2's (rho 0.975, cos 0.992 on a shared trajectory), so a large
               difference here would be surprising and would point at lambda
               rather than at content.

Also prints the seed-0 lambda bracket for `ce`, because the recurring failure in
this project is an n=3 result run at the wrong lambda.
"""

from __future__ import annotations

import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 39, 55]
REF = (
    "i2 (lam 2.4e5)",
    os.path.join(HERE, "logs", "dropout_none"),
    "dropnone_lam240000_s{s}",
)
ARMS = [
    ("ce (lam 75)", os.path.join(HERE, "logs", "ds_nodropout"), "dsnd_ce_lam75_s{s}"),
    (
        "conflict (lam 2.95e5)",
        os.path.join(HERE, "logs", "ds_nodropout"),
        "dsnd_conflict_lam295000_s{s}",
    ),
]
BRACKET = [
    ("ce lam 40", "dsnd_ce_lam40_s0"),
    ("ce lam 75", "dsnd_ce_lam75_s0"),
    ("ce lam 140", "dsnd_ce_lam140_s0"),
]

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


def paired_t(deltas):
    if len(deltas) < 2:
        return float("nan")
    sd = st.stdev(deltas)
    if sd == 0.0:
        return float("inf") if st.mean(deltas) != 0 else 0.0
    return st.mean(deltas) / (sd / (len(deltas) ** 0.5))


def collect(logdir, pattern):
    out = {}
    for seed in SEEDS:
        got = read(os.path.join(logdir, pattern.format(s=seed) + ".log"))
        if got is not None:
            out[seed] = got
    return out


def main() -> int:
    ref = collect(REF[1], REF[2])
    print(f"{'arm':<24}{'seed':>5}{'diag':>9}{'final':>9}{'bwt':>9}")
    for label, rows in [(REF[0], ref)] + [(a[0], collect(a[1], a[2])) for a in ARMS]:
        for seed in SEEDS:
            if seed in rows:
                d, f, b = rows[seed]
                print(f"{label:<24}{seed:>5}{d:>9.4f}{f:>9.4f}{b:>9.4f}")
        vals = [rows[s][1] for s in SEEDS if s in rows]
        if len(vals) > 1:
            print(
                f"{label:<24}{'mean':>5}{'':>9}{st.mean(vals):>9.4f}  (sd {st.stdev(vals):.4f}, n={len(vals)})"
            )

    for label, logdir, pattern in ARMS:
        rows = collect(logdir, pattern)
        common = [s for s in SEEDS if s in rows and s in ref]
        if len(common) < 2:
            print(f"\n[{label}] incomplete: have seeds {sorted(rows)}")
            continue
        print(f"\npaired deltas ({label} - {REF[0]}), n={len(common)}:")
        for idx, name in ((0, "diag"), (1, "final"), (2, "bwt")):
            deltas = [rows[s][idx] - ref[s][idx] for s in common]
            per = "  ".join(f"s{s}{d:+.4f}" for s, d in zip(common, deltas))
            print(
                f"  {name:<6}{per}   mean {st.mean(deltas):+.4f}  t={paired_t(deltas):+.2f}"
            )

    print("\nseed-0 lambda bracket for ce:")
    for label, name in BRACKET:
        got = read(os.path.join(HERE, "logs", "ds_nodropout", name + ".log"))
        if got:
            print(
                f"  {label:<14}diag {got[0]:.4f}  final {got[1]:.4f}  bwt {got[2]:.4f}"
            )
        else:
            print(f"  {label:<14}(missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
