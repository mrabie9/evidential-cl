#!/usr/bin/env python3
"""Summarise a seed-sweep log directory: mean and sample std over seeded runs.

Given a generic run directory that contains one sub-directory per seed
(e.g. ``logs/lamaml/post-fix_me_til-2026-07-07_23-03-15-6051`` with ``0/``,
``39/``, ``55/`` inside), this reads each seed's ``terminal.log`` and
aggregates the final metrics printed on the ``SUMMARY_TE`` / ``SUMMARY_TR``
lines.

The headline number is **signal-class F1**, computed as the harmonic mean of
``cls_rec`` and ``cls_prec`` (the signal-only macro recall/precision). This is
deliberately *not* the ``cls_f1`` field, which is the mean per-class F1 over
all classes including noise (i.e. f1_total) and does not equal the harmonic
mean of the reported recall/precision.

Usage:
    python scripts/summarise_seed_runs.py logs/lamaml/<run-dir>
    python scripts/summarise_seed_runs.py logs/lamaml/<run-dir> --train --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

# Matches e.g. "cls_rec=0.5867" -> ("cls_rec", "0.5867")
_FIELD_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan)")

# Matches the "Backward: -0.0684" line written to each seed's results.txt.
_BWT_RE = re.compile(r"^Backward:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan)")


def _parse_summary_line(line: str) -> dict[str, float]:
    """Parse a "SUMMARY_TE cls_rec=.. cls_prec=.. ..." line into a dict."""
    out: dict[str, float] = {}
    for key, val in _FIELD_RE.findall(line):
        try:
            out[key] = float(val)
        except ValueError:
            out[key] = float("nan")
    return out


def _find_last_summary(log_path: str, tag: str) -> dict[str, float] | None:
    """Return the fields of the last ``SUMMARY_<tag>`` line in a log, if any."""
    prefix = "SUMMARY_" + tag
    found: dict[str, float] | None = None
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith(prefix):
                found = _parse_summary_line(line)
    return found


def _read_bwt(seed_dir: str) -> float:
    """Parse the 'Backward:' (BWT) line from a seed's results.txt, else nan."""
    path = os.path.join(seed_dir, "results.txt")
    if not os.path.isfile(path):
        return float("nan")
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _BWT_RE.match(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return float("nan")
    return float("nan")


def _signal_f1(rec: float | None, prec: float | None) -> float:
    """Harmonic mean of signal recall and precision (0 if undefined)."""
    if rec is None or prec is None:
        return float("nan")
    if math.isnan(rec) or math.isnan(prec) or (rec + prec) == 0:
        return 0.0 if (rec == 0 and prec == 0) else float("nan")
    return 2.0 * rec * prec / (rec + prec)


def _seed_sort_key(name: str):
    """Sort seed dir names numerically when possible, else lexically."""
    return (0, int(name)) if name.isdigit() else (1, name)


def discover_seed_runs(run_dir: str, tag: str) -> list[tuple[str, dict[str, float]]]:
    """Find (seed_name, summary_fields) for every seed sub-dir with a summary."""
    runs: list[tuple[str, dict[str, float]]] = []
    for name in sorted(os.listdir(run_dir), key=_seed_sort_key):
        sub = os.path.join(run_dir, name)
        if not os.path.isdir(sub):
            continue
        log_path = os.path.join(sub, "terminal.log")
        if not os.path.isfile(log_path):
            continue
        fields = _find_last_summary(log_path, tag)
        if fields is not None:
            runs.append((name, fields))
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and *sample* std (ddof=1). Std is nan for n<2."""
    clean = [v for v in values if not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(clean) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return mean, math.sqrt(var)


def summarise(run_dir: str, tag: str) -> dict:
    """Aggregate one split (TE or TR) across seeds into a structured result."""
    runs = discover_seed_runs(run_dir, tag)
    if not runs:
        raise SystemExit(
            f"No 'SUMMARY_{tag}' lines found under {run_dir!r} "
            "(expected one seed sub-dir per run, each with terminal.log)."
        )

    seeds = [name for name, _ in runs]
    # seeds = [name for name in seeds if name in ["0", "39", "55"]]
    # Per-seed derived signal F1 plus the raw fields we care about.
    per_seed_signal_f1 = [_signal_f1(f.get("cls_rec"), f.get("cls_prec")) for _, f in runs]

    # Columns to report straight from the summary line (order preserved).
    raw_keys = ["cls_rec", "cls_prec", "det", "fa", "cls_f1"]
    columns: dict[str, list[float]] = {"signal_f1": per_seed_signal_f1}
    for key in raw_keys:
        columns[key] = [f.get(key, float("nan")) for _, f in runs]

    # BWT lives in each seed's results.txt (validation recall matrix), so it is
    # only meaningful for the validation split.
    if tag == "TE":
        columns["bwt"] = [_read_bwt(os.path.join(run_dir, name)) for name, _ in runs]

    stats = {name: mean_std(vals) for name, vals in columns.items()}
    return {
        "run_dir": run_dir,
        "tag": tag,
        "seeds": seeds,
        "n": len(seeds),
        "columns": columns,
        "stats": stats,
    }


# Human-readable labels for each reported metric.
_LABELS = {
    "signal_f1": "Signal F1  (2·P·R/(P+R))",
    "cls_rec": "cls_rec  (signal recall)",
    "cls_prec": "cls_prec (signal precision)",
    "det": "det      (detection recall)",
    "fa": "fa       (false alarm)",
    "cls_f1": "cls_f1   (f1_total, reference)",
    "bwt": "bwt      (backward transfer)",
}


def _fmt_row(label: str, mean: float, std: float, per_seed: list[float]) -> str:
    per = ", ".join("nan" if math.isnan(v) else f"{v:.4f}" for v in per_seed)
    std_s = "  nan" if math.isnan(std) else f"{std:.4f}"
    return f"  {label:<30} {mean:.4f} +/- {std_s}   [{per}]"


def print_report(result: dict) -> None:
    split = "Validation" if result["tag"] == "TE" else "Training"
    print(f"Seed-sweep summary ({split}, SUMMARY_{result['tag']})")
    print(f"Dir:   {result['run_dir']}")
    print(f"Seeds: {', '.join(result['seeds'])}")
    print(f"Runs:  {result['n']}")
    print()
    order = ["signal_f1", "cls_rec", "cls_prec", "det", "fa", "cls_f1", "bwt"]
    for key in order:
        if key not in result["columns"]:
            continue
        mean, std = result["stats"][key]
        print(_fmt_row(_LABELS[key], mean, std, result["columns"][key]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Log dir containing one sub-dir per seed.")
    ap.add_argument("--train", action="store_true", help="Also report the training split (SUMMARY_TR).")
    ap.add_argument("--only-train", action="store_true", help="Report only the training split, not validation.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.run_dir):
        ap.error(f"not a directory: {args.run_dir}")

    tags = ["TR"] if args.only_train else ["TE"]
    if args.train and not args.only_train:
        tags.append("TR")

    results = [summarise(args.run_dir, tag) for tag in tags]

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        for i, res in enumerate(results):
            if i:
                print()
            print_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
