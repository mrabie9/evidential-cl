"""PR-3 report: per-lambda paired difference between the two mu arms.

The deciding statistic is the **matched-lambda** difference at ``lambda = 2.4e5``
(Amendment 3). Peak-to-peak, each arm's peak location, and the differences at the
other lambdas are descriptors, reported so the reshape evidence is visible rather
than taken on trust.

Why matched-lambda: a difference cancels a shared *additive* perturbation, but
the pre-pass may reshape the lambda->retention curve rather than offset it. A
reshape cancels only where both arms sit at the same lambda, which peak-to-peak
does not guarantee.

Usage:
    python scripts/mu_frozen_report.py
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECIDING_LAMBDA = "240000.0"
SIGMA = 0.0031  # on-peak, pooled over i2@2.4e5 and ce@75, both n=3, this host.
NO_PREPASS_BAR = 0.5008  # recorded ema no-pre-pass n=3 mean at 2.4e5.

RUN_RE = re.compile(r"mufz_(ema|frozen_pretask)_lam([0-9.e+]+)_s(\d+)-")


def _metric(path: str, key: str) -> float | None:
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith(key):
                    return float(line.split(":")[1].strip())
    except OSError:
        return None
    return None


def collect() -> dict:
    out: dict = defaultdict(dict)
    for results in glob.glob(f"{REPO}/logs/woe_si_lc/mufz_*/*/results.txt"):
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(results)))
        match = RUN_RE.match(run_dir)
        if not match:
            continue
        mode, lam, seed = match.group(1), match.group(2), int(match.group(3))
        final = _metric(results, "Final F1:")
        diag = _metric(results, "Diagonal F1:")
        bwt = _metric(results, "Backward:")
        if final is None:
            continue
        out[(lam, seed)][mode] = (diag, final, bwt)
    return out


def _mean_sd(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var**0.5


def main() -> int:
    data = collect()
    if not data:
        print("no PR-3 runs found yet")
        return 0

    lambdas = sorted({lam for lam, _ in data}, key=float)
    print("Per-lambda, per-seed (diagonal / final / BWT)\n")
    header = f"{'lambda':>10} {'seed':>5} {'ema final':>11} {'frozen final':>13} {'paired D':>10}"
    print(header)
    print("-" * len(header))
    paired_by_lambda: dict[str, list[float]] = defaultdict(list)
    for lam in lambdas:
        for seed in sorted(s for lm, s in data if lm == lam):
            arms = data[(lam, seed)]
            ema = arms.get("ema")
            frozen = arms.get("frozen_pretask")
            e = f"{ema[1]:.4f}" if ema else "-"
            f = f"{frozen[1]:.4f}" if frozen else "-"
            if ema and frozen:
                d = frozen[1] - ema[1]
                paired_by_lambda[lam].append(d)
                ds = f"{d:+.4f}"
            else:
                ds = "-"
            print(f"{lam:>10} {seed:>5} {e:>11} {f:>13} {ds:>10}")

    print("\nPaired difference by lambda (frozen - ema), n and mean")
    for lam in lambdas:
        vals = paired_by_lambda.get(lam, [])
        if not vals:
            continue
        mean, sd = _mean_sd(vals)
        tag = "  <-- DECIDING (Amendment 3)" if lam == DECIDING_LAMBDA else ""
        sd_s = f" +/- {sd:.4f}" if len(vals) > 1 else ""
        print(
            f"  lambda={lam:>10}  n={len(vals)}  D'={mean:+.4f}{sd_s}"
            f"  ({abs(mean) / SIGMA:.1f} sigma){tag}"
        )

    print("\nBands (Amendment 2, by sign):")
    print(f"  |D'| <= {2 * SIGMA:.4f}          tie persists; objection closed")
    print(f"  {2 * SIGMA:.4f} < |D'| <= {4 * SIGMA:.4f}  inconclusive; extend to n=5")
    print(
        f"  D' >  +{4 * SIGMA:.4f}         frozen better; gauge artefact; spine re-examined"
    )
    print(
        f"  D' <  -{4 * SIGMA:.4f}         frozen worse; objection closed weakly, mu unsettled"
    )

    print("\nAmendment 1 clause 3 -- is the pre-pass behaving as common-mode?")
    ema_at_deciding = [
        arms["ema"][1]
        for (lam, _), arms in data.items()
        if lam == DECIDING_LAMBDA and "ema" in arms
    ]
    if ema_at_deciding:
        mean, sd = _mean_sd(ema_at_deciding)
        gap = mean - NO_PREPASS_BAR
        verdict = (
            "OK" if abs(gap) <= SIGMA else "FAILS -- re-examine D' before reading it"
        )
        sd_s = f" +/- {sd:.4f}" if len(ema_at_deciding) > 1 else ""
        print(
            f"  ema+prepass @ {DECIDING_LAMBDA}: n={len(ema_at_deciding)} "
            f"mean={mean:.4f}{sd_s}"
        )
        print(
            f"  vs recorded no-pre-pass {NO_PREPASS_BAR}: {gap:+.4f} "
            f"({abs(gap) / SIGMA:.2f} sigma)  {verdict}"
        )

    print("\nDescriptors (not deciding): each arm's peak")
    for mode in ("ema", "frozen_pretask"):
        cells = [
            (lam, arms[mode][1])
            for (lam, seed), arms in data.items()
            if seed == 0 and mode in arms
        ]
        if cells:
            best = max(cells, key=lambda c: c[1])
            print(
                f"  {mode:>15}: peak at lambda={best[0]} final={best[1]:.4f} (seed 0)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
