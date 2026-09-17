"""RWalk: is the eps degeneracy real, and does the Fisher term do anything?

Reads scripts/run_rwalk_fisher_ablation.sh. The reference is the anchor-only
cell of the LwF 2x2 (eps=0.01, lamb=1000, alpha=0.9), same config and seeds.

Two claims under test:

1. The eps arms (lamb compensated to hold lamb/eps at 1e5) should be FLAT.
   probe_rwalk_riemannian.py puts RWalk's median 0.5*F*delta^2 at 1e-14 of eps,
   so lowering eps by four or eight orders should change nothing at all -- the
   denominator is eps either way. Movement here would falsify that reading.
2. --alpha 0.0 pins the Fisher EMA at its zero init, so importance becomes `s`
   alone. F supplies 0.1-0.9% of the penalty magnitude and its negative-value
   clamp never fires, so it looks vestigial; this measures whether it is.

Usage:
    la-maml_env/bin/python scripts/analyse_rwalk_fisher_ablation.py
"""

from __future__ import annotations

import glob
import itertools
import os
import re
import statistics as st

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
METRICS = ["Final F1", "Diagonal F1", "Backward"]

CELLS = {
    "reference (eps 1e-2, alpha 0.9)": (
        "logs/rwalk_lwf/rwalk_anchoronly_noamp_se_til-*"
    ),
    "eps 1e-5 (lamb 1)": "logs/rwalk_lwf/rwalk_eps1em5_noamp_se_til-*",
    "eps 1e-9 (lamb 1e-4)": "logs/rwalk_lwf/rwalk_eps1em9_noamp_se_til-*",
    "alpha 0 (F == 0)": "logs/rwalk_lwf/rwalk_alpha0_noamp_se_til-*",
}
REFERENCE = "reference (eps 1e-2, alpha 0.9)"


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in open(path):
        m = re.match(r"(Diagonal F1|Final F1|Backward):\s+(-?[\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2)) * 100
    return out


def collect(pattern: str) -> dict[int, dict[str, float]]:
    seeds: dict[int, dict[str, float]] = {}
    for base in sorted(glob.glob(os.path.join(REPO, pattern))):
        for dp, _d, files in os.walk(base):
            leaf = os.path.basename(dp)
            if "results.txt" in files and leaf.isdigit():
                rec = parse(os.path.join(dp, "results.txt"))
                if "Final F1" in rec:
                    seeds.setdefault(int(leaf), rec)
    return seeds


def perm_p(d: list[float]) -> float:
    """Exact two-sided sign-flip test; floor is 2^-(n-1), i.e. 0.25 at n=3."""
    n = len(d)
    if n == 0 or n > 20:
        return float("nan")
    obs = abs(sum(d) / n)
    hits = sum(
        1
        for s in itertools.product((1, -1), repeat=n)
        if abs(sum(a * b for a, b in zip(s, d)) / n) >= obs - 1e-12
    )
    return hits / 2**n


def fmt(vals: list[float]) -> str:
    if not vals:
        return "     --    "
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):6.2f}±{sd:4.2f}"


def paired(a: dict, b: dict, key: str) -> str:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return f"(n={len(common)})"
    d = [a[s][key] - b[s][key] for s in common]
    half = 1.96 * st.stdev(d) / len(d) ** 0.5
    return f"{st.mean(d):+5.2f} ±{half:4.2f} p={perm_p(d):.3f} n={len(d)}"


def main() -> None:
    data = {label: collect(pattern) for label, pattern in CELLS.items()}
    print("RWalk eps degeneracy check + Fisher ablation.")
    print("Anchor only, single-epoch TIL, inner_steps 2, no AMP.")
    print("Percent, mean±sample sd.\n")

    width = max(len(c) for c in CELLS) + 2
    for key in METRICS:
        print(f"--- {key} ---")
        for label in CELLS:
            pool = data[label]
            vals = [pool[s][key] for s in sorted(pool)]
            print(f"  {label:{width}s}{fmt(vals):>14s}   (n={len(vals)})")
        print()

    print(f"--- paired vs {REFERENCE} ---")
    for label in CELLS:
        if label == REFERENCE:
            continue
        print(f"  {label}")
        for key in METRICS:
            print(f"      {key:12s}{paired(data[label], data[REFERENCE], key)}")
    print(
        "\n  The two eps arms should be flat on all three metrics: at 1e-14 of\n"
        "  eps the Riemannian numerator cannot reach the denominator, so\n"
        "  compensated eps is a no-op. Movement there falsifies the probe.\n"
        "  alpha 0 is the real ablation: flat means F is vestigial and RWalk\n"
        "  reduces to an unnormalised path integral, i.e. to SI's estimator.\n"
        "  p floors at 0.25 at n=3; read effect sizes, not significance."
    )


if __name__ == "__main__":
    main()
