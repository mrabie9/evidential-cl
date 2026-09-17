"""AMP vs no-AMP for the Ring-ER grid rows G1/E1 (static ring) and E2 (dynamic ring).

bf16 AMP is not method-neutral in this repo: it costs GEM +1.43 F1 and Res-ER +1.69
diagonal (see memory cmaml-edge-is-clipping-amp / docs/cmaml_advantage_is_numerical.md).
Both Ring-ER rows were originally measured under AMP, so they need an AMP-free reading
before they can sit in a table beside the no-AMP baselines.

G1 and E1 are the SAME configuration (er_ring, lr 0.01, inner_steps 2, static ring) and
were bit-identical on all 6 shared seeds, so one no-AMP run serves both grid rows.

Usage:
    la-maml_env/bin/python scripts/analyse_noamp_ering.py
"""

from __future__ import annotations

import glob
import itertools
import os
import re
import statistics as st

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
METRICS = ["Final F1", "Diagonal F1", "Backward"]

# label -> (amp pool globs, no-amp pool globs). The no-AMP runs now back grid rows E1/E2
# and read from the curated tree; the bf16 originals they superseded were returned to
# logs/er_ring/ by scripts/organise_ablations.py, so they are listed run by run.
PAIRS = {
    "G1/E1 static ring": (
        [
            "logs/er_ring/er_ring_lr10_is2-2026-07-03_*",
            "logs/er_ring/er_ring_lr01_is2_s7-13-21_se_til-*",
            "logs/er_ring/er_ring_lr01_is2_topup-*",
        ],
        ["logs/ablations/til/res-er/E1"],
    ),
    "E2 dynamic ring": (
        [
            "logs/er_ring/er_ring_dynring_bm_lr01_is2_se_til-*",
            "logs/er_ring/er_ring_dynring_s7-13-21-*",
        ],
        ["logs/ablations/til/res-er/E2"],
    ),
}


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in open(path):
        m = re.match(r"(Diagonal F1|Final F1|Backward):\s+(-?[\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2)) * 100
    return out


def collect(patterns: str | list[str]) -> dict[int, dict[str, float]]:
    if isinstance(patterns, str):
        patterns = [patterns]
    seeds: dict[int, dict[str, float]] = {}
    for base in sorted(b for pat in patterns for b in glob.glob(os.path.join(REPO, pat))):
        for dp, _d, files in os.walk(base):
            leaf = os.path.basename(dp)
            if "results.txt" in files and leaf.isdigit():
                rec = parse(os.path.join(dp, "results.txt"))
                if "Final F1" in rec:
                    seeds.setdefault(int(leaf), rec)
    return seeds


def perm_p(d: list[float]) -> float:
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


def fmt(seeds: dict, key: str) -> str:
    vals = [seeds[s][key] for s in sorted(seeds)]
    if not vals:
        return "     --     "
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):6.2f}±{sd:4.2f}({len(vals)})"


def paired(a: dict, b: dict, key: str) -> str:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return f"(n={len(common)}, need >=2)"
    d = [a[s][key] - b[s][key] for s in common]
    half = 1.96 * st.stdev(d) / len(d) ** 0.5
    wins = sum(1 for v in d if v > 0)
    return f"{st.mean(d):+5.2f} ±{half:4.2f} p={perm_p(d):.3f} {wins}/{len(d)}"


def main() -> None:
    data = {k: (collect(v[0]), collect(v[1])) for k, v in PAIRS.items()}
    print("Ring-ER grid rows, single-epoch TIL, lr 0.01, inner_steps 2.")
    print("Percent, mean±sample sd (n).\n")
    for key in METRICS:
        print(f"--- {key} ---")
        print(f"{'row':20s} {'with AMP':>15s} {'no AMP':>15s}   AMP cost (no-AMP minus AMP)")
        for lab, (amp, noamp) in data.items():
            print(
                f"{lab:20s} {fmt(amp, key):>15s} {fmt(noamp, key):>15s}   "
                f"{paired(noamp, amp, key)}"
            )
        print()

    print("--- dynamic vs static ring, within precision (Final F1) ---")
    st_amp, st_no = data["G1/E1 static ring"]
    dy_amp, dy_no = data["E2 dynamic ring"]
    print(f"  E2 minus G1/E1  with AMP : {paired(dy_amp, st_amp, 'Final F1')}")
    print(f"  E2 minus G1/E1  no AMP   : {paired(dy_no, st_no, 'Final F1')}")
    print("\n  p is the exact two-sided sign-flip test (floor 0.031 at n=6).")


if __name__ == "__main__":
    main()
