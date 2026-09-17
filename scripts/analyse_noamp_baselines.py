"""AMP vs no-AMP for the TIL baselines: Res-ER (E0), GEM (G0), C-MAML (M0).

bf16 AMP is not method-neutral in this repo -- it costs Res-ER +1.73 diagonal and
C-MAML nothing (docs/cmaml_advantage_is_numerical.md). This pools each baseline's
AMP and no-AMP pools, reports the paired AMP cost per method, and re-derives the
cross-method gaps under both precisions.

Usage:
    la-maml_env/bin/python scripts/analyse_noamp_baselines.py
"""

from __future__ import annotations

import glob
import itertools
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
METRICS = ["Final F1", "Diagonal F1", "Backward"]

# label -> (amp pool glob, no-amp pool glob). Whichever pool backs the published grid row
# is read from the curated tree (E0/G0 are fp32 there, M0 is bf16); its counterpart stays
# in logs/ under the config stem it was logged to.
PAIRS = {
    "E0 Res-ER": (
        "logs/eralg4/e0_resER_joint_splitfix_se_til-*",
        "logs/ablations/til/res-er/E0",
    ),
    "G0 GEM": (
        "logs/gem/gem_full_lr01abl-*",
        "logs/ablations/til/gem/G0/lr0.01",
    ),
    "M0 C-MAML": (
        "logs/ablations/til/cmaml/M0",
        "logs/cmaml/m0_noamp_joint_se_til-*",
    ),
}


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in f1_stats(open(path).read()).items():
        if key in ('Diagonal F1', 'Final F1', 'Backward'):
            out[key] = value * 100
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
    data = {k: (collect(v[0]), collect(v[1])) for k, v in PAIRS.items()}
    print("TIL baselines, single-epoch, inner_steps 2. Percent, mean±sample sd.\n")
    for key in METRICS:
        print(f"--- {key} ---")
        print(
            f"{'method':11s} {'with AMP':>13s} {'no AMP':>13s}   AMP cost (no-AMP minus AMP)"
        )
        for lab, (amp, noamp) in data.items():
            print(
                f"{lab:11s} {fmt([amp[s][key] for s in sorted(amp)]):>13s} "
                f"{fmt([noamp[s][key] for s in sorted(noamp)]):>13s}   "
                f"{paired(noamp, amp, key)}"
            )
        print()

    print("--- cross-method gaps, Final F1 ---")
    for other in ["E0 Res-ER", "G0 GEM"]:
        m_amp, m_no = data["M0 C-MAML"]
        o_amp, o_no = data[other]
        print(f"  M0 minus {other:11s} with AMP : {paired(m_amp, o_amp, 'Final F1')}")
        print(f"  M0 minus {other:11s} no AMP   : {paired(m_no, o_no, 'Final F1')}")
    print("\n  p is the exact two-sided sign-flip test (floor 0.031 at n=6).")


if __name__ == "__main__":
    main()
