"""Gate B1 instrument check: reproduce Amendment 2's running-vs-batch table.

Nothing in Gate B1 is interpretable unless the harness reproduces the published
ck0/ck3 numbers from the same checkpoints. Writes a JSON cache the arms reuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_b1_lib import SEEDS, Harness  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "logs/eucr/gate_b1"

# Amendment 2, 5-seed means over b0_extra.
PUBLISHED = {
    ("ck0", "running"): 0.776,
    ("ck0", "batch"): 0.844,
    ("ck3", "running"): 0.107,
    ("ck3", "batch"): 0.672,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for seed in SEEDS:
        h = Harness(seed)
        rows[seed] = {
            f"ck{ck}_{reg}": h.f1_task0(ck, reg)
            for ck in (0, 1, 2, 3)
            for reg in ("running", "batch")
        }
        rows[seed]["gap_ck3"] = rows[seed]["ck3_batch"] - rows[seed]["ck3_running"]
        rows[seed]["denominator"] = rows[seed]["ck0_batch"] - rows[seed]["ck3_batch"]
        print(f"seed {seed}: {json.dumps(rows[seed])}", flush=True)
        del h

    (OUT / "repro.json").write_text(json.dumps(rows, indent=2))

    print("\n=== Gate B1 instrument check ===")
    print(f"{'cell':16s} {'mine':>8s} {'published':>10s} {'delta':>8s}")
    for ck in ("ck0", "ck1", "ck2", "ck3"):
        for reg in ("running", "batch"):
            mine = sum(rows[s][f"{ck}_{reg}"] for s in SEEDS) / len(SEEDS)
            pub = PUBLISHED.get((ck, reg))
            if pub is None:
                print(f"{ck+'/'+reg:16s} {mine:8.4f} {'--':>10s} {'--':>8s}")
            else:
                print(f"{ck+'/'+reg:16s} {mine:8.4f} {pub:10.3f} {mine-pub:+8.4f}")
    gaps = [rows[s]["gap_ck3"] for s in SEEDS]
    print(f"\nck3 running-vs-batch gap: {sum(gaps)/len(gaps):.4f} (published 0.565)")
    print("per seed:", " ".join(f"{s}={g:.4f}" for s, g in zip(SEEDS, gaps)))
    dens = [rows[s]["denominator"] for s in SEEDS]
    print(
        f"genuine forgetting batch@ck0-batch@ck3: {sum(dens)/len(dens):.4f} "
        "(published 0.172)"
    )


if __name__ == "__main__":
    main()
