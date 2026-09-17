#!/usr/bin/env python
"""Replay as evidence injection: what should a rehearsal buffer actually store?

This project's headline empirical result is that replay is the only load-bearing
mechanism. The Dempster-Shafer reading of *why* is that replay re-injects
evidence for old classes to counteract the conflict new evidence introduces. That
reading is only worth having if it predicts something, and it predicts two things
that can be wrong:

1. **Distil ``w``, not ``z``.** A logit is ``z_k = w+_k - w-_k``, the difference
   of the two evidence channels, so it discards their common magnitude -- the
   ignorance degree of freedom. ``(8, 1)`` and ``(108, 101)`` are the same logit
   built from wildly different amounts of evidence. If evidence injection is the
   mechanism, distilling ``(w+, w-)`` should beat distilling ``z``, which is Dark
   Experience Replay.
2. **Store ``phi``, not the input.** The evidence is a function of ``phi`` at the
   readout, so ``phi`` is a sufficient statistic for it. Here that is 512 floats
   against a 1024-float input, so a matched byte budget buys 2x the exemplars.

Design
------
Every arm runs with the **anchor off** (``woe_lambda=0``) on a **256-memory**
buffer. Both choices are forced by C5: the anchor caps the result and makes
buffer size stop mattering, and at 5120 memories there is so little forgetting
left that the comparison has no dynamic range at all. 256 is where the ER ladder
shows a buffer and the anchor are level, i.e. where there is still something to
measure.

The two distillation arms are run *alone*, not on top of CE rehearsal. Adding CE
to both would dilute precisely the difference being measured; the ``ce`` arm is
the ER bar and sits in the same sweep.

The footprint claim needs **two** feature-store cells to be attributable:
``feature`` at 512 memories is the matched-*bytes* comparison against ``ce``, and
``feature`` at 256 memories is the matched-*items* control. Without the second,
a win is unattributable between "features are a better thing to store" and
"twice as many exemplars", and a loss is unattributable between "features are
worse" and "stale features are worse".

Usage:
    python scripts/run_evidence_injection_benchmark.py --jobs 2
    python scripts/run_evidence_injection_benchmark.py --collect-only
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

DIAGONAL_RE = re.compile(r"^Diagonal\s+\S+:\s+(-?[\d.]+)", re.MULTILINE)
FINAL_RE = re.compile(r"^Final\s+\S+:\s+(-?[\d.]+)", re.MULTILINE)
BACKWARD_RE = re.compile(r"^Backward:\s+(-?[\d.]+)", re.MULTILINE)
RUNTIME_RE = re.compile(r"^total_runtime_seconds:\s+([\d.]+)", re.MULTILINE)

# The one-shot schedule every WoE result in docs/woe-cl/README.md was measured on.
ONE_SHOT: Tuple[str, ...] = ("--n_epochs", "1", "--inner_steps", "2")

# Floats per stored item, for the byte-budget bookkeeping. The canonicalised
# input is 1024 floats; the penultimate feature vector is `feature_dim` = 512.
INPUT_FLOATS = 1024
FEATURE_FLOATS = 512

# Distillation strength for the `logit` / `evidence_sym` arms. No lambda in this
# project has ever transferred between penalties, and these two are on related
# but not identical scales (a logit drift is one squared term per class where an
# evidence drift is two), so both are swept over the same grid and read against
# their own best cell. 0 is not included here because with no CE term it would be
# an empty loss; the `ce` arm is the control instead.
EVIDENCE_LAMBDA_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class Arm:
    """One buffer configuration: what it stores, and what it charges."""

    name: str
    memories: int
    extra_args: Tuple[str, ...] = field(default_factory=tuple)
    # Whether this arm's distillation strength is swept.
    sweeps_lambda: bool = False

    @property
    def item_floats(self) -> int:
        return FEATURE_FLOATS if "feature" in self.extra_args else INPUT_FLOATS

    @property
    def budget_floats(self) -> int:
        return self.memories * self.item_floats


ARMS: Dict[str, Arm] = {
    # --- the bar -----------------------------------------------------------
    # Plain reservoir ER at 256 memories, anchor off. Everything else in the
    # sweep is read against this.
    "ce": Arm("ce", 256, ("--woe_replay_mode", "ce")),
    # --- claim 1: distil w, not z ------------------------------------------
    # Matched pair. Same buffer, same draw, same symmetric squared form against a
    # per-item snapshot taken at insertion; only the target differs.
    "logit": Arm("logit", 256, ("--woe_replay_mode", "logit"), sweeps_lambda=True),
    "evidence_sym": Arm(
        "evidence_sym", 256, ("--woe_replay_mode", "evidence_sym"), sweeps_lambda=True
    ),
    # --- claim 2: store phi, not the input ---------------------------------
    # Matched BYTES against `ce`: 512 features x 512 floats == 256 inputs x 1024.
    # This is the cell the footprint claim lives or dies on.
    "feature_bytes": Arm(
        "feature_bytes",
        512,
        ("--woe_replay_mode", "ce", "--woe_replay_store", "feature"),
    ),
    # Matched ITEMS against `ce`: same 256 exemplars, half the bytes. Separates
    # "features are a better thing to store" from "twice as many exemplars", and
    # on the downside separates "features are worse" from "stale features are
    # worse". Without this cell neither direction is attributable.
    "feature_items": Arm(
        "feature_items",
        256,
        ("--woe_replay_mode", "ce", "--woe_replay_store", "feature"),
    ),
    # --- the two claims composed -------------------------------------------
    # If both hold, the buffer should store phi *and* distil evidence: run at the
    # matched-bytes capacity. Only worth running once the singles have landed.
    "feature_evidence": Arm(
        "feature_evidence",
        512,
        ("--woe_replay_mode", "evidence_sym", "--woe_replay_store", "feature"),
        sweeps_lambda=True,
    ),
}


@dataclass(frozen=True)
class Run:
    """One benchmark run: an arm, a distillation strength and a seed."""

    arm: str
    evidence_lambda: float
    seed: int

    @property
    def name(self) -> str:
        return f"inj_{self.arm}_lam{self.evidence_lambda:g}_s{self.seed}"

    def command(self) -> List[str]:
        arm = ARMS[self.arm]
        return [
            PYTHON,
            str(REPO_ROOT / "main.py"),
            "--config",
            "configs/base.yaml",
            "--config",
            "configs/models/til/woe_si_injection.yaml",
            *ONE_SHOT,
            "--woe_replay_memories",
            str(arm.memories),
            "--woe_evidence_lambda",
            repr(self.evidence_lambda),
            "--single-seed",
            "--seed",
            str(self.seed),
            "--no-save_checkpoints",
            "--expt_name",
            self.name,
            *arm.extra_args,
        ]


def build_runs(arms: List[str], seeds: Tuple[int, ...]) -> List[Run]:
    """Assemble the run list, the ER bar first.

    Arms that do not sweep the distillation strength get a single cell at 1.0 --
    the flag is inert for them (nothing reads ``woe_evidence_lambda`` on a
    pure-CE arm), and pinning it keeps every run name uniform.
    """
    runs: List[Run] = []
    for seed in seeds:
        for name in arms:
            arm = ARMS[name]
            grid = EVIDENCE_LAMBDA_GRID if arm.sweeps_lambda else (1.0,)
            for evidence_lambda in grid:
                runs.append(Run(name, evidence_lambda, seed))
    return runs


def find_results(run: Run) -> Optional[Path]:
    """Locate the ``results.txt`` a run wrote, by its unique experiment name."""
    pattern = str(REPO_ROOT / "logs" / "*" / f"{run.name}-*" / "*" / "results.txt")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return Path(matches[-1]) if matches else None


def parse_results(path: Path) -> Dict[str, Optional[float]]:
    """Pull the summary metrics out of a ``results.txt``."""
    text = path.read_text(encoding="utf-8")

    def first(pattern: re.Pattern[str]) -> Optional[float]:
        match = pattern.search(text)
        return float(match.group(1)) if match else None

    return {
        "diagonal_f1": first(DIAGONAL_RE),
        "final_f1": first(FINAL_RE),
        "bwt": first(BACKWARD_RE),
        "runtime_s": first(RUNTIME_RE),
    }


def execute(runs: List[Run], jobs: int, log_root: Path) -> None:
    """Run the list with at most ``jobs`` concurrent subprocesses."""
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(runs)
    active: List[tuple[Run, subprocess.Popen, object]] = []
    started = time.time()

    while pending or active:
        while pending and len(active) < jobs:
            run = pending.pop(0)
            handle = (log_root / f"{run.name}.log").open("w", encoding="utf-8")
            print(f"[launch] {run.name}", flush=True)
            environment = dict(os.environ, PYTHONUNBUFFERED="1")
            process = subprocess.Popen(
                run.command(),
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            active.append((run, process, handle))

        time.sleep(5)
        for entry in list(active):
            run, process, handle = entry
            if process.poll() is None:
                continue
            handle.close()
            active.remove(entry)
            elapsed = time.time() - started
            status = "ok" if process.returncode == 0 else f"FAIL({process.returncode})"
            print(
                f"[done]   {run.name} {status} "
                f"({len(pending)} queued, {elapsed / 60:.1f} min elapsed)",
                flush=True,
            )


def _format_metric(value: Optional[float]) -> str:
    """Render one metric cell, blanking metrics a failed run never produced."""
    return f"{value:8.4f}" if value is not None else f"{'-':>8s}"


def collect(runs: List[Run], csv_path: Path) -> None:
    """Join every run's summary metrics into one CSV and print it.

    ``budget_kf`` (thousands of floats) is printed alongside so the matched-bytes
    comparison is legible in the table rather than something the reader has to
    reconstruct: ``ce`` and ``feature_bytes`` must show the same number.
    """
    rows = []
    for run in runs:
        arm = ARMS[run.arm]
        path = find_results(run)
        row: Dict[str, object] = {
            "arm": run.arm,
            "memories": arm.memories,
            "budget_kfloats": arm.budget_floats / 1000.0,
            "evidence_lambda": run.evidence_lambda,
            "seed": run.seed,
            "diagonal_f1": None,
            "final_f1": None,
            "bwt": None,
            "runtime_s": None,
            "results_path": str(path) if path else "",
        }
        if path is not None:
            row.update(parse_results(path))
        rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'arm':17s} {'mem':>5s} {'budget_kf':>10s} {'lambda':>8s} {'seed':>5s} "
        f"{'diag':>8s} {'final':>8s} {'bwt':>8s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(
            _format_metric(row[key]) for key in ("diagonal_f1", "final_f1", "bwt")
        )
        print(
            f"{row['arm']:17s} {row['memories']:5d} {row['budget_kfloats']:10.1f} "
            f"{row['evidence_lambda']:8g} {row['seed']:5d} {cells}"
        )
    print(f"\nwrote {csv_path}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--arms",
        type=str,
        default="ce,logit,evidence_sym,feature_bytes,feature_items",
        help="Comma-separated subset of "
        + ",".join(ARMS)
        + ". The default runs both claims and their controls; feature_evidence "
        "composes the two and is worth running only once the singles land.",
    )
    argument_parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seeds. The project requires n=3 (0,39,55) before "
        "any result is believed; sweep at seed 0 first, then confirm.",
    )
    argument_parser.add_argument("--jobs", type=int, default=2)
    argument_parser.add_argument("--collect-only", action="store_true")
    argument_parser.add_argument(
        "--csv", type=str, default="scripts/logs/evidence_injection_benchmark.csv"
    )
    argument_parser.add_argument(
        "--log-root", type=str, default="scripts/logs/evidence_injection"
    )
    args = argument_parser.parse_args()

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        sys.exit(f"unknown arm(s): {', '.join(unknown)}")
    seeds = tuple(int(piece) for piece in args.seeds.split(",") if piece.strip())

    runs = build_runs(arms, seeds)
    if not runs:
        sys.exit("no runs selected")

    if not args.collect_only:
        execute(runs, args.jobs, REPO_ROOT / args.log_root)

    csv_path = REPO_ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    collect(runs, csv_path)


if __name__ == "__main__":
    main()
