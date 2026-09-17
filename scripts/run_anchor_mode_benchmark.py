#!/usr/bin/env python
"""Benchmark the loss-form vs proximal quadratic anchor on EWC / SI / RWalk / UCL.

Each learner's anchor can be applied two ways (``--anchor_mode``): added to the
training loss and descended explicitly, or applied in closed form after the
optimiser step (see ``utils.proximal_anchor``). The two are the *same* penalty,
so the only fair comparison sweeps the penalty strength in both modes over the
same grid -- the modes do not share a useful strength range, and at a strength
low enough to be inert they are trivially identical.

The grids below are anchored on measured importance magnitudes: the proximal
blend weight is ``b_i = lr * k * Omega_i``, and a strength only does anything
once ``b`` reaches order 1 somewhere in the distribution. At the strengths the
YAML configs ship with, ``b_max`` is 1e-5 to 1e-3 for every method here, i.e.
the anchor is numerically inert and the run is naive fine-tuning; those points
are kept in the grid as the as-configured reference.

Usage:
    python scripts/run_anchor_mode_benchmark.py --stage baseline
    python scripts/run_anchor_mode_benchmark.py --stage sweep --jobs 2
    python scripts/run_anchor_mode_benchmark.py --collect-only
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from results_txt import f1_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

RUNTIME_RE = re.compile(r"^total_runtime_seconds:\s+([\d.]+)", re.MULTILINE)

# model -> (config stem, penalty-strength flag, as-configured strength, sweep grid)
#
# Sweep grids are three decades wide because the measured importance grows with
# training length: the quantiles this grid was sized from came from a 60-step
# probe, and a full task is ~2000 steps, so the strength that puts b ~ 1 moves
# down as the path integral accumulates.
METHODS: Dict[str, Dict[str, object]] = {
    "ewc": {
        "config": "ewc",
        "flag": "--lamb",
        "configured": 1.0,
        "grid": [1e3, 1e4],
    },
    "si": {
        "config": "si",
        "flag": "--si_c",
        "configured": 0.4,
        # Measured 2026-09-03 at n_epochs 1 / inner_steps 2 / p=0, seed 0: the
        # proximal peak is 1e3, a decade above the old top of this grid, and it
        # is interior (1e3 -> 0.5876, 3e3 -> 0.5520, 1e4 -> 0.5240, 3e4 ->
        # 0.5107). The two points the old grid did cover scored 0.3429 (1e1) and
        # 0.5020 (1e2), so every proximal-si number taken from it understated the
        # method by up to 0.25.
        "grid": [1e2, 1e3, 1e4],
    },
    "rwalk": {
        "config": "rwalk",
        "flag": "--lamb",
        "configured": 1.0,
        "grid": [1e2, 1e3, 1e4],
    },
    "ucl": {
        "config": "ucl",
        "flag": "--alpha",
        "configured": 10.0,
        "grid": [1e11],
    },
}


@dataclass
class Run:
    """One benchmark run: a method, an anchor mode and a penalty strength."""

    method: str
    anchor_mode: str
    strength: float
    # Extra main.py flags shared by every run in a campaign, e.g. the one-shot
    # schedule. Part of the run identity so a short and a full campaign do not
    # collide in the logs.
    extra_args: Tuple[str, ...] = ()
    suffix: str = ""

    @property
    def name(self) -> str:
        return (
            f"anchorbench{self.suffix}_{self.method}_"
            f"{self.anchor_mode}_{self.strength:g}"
        )

    def command(self) -> List[str]:
        config = METHODS[self.method]["config"]
        return [
            PYTHON,
            str(REPO_ROOT / "main.py"),
            "--config",
            "configs/base.yaml",
            "--config",
            f"configs/models/til/{config}.yaml",
            "--single-seed",
            "--no-save_checkpoints",
            "--expt_name",
            self.name,
            "--anchor_mode",
            self.anchor_mode,
            str(METHODS[self.method]["flag"]),
            repr(self.strength),
            *self.extra_args,
        ]


def build_runs(
    stage: str, extra_args: Tuple[str, ...] = (), suffix: str = ""
) -> List[Run]:
    """Assemble the run list for a stage.

    Args:
        stage: ``"baseline"`` runs each method once at its as-configured
            strength (where both modes coincide numerically, so only the loss
            form is run). ``"sweep"`` runs both modes across the grid.
            ``"all"`` is both.

    Returns:
        The runs to execute, in the order they should be launched.
    """
    runs: List[Run] = []
    if stage in ("baseline", "all"):
        for method, spec in METHODS.items():
            runs.append(
                Run(method, "loss", float(spec["configured"]), extra_args, suffix)
            )
    if stage in ("sweep", "all"):
        for method, spec in METHODS.items():
            for strength in spec["grid"]:
                for mode in ("loss", "proximal"):
                    runs.append(Run(method, mode, float(strength), extra_args, suffix))
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

    f1_values = f1_stats(text)
    return {
        "diagonal_f1": f1_values.get("Diagonal F1"),
        "final_f1": f1_values.get("Final F1"),
        "bwt": f1_values.get("Backward"),
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
    """Join every run's summary metrics into one CSV and print it."""
    rows = []
    for run in runs:
        path = find_results(run)
        row = {
            "method": run.method,
            "anchor_mode": run.anchor_mode,
            "strength": run.strength,
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
        f"{'method':8s} {'mode':9s} {'strength':>10s} "
        f"{'diag':>8s} {'final':>8s} {'bwt':>8s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(
            _format_metric(row[key]) for key in ("diagonal_f1", "final_f1", "bwt")
        )
        print(
            f"{row['method']:8s} {row['anchor_mode']:9s} "
            f"{row['strength']:10g} {cells}"
        )
    print(f"\nwrote {csv_path}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--stage", choices=["baseline", "sweep", "all"], default="all"
    )
    argument_parser.add_argument("--jobs", type=int, default=2)
    argument_parser.add_argument("--methods", type=str, default="")
    argument_parser.add_argument(
        "--one-shot",
        action="store_true",
        help=(
            "Short schedule: --n_epochs 1 --inner_steps 2, matching "
            "scripts/full_experiments.sh --one-shot. Runs land under a distinct "
            "'oneshot' experiment name so they never mix with full-schedule runs."
        ),
    )
    argument_parser.add_argument("--collect-only", action="store_true")
    argument_parser.add_argument(
        "--csv", type=str, default="scripts/logs/anchor_mode_benchmark.csv"
    )
    argument_parser.add_argument(
        "--log-root", type=str, default="scripts/logs/anchor_bench"
    )
    args = argument_parser.parse_args()

    extra_args: Tuple[str, ...] = ()
    suffix = ""
    if args.one_shot:
        extra_args = ("--n_epochs", "1", "--inner_steps", "2")
        suffix = "-oneshot"
    runs = build_runs(args.stage, extra_args, suffix)
    if args.methods:
        wanted = {name.strip() for name in args.methods.split(",") if name.strip()}
        runs = [run for run in runs if run.method in wanted]
    if not runs:
        sys.exit("no runs selected")

    if not args.collect_only:
        execute(runs, args.jobs, REPO_ROOT / args.log_root)

    csv_path = REPO_ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    collect(runs, csv_path)


if __name__ == "__main__":
    main()
