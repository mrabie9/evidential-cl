#!/usr/bin/env python
"""Sweep cautious (max) vs Dempster (sum) accumulation of importance.

Every importance-based continual learner in the literature -- SI, EWC, online
EWC, RWalk, and every arm of this project -- combines per-task importance across
tasks by **summing** it. Under the Dempster-Shafer reading that is Dempster's
rule of combination, which is licensed only for *distinct* bodies of evidence.
Sequential tasks are the textbook counter-example: they share a backbone and each
is initialised from the previous solution. The rule for non-distinct evidence is
Denoeux's **cautious** rule, whose canonical weight functions combine by minimum
-- an elementwise **maximum** on the weight-of-evidence scale that ``Omega``
lives on (see ``model.woe_si._OMEGA_ACCUMS`` for the derivation).

The practical stake is saturation. Summed importance grows without bound for a
parameter that many tasks find useful, until it is frozen outright; online EWC
and SI both patch this with a hand-chosen decay factor. A max is idempotent and
bounded by the single most demanding task, so if it wins, the cautious rule
*derives* a fix the literature currently applies as a hyper-parameter -- which is
the most defensible kind of contribution, and one that survives B6 (the finding
that the DS content of the tracked *scalar* does no distinguishing work). This
sweep is about the combination rule, not the scalar.

Host and bar
------------
Run on the exact configuration A6 was measured on -- ``woe_si_lc`` config, one
shot, proximal anchor, ``woe_omega_transform=abs``, ``lr 0.003`` -- so the
``sum`` control arm reproduces the recorded 0.5008 +/- 0.0031 and the ``max``
arms are read against it directly.

Sizing lambda
-------------
``max`` cannot change *which* parameters are anchored (a max of non-negatives is
non-zero wherever any term is), only their magnitudes, so A6's "size the grid by
count, not mass" does not apply here -- count is fixed by construction and the
mass ratio is the whole story. That ratio is between 1x and n_tasks; the grid
spans 1x to 30x the ``abs`` peak of 2.4e5 to bracket it either way. Every run
exports ``WOE_LC_DEBUG=1``, so each consolidation prints ``total_omega`` against
the pre-combination ``task_omega`` and the sweep measures its own ratio rather
than trusting this estimate.

Usage:
    python scripts/run_cautious_rule_benchmark.py --jobs 2
    python scripts/run_cautious_rule_benchmark.py --collect-only
    python scripts/run_cautious_rule_benchmark.py --omega-trace
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
# `[LC] consolidated task=3 total_omega=1.2e+05 task_omega=4.4e+04 accum=max ...`
OMEGA_RE = re.compile(
    r"\[LC\] consolidated task=(\d+) total_omega=(\S+) task_omega=(\S+) "
    r"accum=(\S+) nonzero_frac=(\S+)"
)

# The one-shot schedule every WoE result in docs/woe-cl/README.md was measured
# on. Kept here rather than on the command line so no arm can be run on a
# different schedule to the bar it is compared against.
ONE_SHOT: Tuple[str, ...] = ("--n_epochs", "1", "--inner_steps", "2")

# A6's peak for `abs` under summed accumulation. The `sum` control runs here and
# nowhere else; the `max` arms sweep upward from it because dropping the
# saturation can only lower total Omega.
ABS_PEAK = 240000.0
LAMBDA_GRID: Tuple[float, ...] = (
    ABS_PEAK,
    3.0 * ABS_PEAK,
    1e1 * ABS_PEAK,
    3e1 * ABS_PEAK,
)


@dataclass(frozen=True)
class Arm:
    """One accumulation rule, and the lambda grid it is swept over."""

    name: str
    accum: str
    lambdas: Tuple[float, ...]
    extra_args: Tuple[str, ...] = field(default_factory=tuple)


# Shared by every arm: the A6 configuration, so `sum` at 2.4e5 *is* the recorded
# 0.5008 cell and nothing else in the comparison moves.
A6_ARGS: Tuple[str, ...] = (
    "--lr",
    "0.003",
    "--woe_omega_transform",
    "abs",
    "--woe_anchor_mode",
    "proximal",
)

ARMS: Dict[str, Arm] = {
    # Dempster's rule at A6's peak: the control this whole sweep is read against.
    # One lambda only -- it is a bar, not a sweep, and it is already a 3-seed
    # recorded number (0.5008 +/- 0.0031).
    "sum": Arm("sum", "sum", (ABS_PEAK,), A6_ARGS),
    # The cautious rule. Swept upward because removing saturation lowers Omega.
    "max": Arm("max", "max", LAMBDA_GRID, A6_ARGS),
    # --- the control that decides what the result is *about* -----------------
    # B6 found the DS content of the tracked scalar does no distinguishing work:
    # swapping I_2 for the plain task loss is indistinguishable at n=3. If the
    # cautious rule is a fact about *combination* rather than about evidence
    # theory, it should transfer to the SI scalar unchanged -- and that is the
    # stronger claim, because it makes the result a statement about every
    # importance-based method rather than about this one. Run only if `max` wins.
    "max_ce": Arm(
        "max_ce",
        "max",
        LAMBDA_GRID,
        A6_ARGS + ("--woe_importance_scalar", "ce"),
    ),
    "sum_ce": Arm(
        "sum_ce",
        "sum",
        (ABS_PEAK,),
        A6_ARGS + ("--woe_importance_scalar", "ce"),
    ),
}


@dataclass(frozen=True)
class Run:
    """One benchmark run: an accumulation rule, an anchor strength and a seed."""

    arm: str
    woe_lambda: float
    seed: int

    @property
    def name(self) -> str:
        return f"caut_{self.arm}_lam{self.woe_lambda:g}_s{self.seed}"

    def command(self) -> List[str]:
        arm = ARMS[self.arm]
        return [
            PYTHON,
            str(REPO_ROOT / "main.py"),
            "--config",
            "configs/base.yaml",
            "--config",
            "configs/models/til/woe_si_lc.yaml",
            *ONE_SHOT,
            "--woe_omega_accum",
            arm.accum,
            "--woe_lambda",
            repr(self.woe_lambda),
            "--single-seed",
            "--seed",
            str(self.seed),
            "--no-save_checkpoints",
            "--expt_name",
            self.name,
            *arm.extra_args,
        ]


def build_runs(
    arms: List[str],
    seeds: Tuple[int, ...],
    lambdas: Optional[Tuple[float, ...]] = None,
) -> List[Run]:
    """Assemble the run list, controls first.

    Ordered so every arm's cheapest informative cell launches early: the ``sum``
    control is the bar everything is read against, and a broken code path then
    shows up in the first few minutes rather than an hour into the campaign.

    Args:
        arms: Arm names to run.
        seeds: Seeds to run each cell at.
        lambdas: Override grid applied to *every* selected arm. Used to add a
            re-centred cell once the sweep's own Omega trace has measured the
            sum/max mass ratio, which is the only honest way to place it.
    """
    runs: List[Run] = []
    for seed in seeds:
        for name in arms:
            for woe_lambda in lambdas or ARMS[name].lambdas:
                runs.append(Run(name, woe_lambda, seed))
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
    """Run the list with at most ``jobs`` concurrent subprocesses.

    ``WOE_LC_DEBUG=1`` is exported for every run so each consolidation traces its
    ``Omega`` mass. That trace is the point of the sweep as much as the metrics:
    it measures the sum/max ratio rather than leaving it to the grid to guess.
    """
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(runs)
    active: List[tuple[Run, subprocess.Popen, object]] = []
    started = time.time()

    while pending or active:
        while pending and len(active) < jobs:
            run = pending.pop(0)
            handle = (log_root / f"{run.name}.log").open("w", encoding="utf-8")
            print(f"[launch] {run.name}", flush=True)
            environment = dict(os.environ, PYTHONUNBUFFERED="1", WOE_LC_DEBUG="1")
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
        row: Dict[str, object] = {
            "arm": run.arm,
            "accum": ARMS[run.arm].accum,
            "woe_lambda": run.woe_lambda,
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
        f"{'arm':8s} {'lambda':>10s} {'seed':>5s} "
        f"{'diag':>8s} {'final':>8s} {'bwt':>8s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(
            _format_metric(row[key]) for key in ("diagonal_f1", "final_f1", "bwt")
        )
        print(f"{row['arm']:8s} {row['woe_lambda']:10g} {row['seed']:5d} {cells}")
    print(f"\nwrote {csv_path}")


def omega_trace(runs: List[Run], log_root: Path) -> None:
    """Report the per-run ``Omega`` mass series the debug trace recorded.

    ``saturation`` is the cumulative total divided by the largest single task's
    contribution. Under ``sum`` it is how many tasks' worth of importance the
    anchor has piled onto the same parameters, i.e. the quantity online-EWC's
    decay factor exists to bound.

    Under ``max`` it is **not** 1.0, and the reason is worth stating because the
    obvious guess is wrong: these are sums *over parameters* of an elementwise
    maximum, and different parameters attain their maximum on different tasks. So
    the total only collapses to a single task's when one task dominates every
    parameter at once. What the ``max`` figure actually measures is how
    *distributed* the per-parameter maxima are across the sequence -- 1.0 would
    mean one task owns the whole network, and the measured 2.0 means the peak
    importance is spread over roughly two tasks' worth of parameters.

    The ratio between the two arms' final ``total_omega`` is what ``woe_lambda``
    has to be re-centred by.
    """
    header = (
        f"{'run':34s} {'accum':>6s} {'tasks':>6s} {'final_omega':>13s} "
        f"{'max_task':>13s} {'saturation':>11s} {'nonzero':>8s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for run in runs:
        path = log_root / f"{run.name}.log"
        if not path.exists():
            continue
        entries = OMEGA_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
        if not entries:
            print(f"{run.name:34s} {'-':>6s} {'no trace':>6s}")
            continue
        totals = [float(entry[1]) for entry in entries]
        per_task = [float(entry[2]) for entry in entries]
        accum = entries[-1][3]
        nonzero = float(entries[-1][4])
        largest = max(per_task) if per_task else 0.0
        saturation = totals[-1] / largest if largest > 0 else float("nan")
        print(
            f"{run.name:34s} {accum:>6s} {len(entries):6d} {totals[-1]:13.4e} "
            f"{largest:13.4e} {saturation:11.3f} {nonzero:8.4f}"
        )


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--arms",
        type=str,
        default="sum,max",
        help="Comma-separated subset of "
        + ",".join(ARMS)
        + ". Default runs the headline comparison only; the *_ce arms are the "
        "B6 transfer control and are worth running only if `max` wins.",
    )
    argument_parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seeds. The project requires n=3 (0,39,55) before "
        "any result is believed; sweep at seed 0 first, then confirm.",
    )
    argument_parser.add_argument(
        "--lambdas",
        type=str,
        default="",
        help="Comma-separated woe_lambda override applied to every selected arm "
        "(default: each arm's own grid). Use it to add a re-centred cell once "
        "--omega-trace has measured the sum/max mass ratio.",
    )
    argument_parser.add_argument("--jobs", type=int, default=2)
    argument_parser.add_argument("--collect-only", action="store_true")
    argument_parser.add_argument(
        "--omega-trace",
        action="store_true",
        help="Also report the Omega mass / saturation series from the run logs.",
    )
    argument_parser.add_argument(
        "--csv", type=str, default="scripts/logs/cautious_rule_benchmark.csv"
    )
    argument_parser.add_argument(
        "--log-root", type=str, default="scripts/logs/cautious"
    )
    args = argument_parser.parse_args()

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        sys.exit(f"unknown arm(s): {', '.join(unknown)}")
    seeds = tuple(int(piece) for piece in args.seeds.split(",") if piece.strip())
    lambdas = (
        tuple(float(piece) for piece in args.lambdas.split(",") if piece.strip())
        or None
    )

    runs = build_runs(arms, seeds, lambdas)
    if not runs:
        sys.exit("no runs selected")

    log_root = REPO_ROOT / args.log_root
    if not args.collect_only:
        execute(runs, args.jobs, log_root)

    csv_path = REPO_ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    collect(runs, csv_path)
    if args.omega_trace:
        omega_trace(runs, log_root)


if __name__ == "__main__":
    main()
