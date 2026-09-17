#!/usr/bin/env python
"""Benchmark the evidential objective (``--woe_evidential_mode``) on ``woe_si``.

Every mechanism in ``docs/woe-cl/README.md`` sections A-D constrains the model
relative to its own past. Section E charges the current task instead, and E1/E2
did so by *removing* commitment (the Least-Commitment objective), which measured
monotonically harmful. This is the asymmetric version: replace cross-entropy with
a two-sided log loss on a bounded evidential score, so evidence is asked to
support the label and not the other classes.

The novel content sits on one axis. With ``d_k = w+_k - w-_k = z'_k`` and
``s_k = w+_k + w-_k``, cross-entropy constrains only ``d``; ``s`` -- the total
contribution magnitude ``sum_j |beta_kj phi'_j + beta_0k/J|`` -- is invisible to
it. In ``balance`` mode the objective asks for ``|d_k| -> s_k``, i.e. every
contribution to a class sharing one sign, which is a *selectivity* constraint on
the readout rows rather than a constraint against the past. Two features of the
implementation exist to keep that test honest, both measured rather than assumed:

* ``1/(K-1)`` on the non-target term plus inverse-frequency sample weighting.
  Each row is the target for ``p_k`` of a batch and a non-target for the rest, and
  ``s_k`` enters ``w+_k`` positively either way, so the unweighted form has a net
  shrinkage pressure of ``-(1 - 2 p_k)`` on commitment -- approximately E1, the
  thing that already failed. ``--woe_evidential_class_balance false`` measures
  that degeneration deliberately.
* ``woe_centering_mode: raw_uniform``. Centred features give
  ``w+ - w- = z - beta.mu``, so the optimised ranking is the evaluated one shifted
  per class; raw features make the two identical and drop the dependence on a
  running mean reset at every task boundary.

Wave 1 (this default) is the anchor-off cell at n=3. It answers the question E1
could only answer in retrospect -- does the objective do anything *on its own*,
against naive fine-tuning's 0.3030 / 0.6566 / -0.3536 -- and it is the only cell
that can be run correctly before Omega has been measured, since the anchor's
lambda has to be re-centred against whatever Omega the new objective produces
(``anchor strength = lambda * Omega``; readme B6 spans five decades of it).

Diagnostics come from ``WOE_EV_DEBUG=1`` (argmax agreement between the raw logit,
the centred logit and the evidential score; per-class readout norms and biases at
each consolidation; the train-vs-eval dropout conflict share) and
``WOE_LC_DEBUG=1`` (cumulative Omega at each consolidation, the logit/conflict
split of I_2). Both are set by this driver.

Usage:
    python scripts/run_evidential_loss_benchmark.py --jobs 3
    python scripts/run_evidential_loss_benchmark.py --arms anchor --seeds 0
    python scripts/run_evidential_loss_benchmark.py --collect-only
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

# The one-shot schedule every WoE result in docs/woe-cl/README.md was measured
# on. Kept here rather than on the command line so no arm can be run on a
# different schedule to the bar it is compared against.
ONE_SHOT: Tuple[str, ...] = ("--n_epochs", "1", "--inner_steps", "2")

# The project's standard triple. Seed 0 has pointed the wrong way four times in
# this campaign, so an n=1 reading is only ever a bracket.
DEFAULT_SEEDS: Tuple[int, ...] = (0, 39, 55)


@dataclass(frozen=True)
class Arm:
    """One configuration of the evidential objective."""

    name: str
    config: str
    extra_args: Tuple[str, ...] = field(default_factory=tuple)


ARMS: Dict[str, Arm] = {
    # Wave 1. Anchor off, so nothing depends on a lambda that has not been
    # re-centred yet, and the comparison is directly against naive fine-tuning:
    # does replacing CE with the evidential objective change retention at all?
    "noanchor": Arm("noanchor", "woe_si_evidential", ("--woe_lambda", "0")),
    # The same-code-path CE control for that cell. `woe_si` at woe_lambda=0 with
    # the objective off is naive fine-tuning through this module, which is a
    # tighter bar than the recorded B4 number from a different learner.
    "noanchor_ce": Arm(
        "noanchor_ce",
        "woe_si_evidential",
        ("--woe_lambda", "0", "--woe_evidential_mode", "off"),
    ),
    # The same cell as `noanchor`, evaluated with the rule it was trained for.
    # argmax over b_k (equivalently d_k/s_k) rather than over the raw logit: the
    # two disagreed on 3-18% of training samples and b was the more accurate of
    # the pair on 8 of 9 tasks, so the `noanchor` numbers are a lower bound of
    # unknown tightness until this arm exists.
    "noanchor_predict_b": Arm(
        "noanchor_predict_b",
        "woe_si_evidential",
        ("--woe_lambda", "0", "--woe_evidential_predict", "score"),
    ),
    # Wave 2. The anchor at the `abs` peak pinned in the config -- valid only
    # once wave 1's Omega trace says the anchor strength still lands where the
    # 2.4e5 measurement put it. Re-centre before trusting this cell.
    "anchor": Arm("anchor", "woe_si_evidential"),
    # The centred-feature host, for the one comparison that isolates what
    # raw_uniform costs or buys. Everything else on record uses this centring.
    "centered": Arm(
        "centered",
        "woe_si_evidential",
        ("--woe_lambda", "0", "--woe_centering_mode", "centered_uniform"),
    ),
    # The DS-faithful score instead of the scale-free one. Needs tau near the
    # measured w_plus; 4.0 is the default and matches this model's 4.1-5.3.
    "belief": Arm(
        "belief",
        "woe_si_evidential",
        ("--woe_lambda", "0", "--woe_evidential_mode", "belief"),
    ),
    # The deliberate degeneration: unweighted non-target pressure, which the
    # algebra says collapses into a net shrinkage of commitment (~E1).
    "unbalanced": Arm(
        "unbalanced",
        "woe_si_evidential",
        ("--woe_lambda", "0", "--woe_evidential_class_balance", "false"),
    ),
}

# Wave 1 is the default: one cell, three seeds, run concurrently.
DEFAULT_ARMS: Tuple[str, ...] = ("noanchor",)


@dataclass(frozen=True)
class Run:
    """One benchmark run: an arm and a seed."""

    arm: str
    seed: int

    @property
    def name(self) -> str:
        return f"evobj_{self.arm}_s{self.seed}"

    def command(self) -> List[str]:
        arm = ARMS[self.arm]
        return [
            PYTHON,
            str(REPO_ROOT / "main.py"),
            "--config",
            "configs/base.yaml",
            "--config",
            f"configs/models/til/{arm.config}.yaml",
            *ONE_SHOT,
            "--lr",
            "0.003",
            "--single-seed",
            "--seed",
            str(self.seed),
            "--no-save_checkpoints",
            "--expt_name",
            self.name,
            *arm.extra_args,
        ]


def build_runs(arms: List[str], seeds: Tuple[int, ...]) -> List[Run]:
    """Assemble the run list, seed-inner so one arm's triple lands together."""
    return [Run(arm, seed) for arm in arms for seed in seeds]


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
            # Both debug gates on: the objective's own diagnostics and the Omega
            # trace the anchor's lambda has to be re-centred against.
            environment = dict(
                os.environ,
                PYTHONUNBUFFERED="1",
                WOE_EV_DEBUG="1",
                WOE_LC_DEBUG="1",
            )
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

    header = f"{'arm':14s} {'seed':>5s} {'diag':>8s} {'final':>8s} {'bwt':>8s}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(
            _format_metric(row[key]) for key in ("diagonal_f1", "final_f1", "bwt")
        )
        print(f"{row['arm']:14s} {row['seed']:5d} {cells}")

    _print_seed_means(rows)
    print(f"\nwrote {csv_path}")


def _print_seed_means(rows: List[Dict[str, object]]) -> None:
    """Per-arm mean and sample sd over seeds, which is what gets believed."""
    by_arm: Dict[str, List[float]] = {}
    for row in rows:
        value = row["final_f1"]
        if isinstance(value, float):
            by_arm.setdefault(str(row["arm"]), []).append(value)
    if not by_arm:
        return
    print(f"\n{'arm':14s} {'n':>3s} {'final mean':>11s} {'sd':>8s}")
    print("-" * 38)
    for arm, values in by_arm.items():
        count = len(values)
        mean = sum(values) / count
        if count > 1:
            variance = sum((value - mean) ** 2 for value in values) / (count - 1)
            spread = f"{variance ** 0.5:8.4f}"
        else:
            spread = f"{'-':>8s}"
        print(f"{arm:14s} {count:3d} {mean:11.4f} {spread}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--arms",
        type=str,
        default="",
        help="Comma-separated subset of "
        + ",".join(ARMS)
        + f" (default: {','.join(DEFAULT_ARMS)}).",
    )
    argument_parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: the project's 0,39,55 triple).",
    )
    argument_parser.add_argument("--jobs", type=int, default=3)
    argument_parser.add_argument("--collect-only", action="store_true")
    argument_parser.add_argument(
        "--csv", type=str, default="scripts/logs/evidential_loss_benchmark.csv"
    )
    argument_parser.add_argument(
        "--log-root", type=str, default="scripts/logs/evidential_loss"
    )
    args = argument_parser.parse_args()

    arms = [name.strip() for name in args.arms.split(",") if name.strip()] or list(
        DEFAULT_ARMS
    )
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
