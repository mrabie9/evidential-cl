#!/usr/bin/env python
"""Sweep the ``I_p`` family: p=1 (sparsity) against p=2 (the recorded default).

Denoeux fixes ``p = 2`` in Eq 10 for tractability and says so; the exponent is a
free parameter. ``p = 1`` is a *different mechanism*, not a milder one: since
``w+_k + w-_k = sum_j |w_jk|``, ``I_1`` is the L1 norm of the weight-of-evidence
matrix, so minimising it drives most features to vacuity and concentrates the
evidence on a few. That is structurally what PackNet and HAT achieve by masking,
which would make ``p`` a one-parameter bridge between the regularisation and
architectural families in this project's own taxonomy.

Measured off-GPU first (the claim is checkable at the desk and was): alongside
cross-entropy, at strengths that both leave the task perfectly separable, ``p=1``
drives 44-53% of features below 5% of peak evidence while ``p=2`` plateaus around
19-22% however hard it is pushed. So the mechanism is real. What this sweep asks
is whether it *buys* anything in continual learning.

Read E1/E2 before the results
-----------------------------
The prior here is poor and should be stated up front. E1 found the ``p = 2``
objective inert on its own (0.3119, i.e. naive fine-tuning) and *destructive* on
top of the anchor (0.4847 -> 0.3198 at lambda 1e4), because an objective that
pushes ``I_2`` down moves 61% of Omega out of the positive part of the path
integral that consolidation keeps. E2 then found the DS-specific half was not
what was doing it.

That confound is designed out here rather than re-run:

* the ``eralg4`` arms have no path integral to starve at all, so the objective is
  measured where it cannot dismantle its host;
* the ``woe_si`` arms run under ``woe_omega_transform=abs``, which keeps the
  magnitude of the path integral rather than its positive part, so an
  objective-induced negative omega still counts as important -- with the anchor
  re-centred to abs's own peak of 2.4e5 (README "Not yet tried" #5, which this
  subsumes for the p=1 case).

Every arm includes a ``woe_lc_lambda=0`` control, without which "best setting"
and "no benefit" are indistinguishable.

Sizing lambda
-------------
``I_p`` is homogeneous of degree ``p`` in the readout, so even after the ``J^p``
normalisation the two exponents are decades apart and lambda cannot transfer.
The shift between the grids is ``J / w ~ 132``, derived from E1's own measured
p=2 term rather than guessed -- see ``P1_GRID`` for the arithmetic, and note the
guess ("roughly ``J``") was wrong by a factor of four in the direction that would
have run the sweep too weak to see anything. Verify with ``WOE_LC_DEBUG=1``
before believing any of it.

Usage:
    python scripts/run_ip_family_benchmark.py --jobs 2
    python scripts/run_ip_family_benchmark.py --arms woe_si_abs_p1 --seeds 0,39,55
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

ONE_SHOT: Tuple[str, ...] = ("--n_epochs", "1", "--inner_steps", "2")

# E1's measured grid for p=2: the raw term I_2 / J^2 sits at 6.9e-4 against a
# cross-entropy near 1.9, so lambda 1e1..1e4 runs the penalty from 0.4% to 390%
# of CE -- "numerically inert" to "dominates the objective", which is what
# brackets a peak.
P2_GRID: Tuple[float, ...] = (0.0, 1e1, 1e2, 1e3, 1e4)

# The p=1 grid is that grid divided by the *measured* ratio between the two
# terms, not by a guess. Both are per-feature averages (I_p / J^p), so
#
#     (I_1 / J) / (I_2 / J^2) = J * sum_k (w+ + w-) / sum_k (w+^2 + w-^2) ~ J / w
#
# for a typical channel magnitude w. Back that w out of E1's own measurement
# rather than assuming it: 6.9e-4 = sum_k (w+^2 + w-^2) / J^2 with J = 512 gives
# a per-class w+^2 + w-^2 of 30.1 over the ~6 columns of a task, i.e. w ~ 3.9 --
# consistent with the separately recorded finding that weights of evidence here
# are single digits, not O(J). So the ratio is 512 / 3.9 ~ 132, and the p=1 grid
# is the p=2 grid over 132.
#
# This matters: an earlier version of this file divided by J itself (~500),
# which is the ratio you get only if w ~ 1, and would have run the whole sweep a
# factor of four too weak. Re-derive it with WOE_LC_DEBUG=1 for any new J,
# schedule or learning rate -- no lambda in this project has ever transferred.
P1_GRID: Tuple[float, ...] = (0.0, 7.5e-2, 7.5e-1, 7.5e0, 7.5e1)

# A6's peak for the `abs` path integral. `relu`'s peak of 1e6 does NOT transfer.
ABS_PEAK = "240000.0"


@dataclass(frozen=True)
class Arm:
    """One host, at one exponent."""

    name: str
    config: str
    exponent: int
    lambdas: Tuple[float, ...]
    extra_args: Tuple[str, ...] = field(default_factory=tuple)


# The abs anchor, re-centred to its own peak: the host on which an objective that
# induces negative path integrals is not silently starved by the relu projection.
WOE_SI_ABS: Tuple[str, ...] = (
    "--lr",
    "0.003",
    "--woe_omega_transform",
    "abs",
    "--woe_anchor_mode",
    "proximal",
    "--woe_lambda",
    ABS_PEAK,
)

ARMS: Dict[str, Arm] = {
    # --- replay host: no path integral to dismantle ------------------------
    # eralg4 is plain reservoir ER, so the objective is measured somewhere it
    # cannot damage an importance estimator. Judged against its own lambda=0 arm
    # (0.6082 +/- 0.0040), never against the 0.6102 ER bar.
    "eralg4_p1": Arm("eralg4_p1", "eralg4", 1, P1_GRID),
    "eralg4_p2": Arm("eralg4_p2", "eralg4", 2, P2_GRID),
    # --- anchored host, with the starvation route closed -------------------
    # Subsumes README "Not yet tried" #5 for the p=1 case: the one arm that could
    # redeem the objective is the one where `abs` keeps the magnitude of the path
    # integral the term drives negative.
    "woe_si_abs_p1": Arm("woe_si_abs_p1", "woe_si_lc", 1, P1_GRID, WOE_SI_ABS),
    "woe_si_abs_p2": Arm("woe_si_abs_p2", "woe_si_lc", 2, P2_GRID, WOE_SI_ABS),
    # --- the tracked scalar, not the objective -----------------------------
    # B6 asked whether the DS content of the *measured* scalar distinguishes
    # anything and found it does not. This asks the sharper version: does the
    # exponent -- the part Denoeux chose for tractability rather than principle --
    # matter either? No LC objective at all; only what the path integral tracks.
    # lambda for the anchor is unchanged from abs's peak, which is an assumption:
    # I_1 and I_2 put Omega on different scales, so read a null here as "not
    # measured at its own peak" unless the Omega trace says otherwise.
    "scalar_i1": Arm(
        "scalar_i1",
        "woe_si_lc",
        2,
        (0.0,),
        WOE_SI_ABS + ("--woe_importance_scalar", "i1"),
    ),
}


@dataclass(frozen=True)
class Run:
    """One benchmark run: an arm, an objective strength and a seed."""

    arm: str
    lc_lambda: float
    seed: int

    @property
    def name(self) -> str:
        return f"ip_{self.arm}_lam{self.lc_lambda:g}_s{self.seed}"

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
            "--woe_lc_lambda",
            repr(self.lc_lambda),
            "--woe_lc_p",
            str(arm.exponent),
            "--single-seed",
            "--seed",
            str(self.seed),
            "--no-save_checkpoints",
            "--expt_name",
            self.name,
            *arm.extra_args,
        ]


def build_runs(arms: List[str], seeds: Tuple[int, ...]) -> List[Run]:
    """Assemble the run list, every arm's lambda=0 control first.

    Controls launch before the grid so a broken code path surfaces in the first
    few minutes, and so the bars exist even if the campaign is cut short.
    """
    runs: List[Run] = []
    for seed in seeds:
        for name in arms:
            if 0.0 in ARMS[name].lambdas:
                runs.append(Run(name, 0.0, seed))
    for seed in seeds:
        for name in arms:
            for lc_lambda in ARMS[name].lambdas:
                if lc_lambda != 0.0:
                    runs.append(Run(name, lc_lambda, seed))
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

    ``WOE_LC_DEBUG=1`` is exported so each run traces the raw penalty magnitude
    and the Omega mass it leaves behind -- the two numbers needed to tell "this
    exponent does not help" from "this lambda grid missed".
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
            "p": ARMS[run.arm].exponent,
            "lc_lambda": run.lc_lambda,
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
        f"{'arm':15s} {'p':>2s} {'lc_lambda':>10s} {'seed':>5s} "
        f"{'diag':>8s} {'final':>8s} {'bwt':>8s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(
            _format_metric(row[key]) for key in ("diagonal_f1", "final_f1", "bwt")
        )
        print(
            f"{row['arm']:15s} {row['p']:2d} {row['lc_lambda']:10g} "
            f"{row['seed']:5d} {cells}"
        )
    print(f"\nwrote {csv_path}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--arms",
        type=str,
        default="eralg4_p1,woe_si_abs_p1,scalar_i1",
        help="Comma-separated subset of "
        + ",".join(ARMS)
        + ". The default runs only the p=1 arms and the i1 scalar, because the "
        "p=2 objective is already measured (E1/E2) -- add the *_p2 arms only if "
        "a same-host control is wanted at the re-centred abs anchor.",
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
        "--csv", type=str, default="scripts/logs/ip_family_benchmark.csv"
    )
    argument_parser.add_argument(
        "--log-root", type=str, default="scripts/logs/ip_family"
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
