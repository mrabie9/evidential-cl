#!/usr/bin/env python
"""Sweep the Least-Commitment objective (``--woe_lc_lambda``) on two learners.

The term minimises the Dempster-Shafer information content ``I_2`` of the
readout's mass function on the current task alongside cross-entropy: commit no
more evidence than the data requires, leaving evidential room for later tasks
(``docs/woe-cl/README.md``, "Not yet tried" #5).

It is swept on the two hosts it composes with differently:

* ``woe_si`` with the parameter-space proximal anchor at its measured peak
  (``woe_lambda=1e6``, ``lr 0.003``) -- a replay-free arm, judged against the
  anchor-only cell (0.4847 at seed 0).
* ``eralg4`` -- reservoir experience replay, judged against its own
  ``woe_lc_lambda=0`` arm, since no one-shot eralg4 number is on record.

Every arm includes ``woe_lc_lambda=0``, which is what separates "best setting"
from "no benefit"; several earlier WoE grids lacked one and could not.

The grid is centred on a measurement, not a guess: the raw term is ``I_2 / J^2``
with ``J = 512``, measured at 6.9e-4 against a cross-entropy near 1.9 on the real
sequence. ``WOE_LC_DEBUG=1`` re-measures it for any new schedule or learning
rate -- no lambda in this project has ever transferred. See ``LAMBDA_GRID``.

Usage:
    WOE_LC_DEBUG=1 python main.py ... --woe_lc_lambda 1   # measure first
    python scripts/run_least_commitment_benchmark.py --jobs 2
    python scripts/run_least_commitment_benchmark.py --collect-only
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

# Measured, not guessed (WOE_LC_DEBUG=1, woe_si, lr 0.003, first steps of task 0
# of the real 10-task sequence): the raw term I_2 / J^2 sits at 6.9e-4 against a
# cross-entropy of 1.8-2.0. So lambda * term as a fraction of CE runs
#   1e1 -> 0.4%   1e2 -> 4%   1e3 -> 39%   1e4 -> 390%
# and this grid spans "numerically inert" to "dominates the objective", which is
# what brackets a peak. lambda=0 is the control arm every sweep here needs.
LAMBDA_GRID: Tuple[float, ...] = (0.0, 1e1, 1e2, 1e3, 1e4)


@dataclass(frozen=True)
class Arm:
    """One learner the objective is swept on."""

    name: str
    config: str
    # Flags that pin this arm to the protocol its reference number was measured
    # under (learning rate, anchor strength, ...).
    extra_args: Tuple[str, ...] = field(default_factory=tuple)


ARMS: Dict[str, Arm] = {
    # Proximal parameter anchor held at its peak; only the LC term moves.
    "woe_si": Arm("woe_si", "woe_si_lc", ("--lr", "0.003")),
    # Plain reservoir ER at the configured lr 0.01 and the 5120 buffer.
    "eralg4": Arm("eralg4", "eralg4"),
    # --- diagnostic arms, run after the grid to explain what it showed ---
    # The grid's woe_si series loses retention monotonically while the diagonal
    # holds. Two hypotheses predict exactly that, and these separate them.
    #
    # (1) The damage is intrinsic to the objective. Drop the anchor entirely
    #     (woe_lambda=0) and keep the LC term: if retention still degrades
    #     against the recorded naive bar (0.3030 / 0.6566 / -0.3536, readme B4),
    #     the objective harms old tasks on its own.
    "woe_si_noanchor": Arm(
        "woe_si_noanchor", "woe_si_lc", ("--lr", "0.003", "--woe_lambda", "0")
    ),
    # --- term ablation: charge the DS-specific half of I_2 instead of all of it ---
    # I_2 = ||z'||^2 + 2*sum_k w+_k*w-_k. The second half is the only part the
    # Dempster-Shafer reading contributes over a plain confidence penalty, and
    # measured per-sample over the real run it is 85% of the total (the split
    # trace, WOE_LC_DEBUG=1). Charging it alone is therefore a near-neighbour of
    # E1 rather than a new mechanism -- which is exactly why it is worth running:
    # if 85% of the penalty reproduces E1, the remaining logit half is inert; if
    # it does not, that 15% is carrying the effect. lambda is left on the E1 grid
    # because the terms are within 15% of each other in magnitude.
    #
    # Note the two halves are not weaker versions of each other. Minimising
    # conflict drives each class toward *one-sided* evidence and is indifferent
    # to its magnitude (huge w+ with zero w- scores zero conflict), so this is a
    # non-contradiction penalty, not a least-commitment one.
    "woe_si_conflict": Arm(
        "woe_si_conflict",
        "woe_si_lc",
        ("--lr", "0.003", "--woe_lc_term", "conflict"),
    ),
    "eralg4_conflict": Arm("eralg4_conflict", "eralg4", ("--woe_lc_term", "conflict")),
    # --- sign flip: SEEK conflict instead of penalising it (E4) ---------------
    # Hypothesis: penalising conflict pushes every class toward one-sided
    # evidence, which lets many features vote the same way and is exactly the
    # redundancy that leaves nothing free for later tasks. Rewarding it asks for
    # the opposite -- each class score a small residue of large opposing
    # evidence, so features have to take differing positions.
    #
    # Run with NEGATIVE --lambdas. `woe_si_kappa` is the arm to believe: the raw
    # `conflict` term is quadratic in the readout scale and unbounded above, and
    # cross-entropy only ever sees w+ - w-, so a negative lambda on it rewards an
    # inflation direction CE cannot object to. `kappa` is the same wish bounded
    # in [0, 1] by the belief transform, with tau=4 near the measured w+ of 4.1.
    # `woe_si_conflict` at negative lambda is kept as the literal reading, to
    # show that divergence rather than assume it.
    "woe_si_kappa": Arm(
        "woe_si_kappa",
        "woe_si_lc",
        ("--lr", "0.003", "--woe_lc_term", "kappa", "--woe_lc_tau", "4.0"),
    ),
    # --- the retest the E1 result asks for: a host the term cannot starve ---
    # E1's confirmed symptom is that the LC term moves 61% of Omega out of the
    # positive part of the path integral, because consolidation projects with
    # relu and this term exists to push I_2 *down*. `abs` keeps the magnitude,
    # so an LC-induced negative omega still counts as important and the
    # starvation should largely vanish. `abs` is also the better anchor on its
    # own (+0.0130 at n=3), and its peak lambda is 2.4e5, not the relu peak of
    # 1e6 -- Omega changes, so the anchor strength has to be re-centred with it
    # or the arm is only measuring the wrong lambda. Run at lc_lambda 0 and 100.
    "woe_si_abs": Arm(
        "woe_si_abs",
        "woe_si_lc",
        (
            "--lr",
            "0.003",
            "--woe_omega_transform",
            "abs",
            "--woe_lambda",
            "240000.0",
        ),
    ),
    # (2) The damage is the phi -> mu route. I_2 is zero both when the readout
    #     vanishes and when features collapse onto their running mean; detaching
    #     the backbone leaves only the first. If the loss disappears here, the
    #     term survives as a readout-only confidence penalty.
    "woe_si_readout": Arm(
        "woe_si_readout", "woe_si_lc", ("--lr", "0.003", "--woe_lc_readout_only")
    ),
}


@dataclass(frozen=True)
class Run:
    """One benchmark run: an arm, a penalty strength and a seed."""

    arm: str
    lc_lambda: float
    seed: int

    @property
    def name(self) -> str:
        return f"lcobj_{self.arm}_lam{self.lc_lambda:g}_s{self.seed}"

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
            "--single-seed",
            "--seed",
            str(self.seed),
            "--no-save_checkpoints",
            "--expt_name",
            self.name,
            *arm.extra_args,
        ]


def build_runs(
    arms: List[str], lambdas: Tuple[float, ...], seeds: Tuple[int, ...]
) -> List[Run]:
    """Assemble the run list, control arms first and both learners interleaved.

    Ordered lambda-outer so the two ``lc_lambda=0`` controls launch first: they
    are the bars everything else is read against, and launching one run per
    learner immediately means a broken code path shows up in the first few
    minutes rather than an hour into the campaign.
    """
    return [
        Run(arm, lc_lambda, seed)
        for lc_lambda in lambdas
        for arm in arms
        for seed in seeds
    ]


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
    """Join every run's summary metrics into one CSV and print it."""
    rows = []
    for run in runs:
        path = find_results(run)
        row: Dict[str, object] = {
            "arm": run.arm,
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
        f"{'arm':8s} {'lc_lambda':>10s} {'seed':>5s} "
        f"{'diag':>8s} {'final':>8s} {'bwt':>8s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(
            _format_metric(row[key]) for key in ("diagonal_f1", "final_f1", "bwt")
        )
        print(f"{row['arm']:8s} {row['lc_lambda']:10g} {row['seed']:5d} {cells}")
    print(f"\nwrote {csv_path}")


def _parse_floats(text: str, fallback: Tuple[float, ...]) -> Tuple[float, ...]:
    """Parse a comma-separated numeric list, falling back to the default grid."""
    if not text:
        return fallback
    return tuple(float(piece) for piece in text.split(",") if piece.strip())


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--arms",
        type=str,
        default="",
        help="Comma-separated subset of " + ",".join(ARMS) + " (default: all).",
    )
    argument_parser.add_argument(
        "--lambdas",
        type=str,
        default="",
        help="Comma-separated woe_lc_lambda grid (default: the measured grid).",
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
        "--csv", type=str, default="scripts/logs/least_commitment_benchmark.csv"
    )
    argument_parser.add_argument(
        "--log-root", type=str, default="scripts/logs/least_commitment"
    )
    args = argument_parser.parse_args()

    arms = [name.strip() for name in args.arms.split(",") if name.strip()] or list(ARMS)
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        sys.exit(f"unknown arm(s): {', '.join(unknown)}")
    lambdas = _parse_floats(args.lambdas, LAMBDA_GRID)
    seeds = tuple(int(piece) for piece in args.seeds.split(",") if piece.strip())

    runs = build_runs(arms, lambdas, seeds)
    if not runs:
        sys.exit("no runs selected")

    if not args.collect_only:
        execute(runs, args.jobs, REPO_ROOT / args.log_root)

    csv_path = REPO_ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    collect(runs, csv_path)


if __name__ == "__main__":
    main()
