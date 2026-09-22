#!/usr/bin/env python3
"""Collapse a model's split seed-sweep run dirs under ``logs/00_sync`` into one.

Promoted runs sometimes accumulate more than one completed run directory per
model (e.g. a ``seedbase`` sweep with seeds 0/39/55 promoted separately from a
``seedext`` sweep with seeds 100/390/550). This script merges all of a
model's completed run directories (those with a top-level ``results.txt``)
into a single run directory holding every seed, e.g.::

    logs/00_sync/1e_CIL/saved_models/smaml/2026-09-17_..._one-pass_cil_lr0.03/{0,39,55}
    logs/00_sync/1e_CIL/saved_models/smaml/2026-09-18_..._seedext_1e_cil_smaml/{100,390,550}

becomes::

    logs/00_sync/1e_CIL/saved_models/smaml/2026-09-18_..._seedext_1e_cil_smaml/{0,39,55,100,390,550}

Before moving anything, it checks that every seed's ``training_parameters.json``
agrees outside the fields that legitimately vary per launch (seed,
task_order_seed, timestamp, expt_name, seeds) -- a mismatch means the runs are
not actually comparable, and the model is skipped with an error. After the
merge, the cross-seed ``results.txt`` is regenerated from the *full* set of
seeds using ``main._write_sweep_summary``, the same function the training
pipeline itself uses to write it, so the format matches exactly.

Every move and directory removal is journaled to
``logs/00_sync/.organise_journal.jsonl`` in the same ``{t, op, src, dst,
group}`` shape used by the original run-promotion pass, so the merge stays
auditable.

Run dirs without a ``results.txt`` (smoke tests, abandoned runs) are left
untouched, per the existing ``logs/00_sync`` promotion convention.

Usage:
    python scripts/merge_seed_runs.py logs/00_sync/1e_CIL/saved_models --dry-run
    python scripts/merge_seed_runs.py logs/00_sync/1e_CIL/saved_models
    python scripts/merge_seed_runs.py logs/00_sync/1e_CIL/saved_models --model smaml

A model whose runs fail the training_parameters.json consistency check or
have a seed shared by more than one run dir is skipped by default. Pass
``--force`` to go ahead anyway: mismatched hyperparameters are merged
regardless, and a colliding seed dir is set aside as
``<seed>__dup_from_<run>`` (never overwriting the target's copy, and left out
of the regenerated results.txt) for you to reconcile by hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from main import _write_sweep_summary  # noqa: E402  (reuse the canonical writer)

JOURNAL_NAME = ".organise_journal.jsonl"

# training_parameters.json fields that legitimately differ per seed/launch and
# must be excluded from the cross-seed consistency check.
VARIABLE_PARAM_FIELDS = {"seed", "task_order_seed", "timestamp", "expt_name", "seeds"}


class MergeError(Exception):
    """Raised when a model's runs cannot be safely merged."""


def find_completed_run_dirs(model_dir: Path) -> list[Path]:
    """Return this model's run dirs that have a top-level results.txt, sorted by name."""
    return sorted(
        p for p in model_dir.iterdir() if p.is_dir() and (p / "results.txt").is_file()
    )


def find_seed_dirs(run_dir: Path) -> list[Path]:
    """Return a run dir's seed subdirs (those with a seed_metrics.json), sorted by name."""
    return sorted(
        p
        for p in run_dir.iterdir()
        if p.is_dir() and (p / "seed_metrics.json").is_file()
    )


def _seed_sort_key(name: str):
    """Sort seed dir names numerically when possible, else lexically."""
    return (0, int(name)) if name.isdigit() else (1, name)


def load_filtered_training_parameters(seed_dir: Path) -> dict:
    """Load training_parameters.json with the known per-seed/per-launch fields dropped."""
    path = seed_dir / "training_parameters.json"
    with open(path, encoding="utf-8") as fh:
        params = json.load(fh)
    return {k: v for k, v in params.items() if k not in VARIABLE_PARAM_FIELDS}


def check_consistency(seed_dirs: list[Path]) -> None:
    """Raise MergeError if any seed's training_parameters.json disagrees with the rest."""
    reference, reference_dir = None, None
    for seed_dir in seed_dirs:
        filtered = load_filtered_training_parameters(seed_dir)
        if reference is None:
            reference, reference_dir = filtered, seed_dir
            continue
        if filtered != reference:
            diffs = {
                key: (reference.get(key), filtered.get(key))
                for key in set(reference) | set(filtered)
                if reference.get(key) != filtered.get(key)
            }
            raise MergeError(
                f"training_parameters.json mismatch between {reference_dir} and "
                f"{seed_dir}: {diffs}"
            )


def check_no_duplicate_seeds(run_dirs: list[Path]) -> None:
    """Raise MergeError if the same seed dir name appears under more than one run dir."""
    seen: dict[str, Path] = {}
    for run_dir in run_dirs:
        for seed_dir in find_seed_dirs(run_dir):
            if seed_dir.name in seen:
                raise MergeError(
                    f"seed {seed_dir.name!r} appears in both {seen[seed_dir.name]} "
                    f"and {seed_dir}; refusing to merge"
                )
            seen[seed_dir.name] = seed_dir


def append_journal(
    journal_path: Path, op: str, src: Path, dst: Path | None, group: str
) -> None:
    """Append one {t, op, src, dst, group} line to the organise journal."""
    entry = {
        "t": time.time(),
        "op": op,
        "src": str(src),
        "dst": str(dst) if dst is not None else None,
        "group": group,
    }
    with open(journal_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _dup_dest_path(dst: Path, source_run_dir: Path) -> Path:
    """Build a non-clobbering path for a seed dir whose name collides at the target."""
    return dst.parent / f"{dst.name}__dup_from_{source_run_dir.name}"


def merge_model(
    model_dir: Path, journal_path: Path, group: str, dry_run: bool, force: bool = False
) -> str:
    """Merge one model's completed run dirs into the most recent one.

    With ``force``, a training_parameters.json mismatch or a seed appearing in
    more than one run dir is downgraded from an aborting error to a printed
    warning: mismatched hyperparameters are merged anyway, and a colliding
    seed dir is moved in under a ``<seed>__dup_from_<run>`` name instead of
    being merged (or ever overwriting the target's copy), so no run data is
    lost or silently averaged in twice; it is excluded from the regenerated
    results.txt and left for manual reconciliation.

    Returns a short human-readable status string.
    """
    run_dirs = find_completed_run_dirs(model_dir)
    if len(run_dirs) <= 1:
        return "already single run, nothing to do"

    try:
        check_no_duplicate_seeds(run_dirs)
    except MergeError as exc:
        if not force:
            raise
        print(f"[{model_dir.name}] WARNING (--force): {exc}")

    all_seed_dirs = [
        seed_dir for run_dir in run_dirs for seed_dir in find_seed_dirs(run_dir)
    ]
    if not all_seed_dirs:
        raise MergeError("no seed_metrics.json found in any completed run dir")
    try:
        check_consistency(all_seed_dirs)
    except MergeError as exc:
        if not force:
            raise
        print(f"[{model_dir.name}] WARNING (--force): {exc}")

    target = run_dirs[-1]
    sources = run_dirs[:-1]

    if dry_run:
        moved = [
            seed_dir.name for run_dir in sources for seed_dir in find_seed_dirs(run_dir)
        ]
        return f"would merge {[r.name for r in sources]} -> {target.name} (seeds {sorted(moved, key=_seed_sort_key)})"

    dup_names: set[str] = set()
    for run_dir in sources:
        for seed_dir in find_seed_dirs(run_dir):
            dst = target / seed_dir.name
            if dst.exists():
                if not force:
                    raise MergeError(
                        f"destination {dst} already exists (unexpected duplicate)"
                    )
                dst = _dup_dest_path(dst, run_dir)
                dup_names.add(dst.name)
                print(
                    f"[{model_dir.name}] WARNING (--force): seed collision, moving {seed_dir} -> {dst.name}"
                )
            shutil.move(str(seed_dir), str(dst))
            append_journal(journal_path, "move", seed_dir, dst, group)

        remaining = [p for p in run_dir.iterdir() if p.is_dir()]
        if remaining:
            raise MergeError(
                f"{run_dir} still has unrecognised subdirs after moving seeds out "
                f"({[p.name for p in remaining]}); leaving it in place rather than deleting"
            )
        shutil.rmtree(run_dir)
        append_journal(journal_path, "rmtree", run_dir, None, group)

    seeds = sorted(
        (p.name for p in find_seed_dirs(target) if p.name not in dup_names),
        key=_seed_sort_key,
    )
    _write_sweep_summary(str(target), seeds)
    suffix = f", {len(dup_names)} duplicate(s) set aside" if dup_names else ""
    return f"merged {[r.name for r in sources]} into {target.name} ({len(seeds)} seeds{suffix})"


def _find_00_sync_root(path: Path) -> Path:
    """Walk up from path to find the enclosing "00_sync" dir; fall back to its parent."""
    resolved = path.resolve()
    for candidate in [resolved, *resolved.parents]:
        if candidate.name == "00_sync":
            return candidate
    return resolved.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "group_dir",
        help="A saved_models directory under logs/00_sync, e.g. "
        "logs/00_sync/1e_CIL/saved_models",
    )
    ap.add_argument(
        "--model",
        help="Only process this one model subdir (default: all models under group_dir).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be merged without moving or deleting anything.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Go ahead past a training_parameters.json mismatch or a seed shared by "
        "more than one run dir (both otherwise abort that model): mismatches are "
        "merged anyway, and a colliding seed dir is set aside as "
        "<seed>__dup_from_<run> instead of being merged, so nothing is overwritten "
        "or double-counted. Ignored with --dry-run.",
    )
    args = ap.parse_args(argv)

    group_dir = Path(args.group_dir)
    if not group_dir.is_dir():
        ap.error(f"not a directory: {group_dir}")

    model_dirs = (
        [group_dir / args.model]
        if args.model
        else sorted(p for p in group_dir.iterdir() if p.is_dir())
    )
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            ap.error(f"not a directory: {model_dir}")

    journal_path = _find_00_sync_root(group_dir) / JOURNAL_NAME
    group = group_dir.parent.name

    had_failure = False
    for model_dir in model_dirs:
        try:
            status = merge_model(
                model_dir, journal_path, group, args.dry_run, args.force
            )
        except MergeError as exc:
            had_failure = True
            print(f"[{model_dir.name}] SKIPPED: {exc}")
        else:
            print(f"[{model_dir.name}] {status}")

    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())
