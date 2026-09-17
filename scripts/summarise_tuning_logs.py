#!/usr/bin/env python3
"""Summarise hyperparameter-tuning suite logs (``logs/tuning/suites/<suite>/*.log``).

Each log is the captured stdout of one ``tuning/tune_<model>.py`` invocation as
written by a suite's ``sweep_job.sh``::

    [<iso-time>] START <mode> <model>
    ... per-trial training output ...
    SUMMARY_TR macro_rec=... macro_prec=... macro_f1=...
    SUMMARY_TE macro_rec=... macro_prec=... macro_f1=...
    [<stage>] Trial <i> finished | score=<s> | params={...}
    Trial <i> failed: <error>
    ...
    Best trial #<i> | score=<s> | params={...}
    Logs stored in: <run dir>
    Updated YAML config with best params: <yaml>
    [<iso-time>] END <mode> <model> rc=<rc>

For every log this prints one table per hierarchical stage (one row per trial:
the swept value, the tuner score, final test/train macro metrics and a
``nan@T<k>`` flag when the training loss first went NaN/inf on task ``k``),
followed by an overview table with one row per (mode, model).

Under ``--hierarchical`` the tuner's ``Best trial`` line only shows the winning
trial's own stage parameter; earlier stages' winners were carried in as fixed
overrides. The "full best params" column rebuilds that complete setting from
the per-stage winners in the log (ties go to the earliest trial, as in the
tuner).

Failed trials print no stage or params. Both are recovered from the trial's
run-directory slug (``tuning_sweep_tune_<idx>_<key>-<value>-...``) when the
trial got far enough to save a checkpoint; the slug value is the tuner's
compact encoding (``5em01`` = 5e-01, ``1p5`` = 1.5).

Usage:
    python scripts/summarise_tuning_logs.py logs/tuning/suites/<suite>
    python scripts/summarise_tuning_logs.py logs/tuning/suites/<suite>/til_si.log
    python scripts/summarise_tuning_logs.py logs/tuning/suites/<suite> --brief
    python scripts/summarise_tuning_logs.py logs/tuning/suites/<suite> --csv trials.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Searches are unanchored: torch warnings are sometimes written to the log
# without a trailing newline, gluing them to the start of the next line.
_START_RE = re.compile(r"\[([^\]]+)\] START (\S+) (\S+)")
_END_RE = re.compile(r"\[([^\]]+)\] END (\S+) (\S+) rc=(-?\d+)")
_FINISHED_RE = re.compile(
    r"(?:\[([^\]]+)\] )?Trial (\d+) finished \| score=(\S+) \| params=(\{.*\})"
)
_FAILED_RE = re.compile(r"Trial (\d+) failed: (.*)")
_BEST_RE = re.compile(r"Best trial #(\d+) \| (score=.*?) \| params=(\{.*\})")
_SUMMARY_RE = re.compile(r"SUMMARY_(TR|TE) (.*)")
_FIELD_RE = re.compile(r"(\w+)=(\S+)")
_NAN_LOSS_RE = re.compile(r"\bT(\d+) Ep \d+/\d+ \| L (?:nan|-?inf)\b")
_SLUG_RE = re.compile(r"tuning_sweep_tune_(\d+)_(.+?)-\d{4}-\d{2}-\d{2}_")
_SESSION_RE = re.compile(r"(logs/tuning/\S+?/\d{4}-\d{2}-\d{2}_[\d-]+)/runs/")
_STOPPED_RE = re.compile(r"Hierarchical stage '([^']+)' recorded no successful trials")


@dataclass
class Trial:
    idx: int
    status: str = "ok"
    stage: str | None = None
    params: dict[str, Any] | None = None
    slug: str | None = None
    score: float = math.nan
    te: dict[str, float] = field(default_factory=dict)
    tr: dict[str, float] = field(default_factory=dict)
    nan_task: int | None = None
    error: str | None = None


@dataclass
class TuningLog:
    path: Path
    mode: str = "?"
    model: str = "?"
    start: str | None = None
    end: str | None = None
    rc: int | None = None
    session: str | None = None
    trials: list[Trial] = field(default_factory=list)
    best_idx: int | None = None
    best_score_note: str | None = None
    yaml_updated: str | None = None
    stopped_stage: str | None = None
    no_success: bool = False


def _parse_fields(text: str) -> dict[str, float]:
    out = {}
    for key, raw in _FIELD_RE.findall(text):
        try:
            out[key] = float(raw)
        except ValueError:
            continue
    return out


def _literal(text: str) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def parse_log(path: Path) -> TuningLog:
    log = TuningLog(path=path)
    stem_mode, _, stem_model = path.stem.partition("_")
    log.mode, log.model = stem_mode or "?", stem_model or path.stem

    pending = Trial(idx=-1)
    with path.open(errors="replace") as handle:
        for line in handle:
            if m := _START_RE.search(line):
                log.start, log.mode, log.model = m.group(1), m.group(2), m.group(3)
                continue
            if m := _END_RE.search(line):
                log.end, log.rc = m.group(1), int(m.group(4))
                continue
            if m := _SUMMARY_RE.search(line):
                target = pending.te if m.group(1) == "TE" else pending.tr
                target.update(_parse_fields(m.group(2)))
                continue
            if m := _NAN_LOSS_RE.search(line):
                if pending.nan_task is None:
                    pending.nan_task = int(m.group(1))
            if m := _SLUG_RE.search(line):
                pending.slug = m.group(2)
                if log.session is None and (s := _SESSION_RE.search(line)):
                    log.session = s.group(1)
                continue
            if m := _FINISHED_RE.search(line):
                pending.idx = int(m.group(2))
                pending.stage = m.group(1)
                pending.score = float(m.group(3))
                pending.params = _literal(m.group(4))
                log.trials.append(pending)
                pending = Trial(idx=-1)
                continue
            if m := _FAILED_RE.search(line):
                pending.idx = int(m.group(1))
                pending.status = "failed"
                pending.error = m.group(2).strip()
                log.trials.append(pending)
                pending = Trial(idx=-1)
                continue
            if m := _BEST_RE.search(line):
                log.best_idx = int(m.group(1))
                log.best_score_note = m.group(2)
                continue
            if "Updated YAML config with best params:" in line:
                log.yaml_updated = line.split(":", 1)[1].strip()
                continue
            if m := _STOPPED_RE.search(line):
                log.stopped_stage = m.group(1)
                continue
            if "No successful trials were recorded." in line:
                log.no_success = True

    _fill_failed_trial_stages(log)
    return log


def _fill_failed_trial_stages(log: TuningLog) -> None:
    """Attribute failed trials to a stage via their run slug or neighbours."""
    last_stage = None
    for trial in log.trials:
        if trial.status == "ok":
            last_stage = trial.stage
            continue
        if trial.slug and (m := re.match(r"([A-Za-z_]\w*)-(.+)$", trial.slug)):
            trial.stage = m.group(1)
        else:
            trial.stage = last_stage


def stage_order(log: TuningLog) -> list[str | None]:
    order: list[str | None] = []
    for trial in log.trials:
        if trial.stage not in order:
            order.append(trial.stage)
    return order


def stage_winners(log: TuningLog) -> dict[str | None, Trial]:
    winners: dict[str | None, Trial] = {}
    for trial in log.trials:
        if trial.status != "ok" or math.isnan(trial.score):
            continue
        current = winners.get(trial.stage)
        if current is None or trial.score > current.score:
            winners[trial.stage] = trial
    return winners


def best_trial(log: TuningLog) -> Trial | None:
    by_idx = {t.idx: t for t in log.trials}
    if log.best_idx is not None and log.best_idx in by_idx:
        return by_idx[log.best_idx]
    ok = [t for t in log.trials if t.status == "ok" and not math.isnan(t.score)]
    return max(ok, key=lambda t: t.score) if ok else None


def full_best_params(log: TuningLog) -> dict[str, Any]:
    """Winning params including earlier hierarchical stages' carried winners."""
    best = best_trial(log)
    if best is None:
        return {}
    winners = stage_winners(log)
    merged: dict[str, Any] = {}
    for stage in stage_order(log):
        if stage == best.stage:
            break
        if stage in winners and winners[stage].params:
            merged.update(winners[stage].params)
    merged.update(best.params or {})
    return merged


def _duration(log: TuningLog) -> str:
    if not (log.start and log.end):
        return "running" if log.start else "?"
    try:
        secs = int(
            (
                datetime.fromisoformat(log.end) - datetime.fromisoformat(log.start)
            ).total_seconds()
        )
    except ValueError:
        return "?"
    return f"{secs // 60}m{secs % 60:02d}s"


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_params(params: dict[str, Any] | None) -> str:
    if not params:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _flags(trial: Trial) -> str:
    flags = []
    if trial.nan_task is not None:
        flags.append(f"nan@T{trial.nan_task}")
    if trial.status == "failed":
        flags.append(f"FAILED: {trial.error}")
    return " ".join(flags)


def _print_table(headers: list[str], rows: list[list[str]], indent: str = "") -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]

    def render(cells: list[str]) -> str:
        padded = [c.ljust(w) for c, w in zip(cells[:-1], widths[:-1])]
        return (indent + "  ".join(padded + [cells[-1]])).rstrip()

    print(render(headers))
    print(indent + "  ".join("-" * w for w in widths))
    for row in rows:
        print(render(row))


def print_log_detail(log: TuningLog) -> None:
    rc = "?" if log.rc is None else str(log.rc)
    print(f"== {log.mode} {log.model}  ({_duration(log)}, rc={rc}) ==")
    print(f"log:     {log.path}")
    if log.session:
        print(f"session: {log.session}")

    best = best_trial(log)
    winners = stage_winners(log)
    for stage in stage_order(log):
        trials = [t for t in log.trials if t.stage == stage]
        n_failed = sum(t.status == "failed" for t in trials)
        n_nan = sum(t.nan_task is not None for t in trials)
        winner = winners.get(stage)
        label = stage if stage is not None else "(no stage)"
        print(
            f"\n  stage {label}: {len(trials)} trials, {n_failed} failed, "
            f"{n_nan} NaN loss, stage best "
            f"{'#' + str(winner.idx) if winner else '-'}"
        )
        rows = []
        for t in trials:
            if t.params and stage in t.params:
                value = str(t.params[stage])
            elif t.params:
                value = _fmt_params(t.params)
            else:
                value = f"({t.slug})" if t.slug else "?"
            marker = (
                "*"
                if best is not None and t is best
                else ("+" if winner is not None and t is winner else " ")
            )
            rows.append(
                [
                    f"{marker}{t.idx}",
                    value,
                    _fmt(t.score),
                    _fmt(t.te.get("macro_rec")),
                    _fmt(t.te.get("macro_prec")),
                    _fmt(t.te.get("macro_f1")),
                    _fmt(t.tr.get("macro_f1")),
                    _flags(t),
                ]
            )
        _print_table(
            ["#", "value", "score", "te_rec", "te_prec", "te_f1", "tr_f1", "flags"],
            rows,
            indent="  ",
        )

    print()
    if best is not None:
        note = log.best_score_note or f"score={_fmt(best.score)}"
        print(f"  best: #{best.idx} ({best.stage}) {note}")
        print(f"  full best params: {_fmt_params(full_best_params(log))}")
    if log.stopped_stage:
        print(f"  hierarchical search stopped at stage '{log.stopped_stage}'")
    if log.no_success:
        print("  no successful trials were recorded")
    if log.yaml_updated:
        print(f"  YAML updated: {log.yaml_updated}")
    elif best is not None and log.end:
        print("  YAML not updated")
    print()


def print_overview(logs: list[TuningLog]) -> None:
    rows = []
    for log in sorted(
        logs,
        key=lambda lg: (
            lg.mode,
            -(best.score if (best := best_trial(lg)) else -math.inf),
            lg.model,
        ),
    ):
        best = best_trial(log)
        rows.append(
            [
                log.mode,
                log.model,
                str(len(log.trials)),
                str(sum(t.status == "failed" for t in log.trials)),
                str(sum(t.nan_task is not None for t in log.trials)),
                f"#{best.idx}" if best else "-",
                _fmt(best.score) if best else "-",
                _fmt(best.te.get("macro_f1")) if best else "-",
                _fmt(best.tr.get("macro_f1")) if best else "-",
                _duration(log),
                "?" if log.rc is None else str(log.rc),
                _fmt_params(full_best_params(log)),
            ]
        )
    print("== overview ==")
    _print_table(
        [
            "mode",
            "model",
            "trials",
            "failed",
            "nan",
            "best",
            "score",
            "te_f1",
            "tr_f1",
            "time",
            "rc",
            "full best params",
        ],
        rows,
    )


def write_csv(logs: list[TuningLog], out_path: Path) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "model",
                "trial",
                "stage",
                "status",
                "params",
                "slug",
                "score",
                "te_macro_rec",
                "te_macro_prec",
                "te_macro_f1",
                "tr_macro_rec",
                "tr_macro_prec",
                "tr_macro_f1",
                "nan_task",
                "is_stage_best",
                "is_best",
                "error",
            ]
        )
        for log in logs:
            best = best_trial(log)
            winners = stage_winners(log)
            for t in log.trials:
                writer.writerow(
                    [
                        log.mode,
                        log.model,
                        t.idx,
                        t.stage or "",
                        t.status,
                        _fmt_params(t.params) if t.params else "",
                        t.slug or "",
                        "" if math.isnan(t.score) else t.score,
                        t.te.get("macro_rec", ""),
                        t.te.get("macro_prec", ""),
                        t.te.get("macro_f1", ""),
                        t.tr.get("macro_rec", ""),
                        t.tr.get("macro_prec", ""),
                        t.tr.get("macro_f1", ""),
                        "" if t.nan_task is None else t.nan_task,
                        winners.get(t.stage) is t,
                        best is t,
                        t.error or "",
                    ]
                )


def collect_log_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            # xargs.log is the suite driver's own output, not a tuning run.
            paths.extend(p for p in sorted(path.glob("*.log")) if p.name != "xargs.log")
        elif path.is_file():
            paths.append(path)
        else:
            print(f"warning: {raw} does not exist, skipping", file=sys.stderr)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", nargs="+", help="Tuning log files and/or suite directories."
    )
    parser.add_argument(
        "--brief", action="store_true", help="Only print the overview table."
    )
    parser.add_argument(
        "--mode", choices=["til", "cil"], help="Only include logs for this mode."
    )
    parser.add_argument("--csv", type=Path, help="Also write one row per trial here.")
    args = parser.parse_args()

    logs = [parse_log(p) for p in collect_log_paths(args.inputs)]
    if args.mode:
        logs = [lg for lg in logs if lg.mode == args.mode]
    if not logs:
        print("no tuning logs found", file=sys.stderr)
        return 1

    if not args.brief:
        for log in sorted(logs, key=lambda lg: (lg.mode, lg.model)):
            print_log_detail(log)
    if len(logs) > 1 or args.brief:
        print_overview(logs)
    if args.csv:
        write_csv(logs, args.csv)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
