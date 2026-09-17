#!/usr/bin/env python3
"""
Probe concurrency effects by sampling tqdm ETA for each job.

This script runs `main.py` under a pseudo-TTY
so tqdm is enabled, waits until the first epoch begins, then extracts the tqdm
ETA after N iterations.

At the moment the ETA is extracted, it also records `nvidia-smi` GPU utilization.
Results are written to JSON for later summarisation.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

TQDM_LINE_REGEX = re.compile(
    r"(?P<done>\d+)\s*/\s*(?P<total>\d+)\s*\[.*?<(?P<eta>[0-9:.]+).*?\]"
)
TQDM_DESC_REGEX = re.compile(
    r"T(?P<task>\d+)\s*\|\s*Ep:\s*(?P<ep>\d+)\s*/\s*(?P<ep_total>\d+)"
)
SINGLE_ROUND_TQDM_DESC_REGEX = re.compile(r"Ep:\s*(?P<ep>\d+)\s*/\s*(?P<ep_total>\d+)")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_nvidia_smi_util() -> Dict[str, Any]:
    """Return current GPU util snapshot from `nvidia-smi`.

    Returns:
        Dict with keys:
        - gpus: list of per-GPU dicts {utilization.gpu}
    """

    query = "utilization.gpu"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {
            "error": f"nvidia-smi failed rc={proc.returncode}",
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    gpus: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 1:
            gpus.append({"index": idx, "raw": line})
            continue
        util_s = parts[0]
        gpus.append(
            {
                "index": idx,
                "utilization.gpu": float(util_s),
            }
        )
    return {"gpus": gpus}


def _parse_eta_seconds(eta_str: str) -> Optional[float]:
    """Parse tqdm ETA string into seconds.

    Args:
        eta_str: ETA substring as seen in tqdm brackets, e.g. "00:00", "01:23",
            "00:00:10", or "12s".

    Returns:
        Seconds as float, or None if parsing fails.
    """

    cleaned = eta_str.strip()
    if cleaned.endswith("s"):
        try:
            return float(cleaned[:-1])
        except ValueError:
            return None

    if ":" in cleaned:
        parts = cleaned.split(":")
        try:
            parts_int = [float(p) for p in parts]
        except ValueError:
            return None
        # Support H:MM:SS and MM:SS
        if len(parts_int) == 2:
            minutes, seconds = parts_int
            return minutes * 60.0 + seconds
        if len(parts_int) == 3:
            hours, minutes, seconds = parts_int
            return hours * 3600.0 + minutes * 60.0 + seconds

    try:
        return float(cleaned)
    except ValueError:
        return None


def _select_first_task_token_from_base(base_config_path: Path) -> str:
    """Select the first comma-separated task token from `task_order_files`."""

    data = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    order = data.get("task_order_files", "")
    if not isinstance(order, str) or not order.strip():
        raise SystemExit(
            f"base.yaml at {base_config_path} has empty/missing task_order_files."
        )
    tokens = [tok.strip() for tok in order.split(",") if tok.strip()]
    if not tokens:
        raise SystemExit(
            f"base.yaml at {base_config_path} has task_order_files but no tokens."
        )
    return tokens[0]


def _resolve_model_config_path(
    models_dir: Path, model_name: str, model_mode: str
) -> Path:
    """Resolve model config path for legacy and mode-nested layouts.

    Resolution order:
      1) configs/models/<name>.yaml
      2) configs/models/<model_mode>/<name>.yaml
    """

    legacy_path = models_dir / f"{model_name}.yaml"
    if legacy_path.exists():
        return legacy_path

    mode_path = models_dir / model_mode / f"{model_name}.yaml"
    if mode_path.exists():
        return mode_path

    raise SystemExit(
        "Missing model config for "
        f"'{model_name}'. Checked: '{legacy_path}' and '{mode_path}'."
    )


def _spawn_with_pty(args: list[str]) -> Tuple[subprocess.Popen[str], int]:
    """Spawn a subprocess attached to a pseudo-TTY.

    Returns:
        (proc, master_fd)
    """

    import pty  # stdlib, imported lazily

    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        args,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        text=True,
        bufsize=1,
        close_fds=True,
    )
    os.close(slave_fd)
    return proc, master_fd


def _set_fd_nonblocking(fd: int) -> None:
    import fcntl  # stdlib

    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


@dataclass
class ProbeTarget:
    model_name: str
    base_config_path: Path
    model_config_path: Path
    task_token: str


def _build_main_command(
    target: ProbeTarget,
    n_epochs: int,
    samples_per_task: Optional[int],
) -> list[str]:
    """Build the `python3 ...` command for a single probe job."""

    entrypoint = "main.py"

    cmd = [
        sys.executable,
        entrypoint,
        "--config",
        str(target.base_config_path),
        "--config",
        str(target.model_config_path),
        "--task-order-files",
        target.task_token,
        "--n_epochs",
        str(n_epochs),
    ]
    if samples_per_task is not None:
        cmd += ["--samples_per_task", str(samples_per_task)]
    return cmd


def _parse_stream_until_eta(
    master_fd: int,
    proc: subprocess.Popen[str],
    wait_iters: int,
    expected_task: Optional[int],
    expected_epoch: int,
    expected_ep_total: int,
) -> Dict[str, Any]:
    """Read PTY output and capture ETA + avg GPU util for the window.

    Args:
        master_fd: PTY master fd.
        proc: subprocess running main.py.
        wait_iters: minimum `done` count to consider ETA capture.
        expected_task: optional task id to match in tqdm description (for
            lifelong/CL baselines that include `T{task} | ...` in the tqdm
            description). For single-round baselines set to `None`.
        expected_epoch: expected epoch number (1-indexed in tqdm descriptions).
        expected_ep_total: expected `ep_total` value (usually `n_epochs`).

    Returns:
        Dict with parsed fields:
        - eta_seconds
        - eta_raw
        - done, total
        - util_avg_epochstart_to_eta_capture_percent: mean `utilization.gpu`
          across samples in the epoch->ETA window.
        - util_avg_n_samples: number of util samples used for the mean.
    """

    _set_fd_nonblocking(master_fd)
    buffer = ""

    epoch_started = False
    epoch_start_time: Optional[float] = None
    last_seen_done: Optional[int] = None

    eta_seconds: Optional[float] = None
    eta_raw: Optional[str] = None
    done_val: Optional[int] = None
    total_val: Optional[int] = None

    start_time = time.time()
    util_poll_interval_s = 0.2
    util_samples: list[Tuple[float, float]] = []  # (t_rel_sec, util_percent)
    util_poll_stop_event = threading.Event()
    util_poll_thread: Optional[threading.Thread] = None
    util_avg_epochstart_to_eta_capture_percent: Optional[float] = None
    util_avg_n_samples: Optional[int] = None

    def _extract_util_percent(snapshot: Dict[str, Any]) -> Optional[float]:
        """Extract `utilization.gpu` from an `_run_nvidia_smi_util()` snapshot."""

        gpus = snapshot.get("gpus")
        if not isinstance(gpus, list) or not gpus:
            return None
        gpu0 = gpus[0]
        if not isinstance(gpu0, dict):
            return None
        util_val = gpu0.get("utilization.gpu")
        if isinstance(util_val, (int, float)):
            return float(util_val)
        return None

    def _take_util_sample_immediately() -> None:
        """Take one util sample and append to `util_samples`."""

        if epoch_start_time is None:
            return
        snap = _run_nvidia_smi_util()
        if not isinstance(snap, dict):
            return
        util_percent = _extract_util_percent(snap)
        if util_percent is None:
            return
        t_rel = time.time() - epoch_start_time
        util_samples.append((t_rel, util_percent))

    def _maybe_start_util_polling() -> None:
        """Start the background util polling thread (once) after epoch start."""

        nonlocal util_poll_thread
        if epoch_start_time is None:
            return

        util_samples.clear()
        util_poll_stop_event.clear()

        def _poll_loop() -> None:
            first = True
            while not util_poll_stop_event.is_set():
                # We already took a sample right at epoch detection; wait
                # for the first polling interval before sampling again.
                if first:
                    first = False
                    util_poll_stop_event.wait(util_poll_interval_s)
                    continue

                snap = _run_nvidia_smi_util()
                if isinstance(snap, dict):
                    util_percent = _extract_util_percent(snap)
                    if util_percent is not None and epoch_start_time is not None:
                        t_rel = time.time() - epoch_start_time
                        util_samples.append((t_rel, util_percent))

                util_poll_stop_event.wait(util_poll_interval_s)

        util_poll_thread = threading.Thread(target=_poll_loop, daemon=True)
        util_poll_thread.start()

    def _stop_and_compute_util_avg(eta_capture_time: float) -> None:
        """Stop polling and compute the average util in the epoch->ETA window."""

        nonlocal util_avg_epochstart_to_eta_capture_percent, util_avg_n_samples
        util_poll_stop_event.set()
        if util_poll_thread is not None:
            util_poll_thread.join(timeout=1.0)

        if epoch_start_time is None:
            util_avg_epochstart_to_eta_capture_percent = None
            util_avg_n_samples = None
            return

        window_end_rel = eta_capture_time - epoch_start_time
        used = [
            util_val
            for t_rel, util_val in util_samples
            if t_rel <= window_end_rel + 1e-6
        ]
        util_avg_n_samples = len(used)
        util_avg_epochstart_to_eta_capture_percent = (
            sum(used) / len(used) if used else None
        )

    while True:
        # Process might already be exiting; still try to drain.
        try:
            chunk = os.read(master_fd, 65536)
        except BlockingIOError:
            chunk = b""
        except OSError as e:
            # On some systems, when the PTY gets closed (e.g. child exits),
            # `os.read` can raise EIO (errno=5) instead of returning b''.
            if getattr(e, "errno", None) == 5:
                util_poll_stop_event.set()
                if util_poll_thread is not None:
                    util_poll_thread.join(timeout=1.0)
                rc = proc.poll()
                return {
                    "process_exit_code": rc,
                    "eta_seconds": eta_seconds,
                    "eta_raw": eta_raw,
                    "done": done_val,
                    "total": total_val,
                    "util_avg_epochstart_to_eta_capture_percent": util_avg_epochstart_to_eta_capture_percent,
                    "util_avg_n_samples": util_avg_n_samples,
                    "buffer_tail": buffer[-2000:],
                    "probe_elapsed_seconds": time.time() - start_time,
                }
            raise

        if chunk:
            buffer += chunk.decode("utf-8", errors="replace")

        # Try to detect "epoch start" from tqdm description.
        if not epoch_started:
            # Epoch start for CL baselines includes task id prefix: `T{task} | Ep: ...`.
            if expected_task is not None:
                for m in TQDM_DESC_REGEX.finditer(buffer):
                    task_id = int(m.group("task"))
                    ep = int(m.group("ep"))
                    ep_total = int(m.group("ep_total"))
                    if (
                        task_id == expected_task
                        and ep == expected_epoch
                        and ep_total == expected_ep_total
                    ):
                        epoch_started = True
                        epoch_start_time = time.time()
                        _take_util_sample_immediately()
                        _maybe_start_util_polling()
                        break

            # Epoch start for single-round baselines is `Ep: {}/{} | ...` (no `T{task}` prefix).
            if not epoch_started:
                for m in SINGLE_ROUND_TQDM_DESC_REGEX.finditer(buffer):
                    ep = int(m.group("ep"))
                    ep_total = int(m.group("ep_total"))
                    if ep == expected_epoch and ep_total == expected_ep_total:
                        epoch_started = True
                        epoch_start_time = time.time()
                        _take_util_sample_immediately()
                        _maybe_start_util_polling()
                        break

        # Attempt to parse ETA once we’re in the right epoch.
        if epoch_started:
            for m in TQDM_LINE_REGEX.finditer(buffer):
                done = int(m.group("done"))
                total = int(m.group("total"))
                if done < wait_iters:
                    continue
                if last_seen_done is not None and done == last_seen_done:
                    continue

                # We assume this tqdm line belongs to the current epoch/task.
                # This is typically stable under PTY + n_epochs=1.
                eta_raw_candidate = m.group("eta")
                parsed = _parse_eta_seconds(eta_raw_candidate)
                if parsed is None:
                    continue

                eta_seconds = parsed
                eta_raw = eta_raw_candidate
                done_val = done
                total_val = total
                last_seen_done = done
                eta_capture_time = time.time()
                _stop_and_compute_util_avg(eta_capture_time=eta_capture_time)
                return {
                    "eta_seconds": eta_seconds,
                    "eta_raw": eta_raw,
                    "done": done_val,
                    "total": total_val,
                    "util_avg_epochstart_to_eta_capture_percent": util_avg_epochstart_to_eta_capture_percent,
                    "util_avg_n_samples": util_avg_n_samples,
                    "probe_elapsed_seconds": time.time() - start_time,
                }

        # Terminate if the process exits before capture.
        rc = proc.poll()
        if rc is not None:
            return {
                "process_exit_code": rc,
                "eta_seconds": eta_seconds,
                "eta_raw": eta_raw,
                "done": done_val,
                "total": total_val,
                "util_avg_epochstart_to_eta_capture_percent": util_avg_epochstart_to_eta_capture_percent,
                "util_avg_n_samples": util_avg_n_samples,
                "buffer_tail": buffer[-2000:],
                "probe_elapsed_seconds": time.time() - start_time,
            }

        time.sleep(0.05)


def _terminate_process_tree(
    proc: subprocess.Popen[str], timeout_s: float = 3.0
) -> None:
    """Best-effort termination for a PTY-spawned process."""

    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=timeout_s)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _probe_one_job(
    target: ProbeTarget,
    wait_iters: int,
    out_queue: queue.Queue[Tuple[str, Dict[str, Any]]],
    job_tag: str,
    n_epochs: int,
    samples_per_task: Optional[int],
) -> None:
    """Run one probe job and push result into `out_queue`."""

    cmd = _build_main_command(
        target=target,
        n_epochs=n_epochs,
        samples_per_task=samples_per_task,
    )

    proc, master_fd = _spawn_with_pty(cmd)
    try:
        expected_task_for_tqdm = 0
        result = _parse_stream_until_eta(
            master_fd=master_fd,
            proc=proc,
            wait_iters=wait_iters,
            expected_task=expected_task_for_tqdm,
            expected_epoch=1,
            expected_ep_total=n_epochs,
        )
        result["job_tag"] = job_tag
        result["command"] = cmd
        result["start_time_utc"] = _now_iso()
    finally:
        try:
            _terminate_process_tree(proc)
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass

    out_queue.put((job_tag, result))


def _compute_pair_comparison(
    serial: Dict[str, Any],
    parallel: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute makespan-like metric: max parallel ETA vs serial sum."""

    serial_sum_eta = float(serial["eta_seconds_A"]) + float(serial["eta_seconds_B"])
    parallel_max_eta = max(
        float(parallel["eta_seconds_A_parallel"]),
        float(parallel["eta_seconds_B_parallel"]),
    )
    return {
        "serial_sum_eta_seconds": serial_sum_eta,
        "parallel_max_eta_seconds": parallel_max_eta,
        "parallel_max_eta_minus_serial_sum_eta_seconds": parallel_max_eta
        - serial_sum_eta,
        "parallel_is_faster_than_serial_sum": parallel_max_eta < serial_sum_eta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe tqdm ETA under different concurrency and record GPU util."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Model config names to run. Resolution prefers "
            "configs/models/<name>.yaml, then "
            "configs/models/<model_mode>/<name>.yaml."
        ),
    )
    parser.add_argument(
        "--model-mode",
        type=str,
        choices=["til", "cil"],
        default="til",
        help="Mode used when resolving nested model configs.",
    )
    parser.add_argument(
        "--pair-models",
        nargs=2,
        required=False,
        help="Two model config names for the parallel run.",
    )
    parser.add_argument(
        "--wait-iters",
        type=int,
        default=10,
        help="Wait until tqdm done >= this before capturing ETA.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=1,
        help="Number of epochs to run; this script expects 1 so Ep: 1/1 occurs once.",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=None,
        help="Optional override for --samples_per_task for speed.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Path to configs/base.yaml.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--append-out-json",
        action="store_true",
        default=False,
        help=(
            "If set and --out-json already exists, merge newly probed "
            "`serial` results into the existing JSON instead of overwriting. "
            "Only supported for serial-only probing."
        ),
    )
    parser.add_argument(
        "--auto-pair",
        action="store_true",
        default=False,
        help=(
            "If set (and --pair-models is omitted), run serial probes for all --models, "
            "then automatically pair highest util with lowest util (opposite ends) and "
            "run each pair with 2 jobs in parallel."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=2,
        help="Maximum number of high-low pairs to run when --auto-pair is enabled.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    task_token = _select_first_task_token_from_base(base_config_path)

    repo_root = Path.cwd()
    models_dir = repo_root / "configs" / "models"

    for model_name in args.models:
        _resolve_model_config_path(models_dir, model_name, args.model_mode)

    mode: str = "serial_only"
    serial_model_names: list[str] = list(args.models)
    model_a: Optional[str] = None
    model_b: Optional[str] = None

    if args.pair_models is not None:
        mode = "serial_and_parallel"
        model_a, model_b = args.pair_models
        if args.auto_pair:
            raise SystemExit("Use either --pair-models or --auto-pair, not both.")
        if model_a not in args.models or model_b not in args.models:
            raise SystemExit("--pair-models must be included in --models.")
        serial_model_names = [model_a, model_b]
    elif args.auto_pair:
        mode = "auto_pairing_serial_and_parallel"

    if args.append_out_json and mode != "serial_only":
        raise SystemExit(
            "--append-out-json is only supported for serial-only probes "
            "(i.e. run without --pair-models and without --auto-pair)."
        )

    # Probe serial results (either all requested models, or only the pair).
    out_queue: queue.Queue[Tuple[str, Dict[str, Any]]] = queue.Queue()
    serial_results: Dict[str, Dict[str, Any]] = {}
    for model_name in serial_model_names:
        target = ProbeTarget(
            model_name=model_name,
            base_config_path=base_config_path,
            model_config_path=_resolve_model_config_path(
                models_dir, model_name, args.model_mode
            ).resolve(),
            task_token=task_token,
        )
        job_tag = f"serial_{model_name}"
        _probe_one_job(
            target=target,
            wait_iters=args.wait_iters,
            out_queue=out_queue,
            job_tag=job_tag,
            n_epochs=args.n_epochs,
            samples_per_task=args.samples_per_task,
        )
        tag, result = out_queue.get()
        serial_results[tag] = result

    parallel_results: Optional[Dict[str, Dict[str, Any]]] = None
    parallel_wall_elapsed: Optional[float] = None
    comparison: Optional[Dict[str, Any]] = None
    pair_models_payload: Optional[Dict[str, str]] = None
    auto_pair_comparisons: list[Dict[str, Any]] = []
    auto_pair_models: list[Dict[str, str]] = []
    auto_parallel_wall_total_seconds: float = 0.0

    if mode == "serial_and_parallel":
        assert model_a is not None and model_b is not None

        # Probe parallel: A and B concurrently.
        parallel_results = {}
        threads: list[threading.Thread] = []
        for model_name, job_tag in (
            (model_a, f"parallel_{model_a}"),
            (model_b, f"parallel_{model_b}"),
        ):
            target = ProbeTarget(
                model_name=model_name,
                base_config_path=base_config_path,
                model_config_path=_resolve_model_config_path(
                    models_dir, model_name, args.model_mode
                ).resolve(),
                task_token=task_token,
            )
            t = threading.Thread(
                target=_probe_one_job,
                kwargs={
                    "target": target,
                    "wait_iters": args.wait_iters,
                    "out_queue": out_queue,
                    "job_tag": job_tag,
                    "n_epochs": args.n_epochs,
                    "samples_per_task": args.samples_per_task,
                },
                daemon=True,
            )
            threads.append(t)

        parallel_wall_start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        parallel_wall_elapsed = time.time() - parallel_wall_start

        # Collect two parallel results.
        while len(parallel_results) < 2:
            tag, result = out_queue.get()
            if tag.startswith("parallel_"):
                parallel_results[tag] = result

        # Prepare schema for comparison metric.
        serial_eta_seconds_A = float(serial_results[f"serial_{model_a}"]["eta_seconds"])
        serial_eta_seconds_B = float(serial_results[f"serial_{model_b}"]["eta_seconds"])
        parallel_eta_seconds_A = float(
            parallel_results[f"parallel_{model_a}"]["eta_seconds"]
        )
        parallel_eta_seconds_B = float(
            parallel_results[f"parallel_{model_b}"]["eta_seconds"]
        )

        comparison = _compute_pair_comparison(
            serial={
                "eta_seconds_A": serial_eta_seconds_A,
                "eta_seconds_B": serial_eta_seconds_B,
            },
            parallel={
                "eta_seconds_A_parallel": parallel_eta_seconds_A,
                "eta_seconds_B_parallel": parallel_eta_seconds_B,
            },
        )
        pair_models_payload = {"A": model_a, "B": model_b}
    elif mode == "auto_pairing_serial_and_parallel":
        # Pair highest serial util with lowest serial util.
        serial_util_items: list[Tuple[str, float]] = []
        for model_name in serial_model_names:
            util_val = serial_results[f"serial_{model_name}"].get(
                "util_avg_epochstart_to_eta_capture_percent"
            )
            if util_val is None:
                continue
            serial_util_items.append((model_name, float(util_val)))

        if not serial_util_items:
            raise SystemExit(
                "Could not compute any util_avg values from serial results."
            )

        serial_util_items_sorted = sorted(
            serial_util_items, key=lambda x: x[1], reverse=True
        )

        num_models = len(serial_util_items_sorted)
        num_pairs_possible = num_models // 2
        max_pairs = max(0, min(args.max_pairs, num_pairs_possible))
        if max_pairs == 0:
            raise SystemExit(
                f"Not enough models with util to form pairs (models={num_models})."
            )

        for pair_idx in range(max_pairs):
            high_model = serial_util_items_sorted[pair_idx][0]
            low_model = serial_util_items_sorted[-1 - pair_idx][0]

            auto_pair_models.append({"A": high_model, "B": low_model})

            # Run parallel probe for this pair.
            pair_queue: queue.Queue[Tuple[str, Dict[str, Any]]] = queue.Queue()
            pair_parallel_results: Dict[str, Dict[str, Any]] = {}
            threads: list[threading.Thread] = []
            for model_name, job_tag in (
                (high_model, f"parallel_{high_model}"),
                (low_model, f"parallel_{low_model}"),
            ):
                target = ProbeTarget(
                    model_name=model_name,
                    base_config_path=base_config_path,
                    model_config_path=_resolve_model_config_path(
                        models_dir, model_name, args.model_mode
                    ).resolve(),
                    task_token=task_token,
                )
                t = threading.Thread(
                    target=_probe_one_job,
                    kwargs={
                        "target": target,
                        "wait_iters": args.wait_iters,
                        "out_queue": pair_queue,
                        "job_tag": job_tag,
                        "n_epochs": args.n_epochs,
                        "samples_per_task": args.samples_per_task,
                    },
                    daemon=True,
                )
                threads.append(t)

            pair_wall_start = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            pair_wall_elapsed = time.time() - pair_wall_start
            auto_parallel_wall_total_seconds += pair_wall_elapsed

            # Collect the two parallel results.
            while len(pair_parallel_results) < 2:
                tag, result = pair_queue.get()
                if tag.startswith("parallel_"):
                    pair_parallel_results[tag] = result

            serial_eta_seconds_high = float(
                serial_results[f"serial_{high_model}"]["eta_seconds"]
            )
            serial_eta_seconds_low = float(
                serial_results[f"serial_{low_model}"]["eta_seconds"]
            )

            parallel_eta_seconds_high = float(
                pair_parallel_results[f"parallel_{high_model}"]["eta_seconds"]
            )
            parallel_eta_seconds_low = float(
                pair_parallel_results[f"parallel_{low_model}"]["eta_seconds"]
            )

            pair_comparison = _compute_pair_comparison(
                serial={
                    "eta_seconds_A": serial_eta_seconds_high,
                    "eta_seconds_B": serial_eta_seconds_low,
                },
                parallel={
                    "eta_seconds_A_parallel": parallel_eta_seconds_high,
                    "eta_seconds_B_parallel": parallel_eta_seconds_low,
                },
            )
            pair_comparison["pair_idx"] = pair_idx
            pair_comparison["serial_util_high_percent"] = float(
                serial_util_items_sorted[pair_idx][1]
            )
            pair_comparison["serial_util_low_percent"] = float(
                serial_util_items_sorted[-1 - pair_idx][1]
            )
            pair_comparison["parallel_wall_clock_seconds"] = pair_wall_elapsed

            auto_pair_comparisons.append(pair_comparison)

            parallel_max_eta = pair_comparison["parallel_max_eta_seconds"]
            serial_sum_eta = pair_comparison["serial_sum_eta_seconds"]
            delta = pair_comparison["parallel_max_eta_minus_serial_sum_eta_seconds"]
            print(
                f"[auto-pair {pair_idx+1}/{max_pairs}] "
                f"{high_model}(util={serial_util_items_sorted[pair_idx][1]:.1f}%) + "
                f"{low_model}(util={serial_util_items_sorted[-1 - pair_idx][1]:.1f}%) "
                f"=> serial_eta_sum={serial_sum_eta:.3f}s, "
                f"parallel_eta_max={parallel_max_eta:.3f}s, delta={delta:.3f}s "
                f"(parallel_wall={pair_wall_elapsed:.3f}s)"
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serial_payload = {
        model_name: serial_results[f"serial_{model_name}"]
        for model_name in serial_model_names
    }

    parallel_payload = None
    if parallel_results is not None and pair_models_payload is not None:
        parallel_payload = {
            pair_models_payload["A"]: parallel_results[
                f"parallel_{pair_models_payload['A']}"
            ],
            pair_models_payload["B"]: parallel_results[
                f"parallel_{pair_models_payload['B']}"
            ],
        }

    payload = {
        "schema_version": 1,
        "created_utc": _now_iso(),
        "timestamp_utc_compact": timestamp,
        "base_config": str(base_config_path),
        "task_token_first_from_base": task_token,
        "wait_iters": args.wait_iters,
        "n_epochs": args.n_epochs,
        "samples_per_task": args.samples_per_task,
        "mode": mode,
        "pair_models": pair_models_payload,
        "serial": serial_payload,
        "parallel": parallel_payload,
        "parallel_wall_clock_seconds": parallel_wall_elapsed,
        "comparison": comparison,
        "auto_pair_models": auto_pair_models if auto_pair_models else None,
        "auto_pair_comparisons": (
            auto_pair_comparisons if auto_pair_comparisons else None
        ),
        "auto_parallel_wall_total_seconds": (
            auto_parallel_wall_total_seconds
            if mode == "auto_pairing_serial_and_parallel"
            else None
        ),
    }

    if args.append_out_json and out_path.exists():
        existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(existing_payload, dict):
            raise SystemExit(f"Existing out-json is not an object: {out_path}")

        existing_serial = existing_payload.get("serial")
        if not isinstance(existing_serial, dict):
            raise SystemExit(
                f"Existing out-json missing/invalid 'serial' mapping: {out_path}"
            )

        # Merge/overwrite entries for the newly probed models.
        for model_name in serial_model_names:
            existing_serial[model_name] = serial_results[f"serial_{model_name}"]

        existing_payload["serial"] = existing_serial
        # Keep the original metadata where possible, but refresh the timestamp.
        existing_payload["timestamp_utc_compact"] = timestamp
        existing_payload["n_epochs"] = args.n_epochs
        existing_payload["samples_per_task"] = args.samples_per_task
        existing_payload["wait_iters"] = args.wait_iters

        out_path.write_text(json.dumps(existing_payload, indent=2), encoding="utf-8")
        print(f"Appended probe results to: {out_path}")
    else:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote probe results to: {out_path}")


if __name__ == "__main__":
    main()
