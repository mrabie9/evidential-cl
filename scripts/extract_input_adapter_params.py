#!/usr/bin/env python3
"""Extract and display input_adapter parameters from a results.pt checkpoint.

The input_adapter (AdcIqAdapter or HatInputAdapter wrapping it) learns a linear
mapping from 3 input channels to 2 IQ channels. This script loads a saved
results.pt, finds the adapter parameters in the state_dict, and prints how
each of the 2 output channels is a linear combination of the 3 input channels.

Usage
-----
    python scripts/extract_input_adapter_params.py path/to/results.pt
    python scripts/extract_input_adapter_params.py logs/my_run/0/results.pt --format csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Repo root for imports if needed
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Allow loading checkpoints that pickle argparse.Namespace
try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None


def load_results_pt(path: Path) -> tuple:
    """Load the results.pt bundle; return (state_dict, args)."""
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if add_safe_globals is not None:
        try:
            import argparse as _argparse

            add_safe_globals([_argparse.Namespace])
        except Exception:
            pass
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, (tuple, list)) or len(data) < 3:
        raise ValueError(
            f"Expected results.pt to be a tuple (val_t, val_a, state_dict, ...); got {type(data)} len={len(data) if hasattr(data, '__len__') else 'N/A'}"
        )
    state_dict = data[2]
    args = data[5] if len(data) > 5 else None
    return state_dict, args


def find_input_adapter_keys(state_dict: dict) -> list[str]:
    """Return state_dict keys that belong to the input_adapter."""
    return [k for k in state_dict if "input_adapter" in k]


def get_combined_weight_bias(
    state_dict: dict, adapter_keys: list[str]
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    """Extract (weight_2x3, bias_2) for 3D path and (weight_4d, bias_4d_2) for 4D path if present.

    AdcIqAdapter has:
    - For 3D input (B, 3, L): proj_3ch Conv1d(3, 2, 1) -> weight (2, 3, 1), bias (2,)
    - For 4D input (B, 3, 2, L): weight (3,), shared across the I and Q
      channels, bias (2,). Older checkpoints saved before this weight was
      unified across channels have weight shape (2, 3) instead; both are
      recognized here.
    Returns (weight_3d, bias_3d, weight_4d, bias_4d). Any can be None if not found.
    """
    weight_3d: torch.Tensor | None = None
    bias_3d: torch.Tensor | None = None
    weight_4d: torch.Tensor | None = None
    bias_4d: torch.Tensor | None = None

    for k in adapter_keys:
        v = state_dict[k]
        if not torch.is_tensor(v):
            continue
        if k.endswith("proj_3ch.weight"):
            # (2, 3, 1) -> (2, 3)
            weight_3d = v.squeeze().clone()
        elif k.endswith("proj_3ch.bias"):
            bias_3d = v.clone()
        elif (
            k.endswith(".weight")
            and "proj_3ch" not in k
            and (
                (v.dim() == 1 and v.shape[0] == 3)
                or (v.dim() == 2 and v.shape[0] == 2 and v.shape[1] == 3)
            )
        ):
            weight_4d = v.clone()
        elif k.endswith(".bias") and "proj_3ch" not in k and v.numel() == 2:
            bias_4d = v.clone()

    return weight_3d, bias_3d, weight_4d, bias_4d


def format_linear_combination(
    weight: torch.Tensor, bias: torch.Tensor, channel_names: list[str]
) -> list[str]:
    """Describe each output channel as a linear combination of input channels.

    ``weight`` of shape (in_channels,) is a single combination shared by both
    IQ output channels (the current 4D-path format); shape
    (out_channels, in_channels) gives each output channel its own combination
    (the 3D-path format, and older 4D checkpoints).
    """
    weight = weight.detach().float()
    bias = bias.detach().float()
    lines = []
    if weight.dim() == 1:
        terms = [
            f"{weight[j].item():+.4f} * {channel_names[j]}"
            for j in range(weight.shape[0])
        ]
        expr = " ".join(terms)
        for i in range(bias.shape[0]):
            lines.append(
                f"  out[{i}] (IQ[{i}]), shared weights: {expr} {bias[i].item():+.4f}"
            )
        return lines
    for i in range(weight.shape[0]):
        terms = [
            f"{weight[i, j].item():+.4f} * {channel_names[j]}"
            for j in range(weight.shape[1])
        ]
        expr = " ".join(terms) + f" {bias[i].item():+.4f}"
        lines.append(f"  out[{i}] (IQ[{i}]): {expr}")
    return lines


def normalize_weight_rows(weight: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize weight so each output channel's coefficients sum to 1.

    Args:
        weight: Tensor of shape (out_channels, in_channels), or (in_channels,)
            for a combination shared across output channels.
        eps: Threshold for treating a row-sum as zero.

    Returns:
        Row-normalized tensor with the same shape.
    """
    if weight.dim() == 1:
        weight_sum = weight.sum()
        zero_sum = weight_sum.abs() <= eps
        denom = torch.where(zero_sum, torch.ones_like(weight_sum), weight_sum)
        normalized = weight / denom
        uniform = torch.full_like(weight, 1.0 / weight.size(0))
        return torch.where(zero_sum, uniform, normalized)
    if weight.dim() != 2:
        raise ValueError(
            f"Expected rank-1 or rank-2 weight tensor, got shape {tuple(weight.shape)}"
        )

    row_sums = weight.sum(dim=1, keepdim=True)
    zero_row_mask = row_sums.abs() <= eps
    denom = torch.where(zero_row_mask, torch.ones_like(row_sums), row_sums)
    normalized = weight / denom

    uniform = torch.full_like(weight, 1.0 / weight.size(1))
    normalized = torch.where(
        zero_row_mask.expand_as(normalized),
        uniform,
        normalized,
    )
    return normalized


def print_adapter_params(
    path: Path,
    state_dict: dict,
    *,
    channel_names: list[str] | None = None,
    output_format: str = "human",
    normalize_weights: bool = False,
) -> None:
    """Print input_adapter parameters in a readable way."""
    if channel_names is None:
        channel_names = ["ch0", "ch1", "ch2"]
    adapter_keys = find_input_adapter_keys(state_dict)
    if not adapter_keys:
        print(f"No keys containing 'input_adapter' in {path}")
        return

    weight_3d, bias_3d, weight_4d, bias_4d = get_combined_weight_bias(
        state_dict, adapter_keys
    )

    if normalize_weights:
        if weight_3d is not None:
            weight_3d = normalize_weight_rows(weight_3d)
        if weight_4d is not None:
            weight_4d = normalize_weight_rows(weight_4d)

    print(f"Input adapter parameters from: {path}")
    print("Keys found:", adapter_keys)
    print()

    if weight_3d is not None and bias_3d is not None:
        print("3D path (B, 3, L) -> (B, 2, L) via proj_3ch (1x1 conv):")
        print("Weight matrix (2 x 3): rows = output IQ channels, cols = input channels")
        print(weight_3d.numpy())
        print("Bias (2,):", bias_3d.numpy())
        print("Linear combination (out = weight @ channels + bias):")
        for line in format_linear_combination(weight_3d, bias_3d, channel_names):
            print(line)
        if output_format == "csv":
            print(
                "csv_3d_weight:",
                ",".join(f"{x:.6f}" for x in weight_3d.flatten().tolist()),
            )
            print("csv_3d_bias:", ",".join(f"{x:.6f}" for x in bias_3d.tolist()))
        print()

    if weight_4d is not None and bias_4d is not None:
        shape_note = "(3,), shared across IQ" if weight_4d.dim() == 1 else "(2, 3)"
        print(
            f"4D path (B, 3, 2, L) -> (B, 2, L) via einsum + weight {shape_note} + bias:"
        )
        print(f"Weight {tuple(weight_4d.shape)}:")
        print(weight_4d.numpy())
        print("Bias (2,):", bias_4d.numpy())
        print("Linear combination:")
        for line in format_linear_combination(weight_4d, bias_4d, channel_names):
            print(line)
        if output_format == "csv":
            print(
                "csv_4d_weight:",
                ",".join(f"{x:.6f}" for x in weight_4d.flatten().tolist()),
            )
            print("csv_4d_bias:", ",".join(f"{x:.6f}" for x in bias_4d.tolist()))
        print()

    if weight_3d is None and weight_4d is None:
        print("No 2x3 weight or proj_3ch found in adapter keys.")


def _tensor_changed_from_reference(
    tensor: torch.Tensor, reference: torch.Tensor, atol: float = 1e-8
) -> bool:
    """Return True when tensor differs from a reference tensor."""
    if tensor.shape != reference.shape:
        return True
    return not torch.allclose(tensor, reference, atol=atol, rtol=0.0)


def evaluate_adapter_training_status(
    state_dict: dict,
    *,
    atol: float = 1e-8,
) -> tuple[bool, dict[str, bool]]:
    """Heuristically determine whether input adapter parameters were updated.

    The 4D path parameters have deterministic defaults:
    - weight_4d starts at all ones, shape (3,) (or (2, 3) in older checkpoints)
    - bias_4d starts at all zeros with shape (2,)

    We mark the adapter as "trained" when either deterministic default changes.
    For 3D path Conv1d weights, defaults are random and cannot be compared
    against a fixed reference without an initialization snapshot.

    Args:
        state_dict: Model state dict loaded from a checkpoint.
        atol: Absolute tolerance for equality checks.

    Returns:
        Tuple of:
            - trained flag (bool)
            - detail flags (dict)
    """
    adapter_keys = find_input_adapter_keys(state_dict)
    weight_3d, bias_3d, weight_4d, bias_4d = get_combined_weight_bias(
        state_dict, adapter_keys
    )

    details = {
        "found_adapter_keys": bool(adapter_keys),
        "found_weight_4d": weight_4d is not None,
        "found_bias_4d": bias_4d is not None,
        "weight_4d_changed_from_init": False,
        "bias_4d_changed_from_init": False,
        "found_weight_3d": weight_3d is not None,
        "found_bias_3d": bias_3d is not None,
    }

    if weight_4d is not None:
        details["weight_4d_changed_from_init"] = _tensor_changed_from_reference(
            weight_4d.detach().float(),
            torch.ones_like(weight_4d.detach().float()),
            atol=atol,
        )

    if bias_4d is not None:
        details["bias_4d_changed_from_init"] = _tensor_changed_from_reference(
            bias_4d.detach().float(),
            torch.zeros_like(bias_4d.detach().float()),
            atol=atol,
        )

    trained = bool(
        details["weight_4d_changed_from_init"] or details["bias_4d_changed_from_init"]
    )
    return trained, details


def find_latest_results_pt_for_algo(logs_root: Path, algo: str) -> Path | None:
    """Find the newest results.pt under logs_root/algo."""
    algo_dir = logs_root / algo
    if not algo_dir.exists():
        return None

    candidates = [p for p in algo_dir.glob("**/results.pt") if p.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def scan_latest_runs(logs_root: Path, algos: list[str], atol: float = 1e-8) -> int:
    """Scan latest checkpoint per algorithm and print adapter training status."""
    print(f"Scanning latest runs under: {logs_root}")
    print()

    missing: list[str] = []
    for algo in algos:
        latest_results = find_latest_results_pt_for_algo(logs_root, algo)
        if latest_results is None:
            missing.append(algo)
            print(f"[{algo}] latest results.pt: NOT FOUND")
            print()
            continue

        state_dict, _ = load_results_pt(latest_results)
        trained, details = evaluate_adapter_training_status(state_dict, atol=atol)
        status = "TRAINED" if trained else "UNCHANGED_FROM_4D_INIT"
        print(f"[{algo}] latest results.pt: {latest_results}")
        print(f"  status: {status}")
        print(
            "  checks: "
            f"weight_4d_changed={details['weight_4d_changed_from_init']}, "
            f"bias_4d_changed={details['bias_4d_changed_from_init']}, "
            f"found_weight_3d={details['found_weight_3d']}"
        )
        print()

    if missing:
        print("Algorithms with no checkpoint found:", ", ".join(missing))
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_pt",
        type=Path,
        nargs="?",
        help="Path to results.pt file",
    )
    parser.add_argument(
        "--scan-latest",
        nargs="+",
        default=None,
        metavar="ALGO",
        help="Scan latest results.pt for each listed algorithm under logs root",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root logs directory for --scan-latest (default: logs)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for init-comparison checks (default: 1e-8)",
    )
    parser.add_argument(
        "--format",
        choices=("human", "csv"),
        default="human",
        help="Output format: human-readable or csv-style lines",
    )
    parser.add_argument(
        "--channel-names",
        nargs=3,
        default=None,
        metavar=("C0", "C1", "C2"),
        help="Names for the 3 input channels (e.g. I Q ADC)",
    )
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Row-normalize adapter weights before printing (each output row sums to 1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scan_latest:
        return scan_latest_runs(args.logs_root, args.scan_latest, atol=float(args.atol))
    if args.results_pt is None:
        raise ValueError(
            "Provide results_pt path or use --scan-latest ALGO [ALGO ...]."
        )
    state_dict, _ = load_results_pt(args.results_pt)
    print_adapter_params(
        args.results_pt,
        state_dict,
        channel_names=args.channel_names,
        output_format=args.format,
        normalize_weights=args.normalize_weights,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
