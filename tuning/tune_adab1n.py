#!/usr/bin/env python3
"""Hyperparameter tuning harness for the AdaB1N norm layer.

AdaB1N replaces BatchNorm1d inside the shared ResNet1D backbone, so it is tuned
on a host learner rather than on its own. Pick the host with ``--host``::

    python tuning/tune_adab1n.py --host eralg4
    python tuning/tune_adab1n.py --host ft --seeds 0,39,55  # averages 3 seeds AND 3 task orders

Hosts:

``eralg4``
    Replay batches mix tasks, so AdaB1N's per-task reweighting is active and
    ``adab1n_init_weight`` is swept alongside ``kappa``.
``ft``
    The same host with ``memories: 0``, so no replay is drawn and nothing
    continual-learning-specific is active. Batches are single-task, the
    reweighting cannot engage, and only ``kappa`` is swept.
``iid2``
    Maximal-replay upper bound: the continual schedule, with task ``t`` trained
    on tasks ``0..t`` combined.

Each host supplies its own config chain: the host fragment first (for its tuned
learning rate), then an AdaB1N fragment. Order matters — the harness writes the
winning parameters into the *last* ``--config`` file, so the AdaB1N fragment must
come last to keep norm-layer state out of the host's baseline config. Passing
``--config`` explicitly replaces the whole chain, and therefore also redirects
where the results are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from tuning.hyperparam_tuner import make_main
    from tuning.presets import TUNING_PRESETS
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from tuning.hyperparam_tuner import make_main
    from tuning.presets import TUNING_PRESETS

# host -> (preset key, host config fragment, AdaB1N fragment written back to)
HOSTS = {
    "eralg4": (
        "adab1n_eralg4",
        "configs/models/til/eralg4.yaml",
        "configs/models/til/adab1n_eralg4.yaml",
    ),
    "ft": (
        "adab1n_ft",
        "configs/models/til/eralg4.yaml",
        "configs/models/til/adab1n_ft.yaml",
    ),
    "iid2": (
        "adab1n_iid2",
        "configs/models/til/iid2.yaml",
        "configs/models/til/adab1n_iid2.yaml",
    ),
}
DEFAULT_HOST = "eralg4"


def split_host(argv: list[str]) -> tuple[str, list[str]]:
    """Pull ``--host <name>`` (or ``--host=<name>``) out of ``argv``.

    The tuning harness builds its own parser and reads ``sys.argv`` directly, so
    the host selector is consumed here before handing the rest over.

    Usage:
        >>> split_host(["--host", "ft", "--seeds", "0"])
        ('ft', ['--seeds', '0'])
        >>> split_host(["--host=ft"])
        ('ft', [])
        >>> split_host(["--seeds", "0"])
        ('eralg4', ['--seeds', '0'])
    """
    remaining = list(argv)
    host = DEFAULT_HOST
    for index, token in enumerate(remaining):
        if token == "--host":
            if index + 1 >= len(remaining):
                raise SystemExit(f"--host requires a value, one of {sorted(HOSTS)}")
            host = remaining[index + 1]
            del remaining[index : index + 2]
            break
        if token.startswith("--host="):
            host = token.split("=", 1)[1]
            del remaining[index]
            break
    if host not in HOSTS:
        raise SystemExit(f"unknown --host {host!r}; expected one of {sorted(HOSTS)}")
    return host, remaining


def default_config_args(host: str) -> list[str]:
    """Config chain for ``host``, AdaB1N fragment last so writes land there."""
    _, host_cfg, adab1n_cfg = HOSTS[host]
    return ["--config", host_cfg, "--config", adab1n_cfg]


def main(argv: list[str] | None = None) -> None:
    host, remaining = split_host(sys.argv[1:] if argv is None else list(argv))
    if not any(
        token == "--config" or token.startswith("--config=") for token in remaining
    ):
        remaining = [*default_config_args(host), *remaining]
    sys.argv = [sys.argv[0], *remaining]
    make_main(TUNING_PRESETS[HOSTS[host][0]])()


if __name__ == "__main__":
    main()
