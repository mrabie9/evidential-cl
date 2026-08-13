#!/usr/bin/env python3
"""Hyperparameter tuning harness for WoE-SI + reservoir experience replay.

Usage:
    python tuning/tune_woe_si_replay.py --hierarchical

    python tuning/tune_woe_si_replay.py \\
        --config configs/tuning_defaults.yaml \\
        --config configs/models/til/woe_si_replay.yaml \\
        --hierarchical

When ``--config`` is omitted, defaults are ``configs/tuning_defaults.yaml`` and
``configs/models/til/woe_si_replay.yaml``. Do not pass ``configs/base.yaml`` for
tuning. Add ``--avg-seeds`` to average each trial over seeds 0/39/55.
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

main = make_main(TUNING_PRESETS["woe_si_replay"])

if __name__ == "__main__":
    main()
