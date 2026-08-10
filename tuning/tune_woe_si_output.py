#!/usr/bin/env python3
"""Hyperparameter tuning harness for WoE-SI in output (evidence-distillation) mode.

``woe_reg_level='output'`` replaces the SI path-integral anchor with a functional
penalty: the drift of the DS per-class total evidence ``(w_plus, w_minus)`` on
previously-seen classes, measured against a frozen end-of-task teacher. It is not
on the path-integral scale, so it needs its own ``woe_lambda`` -- which is what this
sweep is for. The ``woe_si`` preset's ``[1e2, 1e5]`` range does not transfer.

Scale measured on the 10-task TIL run (seed 0, checkpoints from the replay-only
run, one batch per probe task):

    task   #old cls        CE   reg(raw)   CE/reg
       1          6    0.4212  0.0009072    464.3
       3         19    0.8806    0.03847     22.9
       5         33    0.4657   0.009709     48.0
       7         45    0.5326   0.009325     57.1
       9         58    0.8267    0.00254    325.4

CE/reg spans 23-464 across the sequence, whose geometric centre is ~103, so
lambda ~ 100 puts the penalty and the cross-entropy in the same range, and the
grid spans 1 (negligible) to 3000 (dominant -- the value inherited from the
parameter-mode grid, which scored a final macro-F1 of 0.107).

Usage:
    python tuning/tune_woe_si_output.py --hierarchical

    python tuning/tune_woe_si_output.py \\
        --config configs/tuning_defaults.yaml \\
        --config configs/models/til/woe_si_output.yaml \\
        --hierarchical

When ``--config`` is omitted, defaults are ``configs/tuning_defaults.yaml`` and
``configs/models/til/woe_si_output.yaml``. Do not pass ``configs/base.yaml``.

CAVEAT: ``configs/tuning_defaults.yaml`` targets a 3-task subset. The output-mode
penalty sums over *previously-seen* classes, so its magnitude grows with task index
(19 old classes at task 3, 58 at task 9 on the full run). A 3-task sweep therefore
sees a systematically smaller penalty than the full run and will bias the selected
lambda upward. Treat the result as a starting point and confirm on the full task
sequence.
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

main = make_main(TUNING_PRESETS["woe_si_output"])

if __name__ == "__main__":
    main()
