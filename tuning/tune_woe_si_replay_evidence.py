#!/usr/bin/env python3
"""Hyperparameter tuning harness for the evidence-decay replay penalty.

``woe_replay_mode='both'`` adds, on top of ordinary rehearsal cross-entropy, a
one-sided penalty on the DS total evidence of a stored item having *decayed* since
it was written to the reservoir::

    L = mean_b sum_k [ relu(w+_stored - w+_now)^2 + relu(w-_now - w-_stored)^2 ]

Only deterioration is charged, so the term never competes for capacity the model
does not need. This sweep chooses ``woe_evidence_lambda``, which is on its own
scale -- the decay term is a J^2-normalised sum of squared hinges while the replay
term is an O(1) cross-entropy, so no other ``lambda`` in this codebase transfers.

Scale measured on the 10-task TIL run (seed 0, checkpoints from the replay-only
run, evidence snapshotted with the weights current at each task):

    task   buffer   CE(cur)   replayCE       decay   CE/decay
       2     1024    0.7565     1.3801   0.0002031       3725
       4     2048    1.0144     0.9317   0.0006194       1638
       6     3072    0.7827     1.3898   0.0002499       3132
       8     4096    0.5983     1.6847   0.0003036       1971
       9     4608    0.8524     1.3330   0.0001891       4508

CE/decay spans 1638-4508 with a geometric centre of ~2800, so lambda ~ 3000 puts
the two terms in the same range; the grid runs 0 to 30000.

``woe_evidence_lambda = 0`` is on the grid on purpose. Under ``woe_replay_mode=
'both'`` it recovers plain experience replay exactly, so the sweep carries its own
null arm and can answer whether the evidence term helps *at all* -- not merely
which positive value is least bad. Earlier WoE grids in this repo lacked such an
arm and could not distinguish "best setting" from "no benefit".

The parameter anchor is disabled (``woe_lambda: 0``) in the default config, since a
full 2x2 on the task sequence found it contributes nothing once replay is present.

Usage:
    python tuning/tune_woe_si_replay_evidence.py --hierarchical

    python tuning/tune_woe_si_replay_evidence.py \\
        --config configs/tuning_defaults.yaml \\
        --config configs/models/til/woe_si_replay_evidence.yaml \\
        --hierarchical

When ``--config`` is omitted, defaults are ``configs/tuning_defaults.yaml`` and
``configs/models/til/woe_si_replay_evidence.yaml``. Do not pass ``configs/base.yaml``.

CAVEAT: ``configs/tuning_defaults.yaml`` targets a 3-task subset, so the reservoir
never fills and evidence has little time to decay. The decay term will be smaller
there than on the full sequence, biasing the selected lambda upward. Confirm the
winner on the full task order.
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

main = make_main(TUNING_PRESETS["woe_si_replay_evidence"])

if __name__ == "__main__":
    main()
