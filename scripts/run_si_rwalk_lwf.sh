#!/bin/bash
# Does a function-space regulariser add retention on top of a parameter-space
# anchor? The woe_si 2x2 (I_2 anchor x LwF) is repeated here for SI and RWalk,
# using the shared model/lwf_regulariser.py so all three methods distil through
# one code path.
#
#          anchor only   lwf only   anchor + lwf
#   si         X            X            X
#   rwalk      X            X            X
#
# Single-epoch TIL (n_epochs 1, inner_steps 2), 3 seeds. Both anchors run in
# their proximal form at the best strength measured so far (si_c = 1e3,
# lamb = 1e3); the as-configured 0.4 / 1.0 are numerically inert, so stacking on
# them would measure LwF against naive fine-tuning instead of against the anchor.
# The anchor-only arms are re-run here rather than cited from
# logs/rwalk/prox-loss_1e3lamb_se_til-* (0.5169 / -0.0458) because that row ran
# with bf16 AMP, which is known to shift final F1 by over a point on this
# benchmark and to flip cross-method orderings; every arm below is --no-amp.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
cd "$REPO" || exit 1

# name : config stem : extra flags
JOBS=(
  "si_anchoronly:si_lwf:--si_lwf_lambda 0.0"
  "si_lwfonly:si_lwf:--si_c 0.0"
  "si_anchor_lwf:si_lwf:"
  "rwalk_anchoronly:rwalk_lwf:--rwalk_lwf_lambda 0.0"
  "rwalk_lwfonly:rwalk_lwf:--lamb 0.0"
  "rwalk_anchor_lwf:rwalk_lwf:"
)

for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  rest="${job#*:}"
  cfg="${rest%%:*}"
  extra="${rest#*:}"
  echo "[si-rwalk-lwf] ===== START $name ====="
  # shellcheck disable=SC2086
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/$cfg.yaml" \
    --n_epochs 1 --inner_steps 2 \
    --no-amp \
    --no-save_checkpoints \
    --expt_name "${name}_noamp_se_til" \
    --seeds "0,39,55" \
    $extra
  echo "[si-rwalk-lwf] ===== END $name exit=$? ====="
done
echo "[si-rwalk-lwf] ALL DONE"
echo "[si-rwalk-lwf] results: logs/{si_lwf,rwalk_lwf}/*_noamp_se_til-*/{0,39,55}/results.txt"
