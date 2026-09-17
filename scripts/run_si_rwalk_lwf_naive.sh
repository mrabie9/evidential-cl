#!/bin/bash
# The fourth cell of the scripts/run_si_rwalk_lwf.sh 2x2: neither mechanism on.
# Both knobs at 0 makes each learner plain SGD fine-tuning at its own lr, which
# is what the "LwF alone" and "anchor alone" contrasts are measured against.
# Matched to the other arms: single-epoch TIL, inner_steps 2, 3 seeds, no AMP.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
cd "$REPO" || exit 1

JOBS=(
  "si_neither:si_lwf:--si_c 0.0 --si_lwf_lambda 0.0"
  "rwalk_neither:rwalk_lwf:--lamb 0.0 --rwalk_lwf_lambda 0.0"
)

for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  rest="${job#*:}"
  cfg="${rest%%:*}"
  extra="${rest#*:}"
  echo "[si-rwalk-lwf-naive] ===== START $name ====="
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
  echo "[si-rwalk-lwf-naive] ===== END $name exit=$? ====="
done
echo "[si-rwalk-lwf-naive] ALL DONE"
