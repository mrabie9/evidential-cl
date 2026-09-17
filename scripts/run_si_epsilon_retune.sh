#!/bin/bash
# Addendum to scripts/run_si_epsilon_sweep.sh: re-tune si_c at eps=1e-6.
#
# The sweep held si_c/epsilon fixed, which isolates the normalisation's *shape*
# but does not ask whether normalised SI could match the shipped point under its
# own best strength. At eps=1e-6 the sweep has si_c=0.1 (47.72) and si_c=1000
# (43.64), which bracket the optimum without locating it. These two fill the
# interior. If the peak stays below the shipped 51.59, the normalisation is
# harmful under re-tuning too, not just at matched strength.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
cd "$REPO" || exit 1

JOBS=(
  "si_eps1em6_c1:--si_epsilon 0.000001 --si_c 1.0"
  "si_eps1em6_c10:--si_epsilon 0.000001 --si_c 10.0"
)

echo "[si-eps-retune] ${#JOBS[@]} jobs queued"
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  extra="${job#*:}"
  echo "[si-eps-retune] ===== START $name ($extra) ====="
  # shellcheck disable=SC2086
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/si_lwf.yaml" \
    --si_lwf_lambda 0.0 \
    --n_epochs 1 --inner_steps 2 \
    --no-amp \
    --no-save_checkpoints \
    --expt_name "${name}_noamp_se_til" \
    --seeds "0,39,55" \
    $extra
  echo "[si-eps-retune] ===== END $name exit=$? ====="
done
echo "[si-eps-retune] ALL DONE"
