#!/bin/bash
# Close EWC's uniform-Omega bracket. Its grid (0.12/0.4/1.2/4/12 ->
# 0.3088/0.3558/0.3514/0.3868/0.4231) rises monotonically to the EDGE, so 0.4231
# is a lower bound on the uniform peak and EWC's +0.0561 gap is an upper bound on
# what its Fisher is worth. SI and WoE-SI both peak in the interior (lambda 3) and
# need no extension.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/omega_uniform
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for lam in 40 120 400; do
  name="omguni_ewc_lam${lam}"
  job_slot
  echo "[launch] $name"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/ewc.yaml \
    --n_epochs 1 --inner_steps 2 \
    --anchor_mode proximal --anchor_omega_uniform \
    --lamb "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
done
wait
echo "[chain] ewc uniform bracket closed"
