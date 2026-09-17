#!/bin/bash
# n=3 on the displacement peak. It is now the load-bearing cell: displacement
# alone reaches 0.4934 against the full path integral's 0.5008, so the gradient
# factor h appears to contribute only ~0.007 -- but 0.0074 is 2.4x the seed sd,
# which is exactly the size of effect this campaign has repeatedly seen reverse.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/displacement
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for seed in 39 55; do
  name="disp_lam600_s${seed}"
  job_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform displacement --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_lambda 600 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
done
wait
echo "[displacement seeds] done"
