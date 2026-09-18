#!/bin/bash
# A7 follow-up, batch 4. Bracketing the sum arm turned up something that matters
# beyond A7: `sum` at lambda 1.2e5 gives 0.5036, above the 0.5008 that A6
# recorded at 2.4e5 and that this project has used as *the* anchor bar ever
# since. If that replicates, the recorded peak is mis-placed by a factor of two
# and several comparisons are being made against a slightly mis-tuned control.
# Two seeds settle it.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/cautious_lowlam
mkdir -p "$LOGS"

for seed in 39 55; do
  name="caut_sum_lam120000_s${seed}"
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_lambda 120000.0 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
done
wait
echo "[batch4] done"
