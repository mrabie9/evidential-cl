#!/bin/bash
# Dropout-schedule arm: the backbone's four trunk dropouts were changed by hand
# from a flat p=0.2 to a decreasing 0.5 / 0.4 / 0.3 / 0.2 (model/resnet1d.py).
# Hypothesis: heavier dropout early stops the SI path integral from concentrating
# importance on a handful of parameters, which should make the anchor's Omega
# broader and the anchor itself more useful.
#
# Everything else is byte-identical to scripts/run_i2_control_recheck.sh, whose
# n=3 control ran on 2026-08-31 (0.4968 / 0.4984 / 0.4941 final cls_f1). The only
# tracked file to change since is model/resnet1d.py, and the only behavioural
# change in it is the dropout schedule -- so those three runs are a valid paired
# control and are not re-run here.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/dropout_schedule
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for seed in 0 39 55; do
  name="dropsched_lam240000_s${seed}"
  job_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar i2 \
    --woe_lambda 240000.0 \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
done
wait
echo "[dropout schedule] done"
