#!/bin/bash
# No-trunk-dropout arm: all four ResNet1D trunk dropouts off (nn.Identity).
#
# Follows the decreasing-schedule arm (scripts/run_dropout_schedule.sh), which
# lost 0.0489 entirely out of the diagonal with BWT flat -- i.e. more dropout was
# pure plasticity cost at one epoch. This asks the other direction: is the
# *existing* flat p=0.2 also costing plasticity that the anchor never repaid?
#
# Everything else is byte-identical to scripts/run_i2_control_recheck.sh, whose
# n=3 flat-p=0.2 control ran on 2026-08-31 (0.4984 / 0.4941 / 0.4968 final
# cls_f1). model/resnet1d.py is still the only tracked file to have changed since
# then, and RESNET1D_DROPOUT pins the only behavioural difference, so those three
# runs remain the paired control and are not re-run.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/dropout_none
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for seed in 0 39 55; do
  name="dropnone_lam240000_s${seed}"
  job_slot
  echo "[launch] $name"
  RESNET1D_DROPOUT=0 WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
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
echo "[dropout none] done"
