#!/bin/bash
# Readme open item 3: "Deterministic LwF teacher."
#
# `_lwf_distillation_loss` scores the frozen teacher with bn_training=True so it
# normalises with the current batch's statistics -- deliberate, and worth 0.068
# of final macro recall (B4). But ResNet1D.forward implements that as
# `self.model.train(True)`, a module-wide switch that also reactivates the
# backbone's four trunk dropout modules, so the distillation target is resampled
# every step. `--woe_teacher_dropout disable` zeroes `p` on the teacher copy,
# which keeps the batch statistics and removes the noise.
#
# Two cells, both at the B4 configuration (n_epochs 1, inner_steps 2, lr 0.003,
# nothing re-tuned): LwF only (woe_lambda=0), which isolates the teacher change,
# and anchor + LwF (relu, woe_lambda=1e6), B4's headline cell. Controls are
# re-run rather than paired against the recorded 2026-08-12 numbers because
# model/woe_si.py has moved since; a control that fails to reproduce 0.4811 /
# 0.5223 is itself the finding.
#
# 12 runs, two at a time -- three concurrent 10-task runs SIGKILLed a job once.
set -u

REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/teacher_dropout
mkdir -p "$LOGS"

launch () {
  local cell=$1 lam=$2 drop=$3 seed=$4
  local name="td-${cell}-${drop}-s${seed}"
  echo "[launch $(date +%H:%M:%S)] $name"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lwf.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform relu --woe_anchor_mode proximal \
    --woe_lambda "$lam" --woe_lwf_lambda 1.0 \
    --woe_teacher_dropout "$drop" \
    --single-seed --seed "$seed" --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1
  echo "[done $(date +%H:%M:%S)] $name"
}

for seed in 0 39 55; do
  launch lwfonly 0.0 keep "$seed" &
  launch lwfonly 0.0 disable "$seed" &
  wait
done

for seed in 0 39 55; do
  launch both 1000000.0 keep "$seed" &
  launch both 1000000.0 disable "$seed" &
  wait
done

echo "[teacher-dropout] all done"
