#!/bin/bash
# PR-3, E3: n=3 for both arms at the MATCHED deciding lambda.
#
# Amendment 3 moved the deciding statistic from peak-to-peak to matched lambda:
# the pre-pass may reshape the lambda->retention curve rather than offset it, and
# a difference cancels a shared perturbation only where both arms sit at the same
# lambda. Peak-to-peak would fold a differential reshape into D' with no way to
# separate it afterwards.
#
# lambda = 2.4e5 because it is (a) the confirmed no-pre-pass i2 peak on this
# host, (b) the cell with the measured on-peak seed sd of 0.0031 that the bands
# are built on, and (c) the only cell with a matched no-pre-pass n=3 behind it,
# which is what makes Amendment 1's clause 3 checkable.
#
# Four runs: seeds 39 and 55 for each arm. Seed 0 already exists for both.
# Amendment 1 clause 3 (is the pre-pass behaving as common-mode?) reads off the
# `ema` half of this set: if its n=3 mean sits more than 1 sigma (0.0031) from
# the recorded no-pre-pass 0.5008, D' must be re-examined before it is read.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/mu_frozen
DUMPS=$REPO/scripts/logs/mu_frozen/omega_dumps
mkdir -p "$LOGS" "$DUMPS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

LAM=${1:-240000.0}

cell () {
  local mode=$1 lam=$2 seed=$3
  local name="mufz_${mode}_lam${lam}_s${seed}"
  job_slot
  echo "[launch] $name"
  WOE_LC_DEBUG=1 WOE_OMEGA_DUMP="$DUMPS/$name" PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar i2 \
    --woe_lambda "$lam" \
    --woe_mu_mode "$mode" \
    --single-seed --seed "$seed" --save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}

for seed in 39 55; do
  cell ema "$LAM" "$seed"
  cell frozen_pretask "$LAM" "$seed"
done
wait
echo "[mu-frozen seeds] done"
bash "$REPO/scripts/summarise_mu_frozen.sh"
