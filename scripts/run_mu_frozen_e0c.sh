#!/bin/bash
# PR-3, E0 round 2. The three-way diagnosis established:
#   A == B  bit-identically  -> the run IS deterministic run-to-run
#   A == C  bit-identically  -> checkpoint writing is inert
#   A != E0                  -> the pre-pass was moving the trajectory
# so the pre-pass now saves and restores CPU+CUDA RNG state.
#
#   D  ema, pre-pass off, --no-save_checkpoints: the recorded command line.
#      Does today's tree still reproduce 0.5206 / 0.5008 / -0.0198?
#   E  ema, pre-pass ON via WOE_MU_DIAG=1. Must equal D exactly, which is what
#      licenses reading `frozen_pretask` as differing from `ema` in the mu read
#      alone rather than in the mu read plus a trajectory perturbation.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/mu_frozen
mkdir -p "$LOGS"

run () {
  local name=$1 diag=$2
  WOE_MU_DIAG="$diag" WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar i2 \
    --woe_lambda 240000.0 \
    --woe_mu_mode ema \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
}

run mufz_e0d_ema_noprepass_s0 0
run mufz_e0e_ema_prepass_s0   1
wait
echo "[E0 round 2] done"
