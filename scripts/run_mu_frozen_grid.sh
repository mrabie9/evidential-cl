#!/bin/bash
# PR-3, E1 + E2: the lambda curve under each mu mode, seed 0.
#
# Same host, same grid, same seed -- the only difference between the arms is
# which mu `compute_weights_of_evidence` reads. The pre-pass runs in *both*
# arms (it is RNG- and BatchNorm-neutral, tests/test_mu_frozen_pretask.py), so
# `||mu_ema - mu_frozen||` is measurable on the `ema` arm too.
#
# Grid: x2 spacing, one decade, centred on the confirmed i2 peak of 2.4e5.
# 1.2e5 is included because it is the cell whose seed-0 reading (0.5036) once
# looked like a better peak and did not replicate (README.md:299-317); it is
# also the widest-variance cell on this host, so it is the one most likely to
# move for reasons that are not the intervention.
#
# Lambda is NOT assumed to transfer between arms: freezing mu changes the
# magnitude of the mu-dependent half of I_2, hence Omega's scale. This project
# has five recorded instances of an Omega ratio mispredicting the optimal
# lambda, so both arms are bracketed on both sides and each arm's peak is read
# from its own curve.
#
# Gate: PR-3 Amendment 1. The original bit-identity gate was replaced after the
# pre-pass was measured to shift the trajectory deterministically by ~0.8 sigma.
# Both arms now run the pre-pass so the cost is common-mode, and the final cell
# below repeats one grid point ALONE (nothing else on the GPU) to test whether
# the trajectory depends on job concurrency -- the suspected mechanism is
# allocator state, which does. If the solo repeat differs from its grid twin,
# every cell must be re-run at a pinned concurrency and D' is not readable until
# then.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/mu_frozen
DUMPS=$REPO/scripts/logs/mu_frozen/omega_dumps
mkdir -p "$LOGS" "$DUMPS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

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

for lam in 60000.0 120000.0 240000.0 480000.0 960000.0; do
  cell ema "$lam" 0
  cell frozen_pretask "$lam" 0
done
wait
echo "[mu-frozen grid] done"

# Concurrency check (Amendment 1, clause 1): the same cell, run alone.
echo "[launch] concurrency check: mufz_ema_lam240000.0_s0 repeated solo"
WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
  --config configs/base.yaml \
  --config configs/models/til/woe_si_lc.yaml \
  --n_epochs 1 --inner_steps 2 --lr 0.003 \
  --woe_omega_transform abs --woe_anchor_mode proximal \
  --woe_omega_accum sum --woe_importance_scalar i2 \
  --woe_lambda 240000.0 --woe_mu_mode ema \
  --single-seed --seed 0 --save_checkpoints \
  --expt_name mufz_concurrency_solo_s0 > "$LOGS/mufz_concurrency_solo_s0.log" 2>&1

echo
echo "--- concurrency check: solo vs grid (must be identical) ---"
grep -E "^\[LC\] split" "$LOGS/mufz_concurrency_solo_s0.log" | sed 's/ *$//' > /tmp/solo.fp
grep -E "^\[LC\] split" "$LOGS/mufz_ema_lam240000.0_s0.log" | sed 's/ *$//' > /tmp/grid.fp
if diff -q /tmp/solo.fp /tmp/grid.fp > /dev/null; then
  echo "IDENTICAL -- trajectory does not depend on concurrency"
else
  echo "DIFFERS -- trajectory depends on concurrency; pin MAXJOBS and re-run"
  diff /tmp/solo.fp /tmp/grid.fp | head -6
fi

bash "$REPO/scripts/summarise_mu_frozen.sh"
