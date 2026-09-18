#!/bin/bash
# Re-run the i2 anchor-only control on the CURRENT tree.
#
# Why: the recorded control (0.5008 +/- 0.0031, seeds 0/39/55) was measured
# 2026-08-18..20 and its logs carry 175 config keys; runs on today's tree carry
# 188. The parser demonstrably changed (woe_mu_mode, woe_lc_tau, eucr_*,
# woe_teacher_dropout, anchor_omega_uniform), and model/woe_si.py has 1913
# uncommitted added lines since its last commit (d3f6a218, 2026-08-13). So every
# paired delta computed against that control -- B6's halves, and the
# displacement arm -- compares across two code versions.
#
# If this reproduces 0.5008/0.5039/0.4977 the comparisons stand. If it does not,
# they are void and the control must be re-measured before anything is claimed.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/i2_recheck
mkdir -p "$LOGS"
# shellcheck source=/dev/null
. "$REPO/scripts/lib_joblimit.sh"

for seed in 0 39 55; do
  name="i2recheck_lam240000_s${seed}"
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
echo "[i2 recheck] done"
