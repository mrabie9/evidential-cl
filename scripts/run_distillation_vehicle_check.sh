#!/bin/bash
# Is distillation a working delivery vehicle on this host?
#
# C6 read the null between `logit` (distil z) and `evidence_sym` (distil w) as
# evidence about what a replay buffer should carry. That inference needs
# distillation to *work* here: both arms recovered only ~19% of the gap that CE
# rehearsal closes (0.32-0.34 against 0.4466, over a naive bar of 0.3030), and a
# null between two targets inside a vehicle that barely delivers cannot
# discriminate the targets. This is the load-bearing repair for C6's Claim 1.
#
# Two things were missing. (1) There was no DER++ analogue -- the `logit` mode
# *replaces* CE rehearsal rather than adding to it, so the arms were DER, not
# DER++, and DER without the CE term is the configuration known to underperform.
# `ce_logit` / `ce_evidence_sym` add it. (2) The lambda grid was three cells over
# two decades; the ends are extended to check the flat response is not a
# saturating term, which is the alternative reading of "flat in lambda".
#
# Note the pre/post-augmentation concern does not apply: this loader is raw
# radar-IQ npz arrays with a fixed 'normalize' scaling and no train-time
# augmentation, so a stored snapshot and its later re-encode see identical input.
#
# Read against, on the identical host: naive 0.3030, `ce` rehearsal 0.4466,
# `logit`/`evidence_sym` 0.3236-0.3358.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/vehicle_check
mkdir -p "$LOGS"

launch () {
  local name=$1 mode=$2 lam=$3
  echo "[launch] $name"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_injection.yaml \
    --n_epochs 1 --inner_steps 2 \
    --woe_replay_memories 256 \
    --woe_replay_mode "$mode" \
    --woe_evidence_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1
}

launch veh_celogit_lam0.1  ce_logit        0.1 &
launch veh_celogit_lam1    ce_logit        1.0 &
launch veh_celogit_lam10   ce_logit       10.0 &
launch veh_cesym_lam0.1    ce_evidence_sym 0.1 &
launch veh_cesym_lam1      ce_evidence_sym 1.0 &
launch veh_cesym_lam10     ce_evidence_sym 10.0 &
wait

launch veh_logit_lam0.01   logit           0.01 &
launch veh_logit_lam100    logit         100.0 &
launch veh_sym_lam100      evidence_sym  100.0 &
wait
echo "[vehicle check] done"
