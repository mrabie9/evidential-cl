#!/bin/bash
# Both review gates, one detached driver with a concurrency cap.
#
# A7 gate  -- commensurability. Denoeux's cautious rule combines weights of
#   evidence on a common scale; the raw per-task path integrals are not on one
#   (task 0 starts from random init and takes the whole drop in the tracked
#   scalar; later tasks travel less under a progressively stronger anchor, so
#   the rule feeds back into what it will combine). `max` keeps only the largest
#   term and is maximally exposed to that, `sum` partially launders it. `*_norm`
#   scales each task's omega to unit mass before combining and rescales the
#   result to the raw-sum total, so lambda transfers. The 2x2 {sum,max} x
#   {raw,norm} is required: normalised-max against raw-sum would confound the
#   rule with the normalisation.
#
# C6 gate  -- delivery vehicle. Both distillation arms recovered only ~19% of
#   the gap CE rehearsal closes, and a null between two targets inside a vehicle
#   that barely delivers cannot discriminate the targets. `ce_logit` /
#   `ce_evidence_sym` are the DER++ analogues (distillation *on top of* CE, not
#   instead of it); the grid ends are extended to separate "flat because the
#   target is irrelevant" from "flat because the term saturates".
#
# Detached via setsid so it survives the launching session.
#
# CONCURRENCY: keep MAXJOBS low. Steady-state RSS is ~16 GB but the transient
# during data loading is much higher (all ten npz task files are opened), and
# six concurrent runs exhausted memory on this box. Three is the tested cap.
# wait_slot also gates on *available* memory, so the driver backs off if
# something else on the machine takes it, and launches are staggered so the
# load spikes do not coincide.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/ds_gates
DUMPS=$REPO/scripts/logs/omega_dumps
mkdir -p "$LOGS" "$DUMPS"
MAXJOBS=3
MINFREEGB=60
STAGGER=45

wait_slot () {
  while true; do
    running=$(jobs -rp | wc -l)
    freegb=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "$running" -lt "$MAXJOBS" ] && [ "$freegb" -ge "$MINFREEGB" ]; then
      break
    fi
    sleep 20
  done
  sleep "$STAGGER"
}

anchor_run () {
  local name=$1 accum=$2 lam=$3 dump=${4:-}
  wait_slot
  echo "[launch] $name accum=$accum lambda=$lam"
  WOE_LC_DEBUG=1 WOE_OMEGA_DUMP="$dump" PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum "$accum" --woe_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
}

replay_run () {
  local name=$1 mode=$2 lam=$3
  wait_slot
  echo "[launch] $name mode=$mode lambda=$lam"
  PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_injection.yaml \
    --n_epochs 1 --inner_steps 2 \
    --woe_replay_memories 256 \
    --woe_replay_mode "$mode" \
    --woe_evidence_lambda "$lam" \
    --single-seed --seed 0 --no-save_checkpoints \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
}

anchor_run cn_sum_regression    sum       240000.0
anchor_run cn_maxnorm_lam60000  max_norm   60000.0
anchor_run cn_maxnorm_lam120000 max_norm  120000.0
anchor_run cn_maxnorm_lam240000 max_norm  240000.0
anchor_run cn_maxnorm_lam480000 max_norm  480000.0
anchor_run cn_maxnorm_lam960000 max_norm  960000.0
anchor_run cn_sumnorm_lam120000 sum_norm  120000.0
anchor_run cn_sumnorm_lam240000 sum_norm  240000.0
anchor_run cn_sumnorm_lam480000 sum_norm  480000.0
# Anchor OFF, with a dump: does task 0 still dominate the path integral when no
# penalty is shrinking later tasks' travel? Separates initialisation (reason 1)
# from anchor feedback (reason 2) as the source of the incommensurability.
anchor_run cn_lam0_dump         sum            0.0 "$DUMPS/lam0_s0"

replay_run veh_celogit_lam0.1   ce_logit         0.1
replay_run veh_celogit_lam1     ce_logit         1.0
replay_run veh_celogit_lam10    ce_logit        10.0
replay_run veh_cesym_lam0.1     ce_evidence_sym  0.1
replay_run veh_cesym_lam1       ce_evidence_sym  1.0
replay_run veh_cesym_lam10      ce_evidence_sym 10.0
replay_run veh_logit_lam0.01    logit            0.01
replay_run veh_logit_lam100     logit          100.0
replay_run veh_sym_lam100       evidence_sym   100.0
wait
echo "[ds gates] all done"
