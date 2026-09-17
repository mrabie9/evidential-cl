#!/bin/bash
# Reduction runner, part 2: the jobs killed at ~08:00 on 2026-07-24 plus one run the ladder was
# missing (the COMBINATION lr.001+is4+distill on the static ring -- singles were near-inert, so
# the ring rescue must be the joint effect). No trigger (nothing else running). 3-wide pool.
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
MODELS="$REPO/configs/models"

SEEDS_DEFAULT="0,39,55,7,13,21,1,2,3"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_reductions}"
mkdir -p "$LOGDIR"

# run_job <stem> <expt> <inner_steps> <extra> <seeds>
run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4" seeds="$5"
  echo "[reduce2] START $expt  (cfg=$stem is=$isteps extra='$extra' seeds=$seeds)  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$MODELS/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" --no-save_checkpoints \
    --expt_name "$expt" \
    --seeds "$seeds" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[reduce2] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

# "stem|expt|is|extra|seeds"  (seeds empty => SEEDS_DEFAULT)
JOBS=(
  # ---- CIL: finish group 1 (the missing COMBINATION) + Z2 seed-3 + reservoir reduction ----
  "cil/er_ring|er_ring_static_combo_se_cil|4|--lr 0.001 --memory_loss_lambda 1 --er_distill|"   # lr+budget+distill together
  "cil/bcl_b3_gres|bcl_b3_gres_b1_se_cil|2||3"                                                   # Z2 fixup: only seed 3
  "cil/eralg4|eralg4_resER_reduce_se_cil|4|--lr 0.001 --eralg4_joint_er --er_distill|"           # reduction: reservoir @ B3+gres point
  # ---- TIL C-MAML ----
  "til/er_ring|er_ring_lr01_se_til|2|--lr 0.01 --memory_loss_lambda 1|"                          # T-M1
  "til/eralg4|eralg4_unmasked_se_til|2|--lr 0.01 --eralg4_unmasked_loss|"                        # T-M3
  # ---- TIL BCL ----
  "til/er_ring|er_ring_lr001_is4_se_til|4|--lr 0.001 --memory_loss_lambda 1|"                    # T-B1
  "til/er_ring|er_ring_lr001_is4_distill_se_til|4|--lr 0.001 --memory_loss_lambda 1 --er_distill|" # T-B2
)

echo "[reduce2] ===== part-2 grid  MAX_PARALLEL=$MAX_PARALLEL  jobs=${#JOBS[@]}  START $(date) ====="
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r j_stem j_expt j_is j_extra j_seeds <<< "$spec"
  [ -z "$j_seeds" ] && j_seeds="$SEEDS_DEFAULT"
  run_job "$j_stem" "$j_expt" "$j_is" "$j_extra" "$j_seeds" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then wait -n; running=$((running - 1)); fi
done
wait
echo "[reduce2] ===== ALL DONE  $(date) ====="
