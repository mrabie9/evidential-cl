#!/bin/bash
# Extra-seed sweep for the leave-one-out ablation grid (docs/ablation_studies.tex,
# Table tab:ablation_optim ONLY -- excludes config-matched and cmaml_highlr tables).
#
# Purpose: add seeds 7,13,21 to every row so each ablation reaches n=6 seeds, the
# threshold at which a paired sign-flip permutation test can first reach p<0.05.
# Single-pass TIL, n_epochs 1, inner_steps 2 (=> 4 SGD steps/batch for inner/outer
# methods, 2 for plain replay).
#
# EFFICIENCY -- bounded worker pool:
#   Instead of running families one-after-another, all jobs go through a pool of
#   MAX_PARALLEL concurrent processes (default 3). The job list is INTERLEAVED across
#   families (round-robin), so at any instant the running set mixes a heavy job
#   (CTN, full precision) with light ones (GEM's QP is CPU-bound; C-MAML is
#   python-overhead-bound; Res-ER is tiny). That recovers the GPU idle time the old
#   sequential 2-lane layout wasted -- ~19.6 GPU-h of work compresses to ~7-8 h
#   wall-clock at MAX_PARALLEL=3 (contention-dependent).
#
#   The GPU here is 49 GB / ~0% idle and each resnet1d process needs ~1-2 GB, so
#   memory is not the limit -- raise MAX_PARALLEL (e.g. 4-5) to trade GPU contention
#   for shorter wall-clock; watch `nvidia-smi` and per-job runtimes to find the knee.
#
# Each job still runs its 3 seeds internally (--seeds 7,13,21) so every config gets
# ONE clean run dir with a cross-seed summary:
#   logs/<config_stem>/<expt_name>-<timestamp>/{7,13,21}/results.txt
# To test at n=6, pool these with the original 0,39,55 runs.
#
# Usage:  bash scripts/run_ablation_extra_seeds.sh
#         MAX_PARALLEL=4 bash scripts/run_ablation_extra_seeds.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes its logs/ dir relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"

SEEDS="7,13,21"
IS=2                 # inner_steps for all rows except B5b (matched-budget single-level)
ER_LR="0.01"         # Res-ER (E0) / Ring-ER-dynamic (E1) matched learning rate.
                     # NOTE: this is the lr=0.01 config-matched comparison point
                     # (Ring-ER 59.6 / Res-ER 61.7), NOT the tuned E0=63.2 in the
                     # current table. Set to the tuned lr if you intend that row.
MAX_PARALLEL="${MAX_PARALLEL:-3}"   # concurrent GPU processes (override from env)

LOGDIR="$REPO/scripts/logs/ablation_extra_7-13-21"
mkdir -p "$LOGDIR"

# run_job <config_stem> <expt_name> <inner_steps> <extra_cli_args>
run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4"
  echo "[queue] START $expt  (cfg=$stem is=$isteps extra='$extra')  $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" \
    --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" \
    --expt_name "$expt" \
    --seeds "$SEEDS" $extra > "$LOGDIR/${expt}.log" 2>&1
  echo "[queue] END   $expt  exit=$?  $(date '+%H:%M:%S')"
}

# ---------------------------------------------------------------------------
# Job list: "stem|expt|inner_steps|extra".  INTERLEAVED across families
# (GEM / C-MAML / CTN / BCL-Dual / Res-ER) so the pool always mixes heavy and
# light work. Grid row IDs from docs/ablation_studies.tex in comments.
# ---------------------------------------------------------------------------
JOBS=(
  # round 1: one baseline per family
  "gem|gem_full_s7-13-21|$IS|"                                                              # G0
  "cmaml|cmaml_full_s7-13-21|$IS|"                                                          # M0
  "ctn|ctn_full_s7-13-21|$IS|--no-amp"                                                      # T0 (--no-amp: ctn.yaml no_amp key is silently dropped)
  "bcl_dual|bcl_dual_s7-13-21|$IS|"                                                         # B0
  "eralg4|eralg4_resER_s7-13-21|$IS|--lr $ER_LR"                                            # E0
  # round 2
  "gem_noqp|gem_noqp_s7-13-21|$IS|"                                                         # G1
  "cmaml|cmaml_secondorder_s7-13-21|$IS|--second_order"                                     # M1 (turns SO on)
  "ctn_nofilm|ctn_nofilm_s7-13-21|$IS|--no-amp"                                             # T1
  "bcl_nodistill|bcl_nodistill_s7-13-21|$IS|"                                               # B1
  "er_ring|er_ring_dynring_s7-13-21|$IS|--lr $ER_LR --memory_loss_lambda 1 --er_dynamic_ring" # E1
  # round 3
  "gem|gem_noreplay_s7-13-21|$IS|--n_memories 0"                                            # G2 (empty buffer -> plain SGD)
  "cmaml_no_replay|cmaml_noreplay_s7-13-21|$IS|"                                            # M2
  "ctn_nodistill|ctn_nodistill_s7-13-21|$IS|--no-amp"                                       # T2
  "bcl_nobilevel|bcl_nobilevel_s7-13-21|$IS|"                                               # B2 (beta=1 identity)
  # round 4
  "cmaml_single_inner|cmaml_singleinner_s7-13-21|$IS|"                                      # M3
  "ctn_noreplay|ctn_noreplay_s7-13-21|$IS|--no-amp"                                         # T3
  "bcl_nodualmem|bcl_nodualmem_s7-13-21|$IS|"                                               # B3
  # round 5
  "cmaml_alpha0|cmaml_alpha0_s7-13-21|$IS|"                                                 # M4
  "bcl_noreplay|bcl_noreplay_s7-13-21|$IS|"                                                 # B4
  # tail: single-level BCL (own inner_steps)
  "bcl_singlelevel|bcl_singlelevel_is2_s7-13-21|2|"                                         # B5a (single loop, 2 steps)
  "bcl_singlelevel|bcl_singlelevel_is4_s7-13-21|4|"                                         # B5b (single loop, budget-matched)
)

echo "[queue] ===== ablation extra-seed sweep  seeds=$SEEDS  MAX_PARALLEL=$MAX_PARALLEL  jobs=${#JOBS[@]}  START $(date) ====="

running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r j_stem j_expt j_is j_extra <<< "$spec"
  run_job "$j_stem" "$j_expt" "$j_is" "$j_extra" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n            # free a slot as soon as ANY running job finishes
    running=$((running - 1))
  fi
done
wait                   # drain the remaining in-flight jobs

echo "[queue] ===== ALL DONE  $(date) ====="
echo "[queue] per-job logs: $LOGDIR/*.log"
echo "[queue] results:      logs/<config_stem>/<expt_name>-<ts>/{7,13,21}/results.txt"
