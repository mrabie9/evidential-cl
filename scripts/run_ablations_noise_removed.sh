#!/bin/bash
# Leave-one-out ablation grids (TIL + CIL) re-measured after the 2026-09-17 La-MAML merge
# (noise class and detection metrics removed, new deeprad/radchar/radnist/uclresm task set,
# upstream re-tuned configs, shared lr 0.03). Every pre-merge ablation pool is on the old
# data and metric, so none of it is comparable; this grid replaces run_me_ablations.sh.
#
# Protocol is run_me_ablations.sh with these changes:
#
#   * Two regimes, same rows. N_EPOCHS=5 (default, tag "nrm") matches the post-merge main
#     experiments (La-MAML logs/full_experiments/run_20260917_182813_*); N_EPOCHS=1 with
#     TAG=nrm1e is the single-pass replicate. inner_steps is always base.yaml's 1 except on
#     B5a/B5b, where the optimizer budget IS the mechanism under test.
#   * AMP ON for every row (--amp), including CTN whose main config says amp: false.
#   * Configs come from the frozen snapshot configs/models/ablations_noise_removed/<mode>/.
#     The baselines are verbatim copies of the re-tuned configs; each twin changes exactly
#     one key. No per-row --lr overrides: every row runs at the shared base lr.
#   * Seeds 7,13,21 then 1,2,3 (n=6), the SAME protocol in both modes, topping up to n=9
#     with a third triplet only for rows whose verdict comes out underpowered. 0, 39 and 55
#     are the main-experiment seeds and are not used here.
#   * C-MAML is anchored on the FIRST-ORDER method -- the variant the La-MAML paper
#     recommends and the one configs/models/*/cmaml.yaml ships, so the family baseline is
#     the C-MAML the main experiments run. Second order is then an ADDED mechanism, not a
#     removed one. run_me_ablations.sh anchored on second order instead (and ran M3/M4
#     first-order, so its M3/M4 deltas mixed two changes).
#   * --eralg4_joint_er / --cmaml_joint_er are gone: joint ER is the only path post-merge.
#   * ROWS THAT REMOVE A MEMORY ARE NOT RUN: G2 (gem_noqp, projection and replay both off),
#     M2 (cmaml_no_replay), B3 (bcl_nodualmem, drops the validation buffer), B4
#     (bcl_noreplay) and T3 (ctn_noreplay). B3 ran at seeds 7,13,21 before this rule and
#     that pool stays on disk, off-table. G1 reads the E1 pool, as before.
#   * B2 (bilevel interpolation off, beta -> 1.0) is back in BOTH modes. It is a real
#     single-knob row now that the re-tuned CIL baseline runs beta 0.9; it would have been
#     degenerate against the old beta-1.0 CIL baseline.
#   * BCL-Dual baselines were re-tuned upstream in commit 7d1028c7 (TIL memory_strength
#     100 -> 1; CIL 1 -> 10 and beta 1.0 -> 0.9). memory_strength is BCL's KL-distillation
#     weight and beta is the Reptile coefficient, so both the baseline and the B1/B2 rows
#     they anchor moved: every BCL row was rerun from scratch and the runs made with the
#     old values are quarantined in logs/ablations_noise_removed_stale_bclcfg/.
#   * GEM-BOB add-one study on the two mechanisms of interest only -- A1 dynamic ring, A2
#     KL distillation, C1 both -- against A0, whose gates are all off and which must
#     reproduce G0. The meta-batch (A5) and bilevel (A3) arms are deliberately held back.
#
# M-FAMILY NAMING. An expt_name always denotes exactly one configuration -- pools are
# never re-pointed -- so the first grid's two C-MAML pools keep their launch names while
# the table reads them in their corrected roles:
#
#     table row              meaning                          pool (expt_name)
#     M0  baseline           C-MAML, first order               m1_nrm_<mode>
#     M1  + second order     second-order term ADDED (+/-)     m0_nrm_<mode>
#     M3  meta-batch 3 -> 1  first order                       m3fo_nrm_<mode>
#     M4  alpha -> 0         first order                       m4fo_nrm_<mode>
#
# Driver row ids m0/m1 therefore keep launching the same configs as before (m0 = second
# order, m1 = first order); only M3/M4 changed, and they run under NEW ids so that no
# expt_name ever covers two configurations. scripts/analyse_ablations_noise_removed.py
# holds the same mapping. The superseded second-order m3_nrm_til / m4_nrm_til pools
# (seeds 7,13,21) stay on disk as off-table data.
#
# Output, two trees. The RUN logs (what the analyser reads) land in
# logs/ablations_noise_removed/<mode>/<stem>/<timestamp>_<row>_<tag>_<mode>/<seed>/
# (a separate --log_dir so these pools can never mix with pre-merge runs of the same stem).
#
# The DRIVER logs -- one stdout/stderr capture per launched job -- land in
# scripts/logs/ablations_noise_removed/<n_epochs>e_<mode>/<row_id>/<seeds>.log, e.g.
# 5e_til/t0/7-13-21.log and 1e_cil/b5b/4-5-6.log. Regime and mode key the top level, the
# row id the second, so one row's whole seed history is a single `ls` and a grid sweep is
# `grep -r 5e_til/`. The path fully qualifies the job, hence the bare seed filename.
# Re-running a row at seeds it already has overwrites that file, as it always has.
#
# Jobs are (row x seed-triplet), group-major, so the whole grid reaches n=3 first.
#
# Usage:
#   bash scripts/run_ablations_noise_removed.sh                      # 5-epoch, n=6
#   MAX_PARALLEL=2 bash scripts/run_ablations_noise_removed.sh
#   MODES=cil ROWS_ONLY=b5a,b5b bash scripts/run_ablations_noise_removed.sh
#   N_EPOCHS=1 TAG=nrm1e bash scripts/run_ablations_noise_removed.sh  # single-pass replicate
#   SEED_GROUP_SPEC="4,5,6" ROWS_ONLY=... bash scripts/...            # n=9 top-up
#   SMOKE=1 bash scripts/run_ablations_noise_removed.sh               # 2 tasks, 512 samples
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py resolves data_path and logs/ against cwd
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/ablations_noise_removed"

MAX_PARALLEL="${MAX_PARALLEL:-3}"
SMOKE="${SMOKE:-0}"

# Regime. N_EPOCHS 5 (the main experiments' multi-epoch point) is the default and its runs
# carry the bare "nrm" tag; the single-pass replicate sets N_EPOCHS=1 TAG=nrm1e, so the two
# regimes pool separately and an expt_name still denotes exactly one configuration.
# inner_steps always comes from base.yaml (1) -- the only rows that set it are B5a/B5b,
# where the optimizer budget IS the mechanism under test.
N_EPOCHS="${N_EPOCHS:-5}"
TAG="${TAG:-nrm}"
DRIVER_LOGDIR="${DRIVER_LOGDIR:-$REPO/scripts/logs/ablations_noise_removed}"
RUN_LOGROOT="logs/ablations_noise_removed"

# id|mode|config_stem|extra_cli_args
ROWS=(
  # ---- TIL: GEM (G1 = E1 pool; G2 is a no-memory row, skipped) ----
  "g0|til|gem|"
  # ---- TIL: Res-ER ----
  "e0|til|eralg4|"
  "e1|til|er_ring|"
  "e2|til|er_ring|--er_dynamic_ring"
  # ---- TIL: C-MAML (M2 skipped; see the M-FAMILY NAMING note above) ----
  "m0|til|cmaml|--second_order"
  "m1|til|cmaml|"
  "m3fo|til|cmaml_single_inner|"
  "m4fo|til|cmaml_alpha0|"
  # ---- TIL: BCL-Dual (B3, B4 skipped: both remove a memory) ----
  "b0|til|bcl_dual|"
  "b1|til|bcl_nodistill|"
  "b2|til|bcl_nobilevel|"
  "b5a|til|bcl_singlelevel|--inner_steps 1"
  "b5b|til|bcl_singlelevel|--inner_steps 2"
  # ---- TIL: CTN (T3 skipped) ----
  "t0|til|ctn|"
  "t1|til|ctn_nofilm|"
  "t2|til|ctn_nodistill|"
  # ---- TIL: GEM-BOB add-one study (dynamic ring + KL distillation only) ----
  "a0|til|gem_bob|"
  "a1|til|gem_bob_dynring|"
  "a2|til|gem_bob_distill|"
  "c1|til|gem_bob_c1|"
  # ---- CIL: Res-ER ----
  "e0|cil|eralg4|"
  "e1|cil|er_ring|"
  "e2|cil|er_ring|--er_dynamic_ring"
  # ---- CIL: C-MAML (M2 skipped; see the M-FAMILY NAMING note above) ----
  "m0|cil|cmaml|--second_order"
  "m1|cil|cmaml|"
  "m3fo|cil|cmaml_single_inner|"
  "m4fo|cil|cmaml_alpha0|"
  # ---- CIL: combination row -- Res-ER + KL distillation, the CIL counterpart of TIL C1 ----
  "c1|cil|eralg4_distill|"
  "c1b|cil|eralg4_distill_ms05|"
  # ---- CIL: BCL-Dual (B3, B4 skipped: both remove a memory) ----
  "b0|cil|bcl_dual|"
  "b1|cil|bcl_nodistill|"
  "b2|cil|bcl_nobilevel|"
  "b5a|cil|bcl_singlelevel|--inner_steps 1"
  "b5b|cil|bcl_singlelevel|--inner_steps 2"
)

SEED_GROUP_SPEC="${SEED_GROUP_SPEC:-7,13,21;1,2,3}"
ROWS_ONLY="${ROWS_ONLY:-}"
MODES="${MODES:-til,cil}"

if [ "$SMOKE" = "1" ]; then
  SEED_GROUP_SPEC="7"
  DRIVER_LOGDIR="$REPO/scripts/logs/ablations_noise_removed_smoke"
  RUN_LOGROOT="logs/ablations_noise_removed_smoke"
fi
mkdir -p "$DRIVER_LOGDIR"

IFS=';' read -r -a SEED_GROUPS <<< "$SEED_GROUP_SPEC"

# run_job <mode> <config_stem> <row_id> <expt_name> <seeds> <extra_cli_args>
run_job() {
  local mode="$1" stem="$2" rid="$3" expt="$4" seeds="$5" extra="$6"
  local outdir="$DRIVER_LOGDIR/${N_EPOCHS}e_${mode}/${rid}"
  mkdir -p "$outdir"
  local out="$outdir/${seeds//,/-}.log"
  local t0=$SECONDS
  local smoke_args=()
  if [ "$SMOKE" = "1" ]; then
    smoke_args=(--task-order-files "deeprad_task_01,radchar_task_01" --samples_per_task 512)
  fi
  echo "[$TAG] START $expt seeds=$seeds (cfg=$mode/$stem extra='$extra') $(date '+%F %T')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/${mode}/${stem}.yaml" \
    --n_epochs "$N_EPOCHS" --amp --no-save_checkpoints --log_dir "$RUN_LOGROOT/$mode" \
    --expt_name "$expt" --seeds "$seeds" $extra "${smoke_args[@]}" > "$out" 2>&1
  local rc=$?
  local dt=$(( SECONDS - t0 ))
  local flag=""
  if [ "$rc" -ne 0 ] || [ "$dt" -lt 60 ]; then flag=" <<< CHECK (rc=$rc, ${dt}s)"; fi
  echo "[$TAG] END   $expt seeds=$seeds exit=$rc ${dt}s $(date '+%F %T')$flag"
}

selected=()
for row in "${ROWS[@]}"; do
  IFS='|' read -r rid rmode _stem _extra <<< "$row"
  if [ -n "$ROWS_ONLY" ] && [[ ",$ROWS_ONLY," != *",$rid,"* ]]; then continue; fi
  if [[ ",$MODES," != *",$rmode,"* ]]; then continue; fi
  selected+=("$row")
done
if [ "${#selected[@]}" -eq 0 ]; then
  echo "[$TAG] ROWS_ONLY='$ROWS_ONLY' MODES='$MODES' matched no rows" >&2
  exit 1
fi

echo "[$TAG] ===== START $(date) regime=${N_EPOCHS}ep/is-from-config/amp smoke=$SMOKE"
echo "[$TAG]       rows=${#selected[@]} groups=${#SEED_GROUPS[@]} jobs=$(( ${#selected[@]} * ${#SEED_GROUPS[@]} )) MAX_PARALLEL=$MAX_PARALLEL"
echo "[$TAG]       seed groups: ${SEED_GROUPS[*]}"

running=0
for group in "${SEED_GROUPS[@]}"; do
  for row in "${selected[@]}"; do
    IFS='|' read -r rid rmode stem extra <<< "$row"
    expt="${rid}_${TAG}_${rmode}"
    run_job "$rmode" "$stem" "$rid" "$expt" "$group" "$extra" &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait
echo "[$TAG] ===== ALL DONE $(date) ====="
