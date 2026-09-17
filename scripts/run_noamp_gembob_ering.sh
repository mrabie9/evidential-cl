#!/bin/bash
# Re-run the gem_bob add-one study and the Ring-ER grid rows with --no-amp, lr 0.01.
#
# WHY: bf16 AMP is not method-neutral. Measured on full TIL: it costs GEM +1.43 F1
# (p 0.031, 6/6 seeds) and 1.50 diagonal, Res-ER 1.69 diagonal, C-MAML nothing.
# GEM --no-amp is 65.97 vs 64.55 published. Every gem_bob arm and every Ring-ER
# row was measured under AMP, so the whole study needs an AMP-free reading.
# See docs/cmaml_advantage_is_numerical.md.
#
# lr 0.01 throughout (the gem_bob config-matched operating point, already the
# default in every gem_bob yaml; passed explicitly to er_ring whose yaml says
# 0.001). No lr=0.03 arm -- dropped by request.
#
# Expt names carry a "_noamp" suffix so compare_gembob.py picks them up with its
# existing replica mechanism:
#     la-maml_env/bin/python scripts/compare_gembob.py --suffix _noamp
#
# n=6 seeds (0,39,55,7,13,21) per the standing protocol: top up to n=9 only for
# rows that come back inconclusive or borderline. The original study ran n=9, so
# expect wider intervals here until any top-up.
#
# inner_steps differs per arm and is preserved exactly from the original
# launchers: the bilevel arms (A3, C2, A1A3, A2A3) run is=1 (budget-matched),
# everything else is=2.
#
# NB G1 and E1 are the SAME configuration (er_ring, lr 0.01, is 2, static ring) --
# the only differing keys in their logged params are flags that did not exist when
# E1 ran. One run serves both grid rows, so er_ring appears once here, not twice.
# G0 (--no-amp, lr 0.01) already exists at n=6 in logs/gem/g0_noamp_lr01_se_til-*.
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
BASE="$REPO/configs/base.yaml"
CFG="$REPO/configs/models/til"
SEEDS="${SEEDS:-0,39,55,7,13,21}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/noamp_gembob_ering}"
mkdir -p "$LOGDIR"

# stem|expt_name|inner_steps|extra
JOBS=(
  # --- heaviest first so the pool drains evenly; Phase 2 combos are the slow ones
  "gem_bob_c3|gembob_c3_noamp|2|"                                  # C3  A1+A2+A5
  "gem_bob_c1|gembob_c1_noamp|2|"                                  # C1  A1+A2
  "gem_bob_a2a5|gembob_a2a5_noamp|2|"                              # A2+A5
  "gem_bob_dynring|gembob_a1_dynring_noamp|2|"                     # A1  ring
  "gem_bob|gembob_a0_noamp|2|"                                     # A0  baseline
  "gem_bob_meta|gembob_a5_meta_noamp|2|"                           # A5  meta-batch K=3
  # DROPPED 2026-07-27: A4 (bilevel at 2x budget) and A2' (distill lambda=0.1) are second
  # settings of mechanisms that already have a row (A3, A2). They are sensitivity checks, not
  # hypotheses, so they sit outside the Holm family and carry no verdict. Dropped to shorten
  # the batch. Re-enable if a verdict on A3 or A2 turns out to be dose/budget dependent.
  # "gem_bob_bilevel|gembob_a4_bilevel_unmatched_noamp|2|"         # A4  bilevel 2x budget (diagnostic)
  # "gem_bob_distill|gembob_a2_distill_noamp|2|--distill_lambda 0.1" # A2' distill lambda=0.1 (diagnostic)
  "gem_bob_a1a5|gembob_a1a5_noamp|2|"                              # A1+A5
  # NB the space before this trailing '#' is load-bearing. Without it bash does not treat the
  # '#' as a comment (it is not at the start of a word), so it fuses onto the quoted string as
  # '--distill_lambda 1#' and the comment text splits into extra junk array elements. That bug
  # silently killed this arm on 2026-07-27 -- argparse rejected '1#' and the run exited in 3s.
  "gem_bob_distill|gembob_a2_distill_l1_noamp|2|--distill_lambda 1" # A2  distill lambda=1
  "gem_bob_c2|gembob_c2_noamp|1|"                                  # C2  A1+A2+A3 (budget-matched)
  "gem_bob_a2a3|gembob_a2a3_noamp|1|"                              # A2+A3
  "gem_bob_bilevel|gembob_a3_bilevel_matched_noamp|1|"             # A3  bilevel budget-matched
  "gem_bob_a1a3|gembob_a1a3_noamp|1|"                              # A1+A3
  # --- Ring-ER grid rows (G1 == E1, one run; E2 = dynamic ring)
  "er_ring|ering_static_lr01_noamp_se_til|2|--lr 0.01 --memory_loss_lambda 1"                     # G1 / E1
  "er_ring|ering_dynring_lr01_noamp_se_til|2|--lr 0.01 --memory_loss_lambda 1 --er_dynamic_ring"  # E2
)

run_job() {
  local stem="$1" expt="$2" isteps="$3" extra="$4"
  [ "$expt" = "SKIP" ] && return 0
  echo "[noamp] START $expt (cfg=$stem is=$isteps extra='$extra') $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$BASE" --config "$CFG/${stem}.yaml" \
    --n_epochs 1 --inner_steps "$isteps" --no-save_checkpoints --no-amp \
    --expt_name "$expt" --seeds "$SEEDS" $extra \
    > "$LOGDIR/${expt}.log" 2>&1
  echo "[noamp] END   $expt exit=$? $(date '+%H:%M:%S')"
}

echo "[noamp] ===== START $(date) seeds=$SEEDS MAX_PARALLEL=$MAX_PARALLEL ====="
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r j_stem j_expt j_is j_extra <<< "$spec"
  [ "$j_expt" = "SKIP" ] && continue
  run_job "$j_stem" "$j_expt" "$j_is" "$j_extra" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "[noamp] ===== ALL DONE $(date) ====="
echo "[noamp] analyse: $PY scripts/compare_gembob.py --suffix _noamp"
