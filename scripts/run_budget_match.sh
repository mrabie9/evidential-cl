#!/bin/bash
# Budget-match ablation (filed under ablations, NOT best-of-best): hold BOTH the learning
# rate (lr=0.03) AND the per-batch SGD-step budget (4 steps) equal across every family, so
# any surviving ranking is algorithmic, not an optimization-budget artifact. Each method
# keeps all its OTHER tuned params (memory_strength, beta, ...); only lr + step budget move.
#
# Step budget = 4 SGD steps/batch, per the doc's accounting (bcl_singlelevel is4 = B0):
#   * replay methods (GEM, Ring-ER, Res-ER) take 4 inner passes  -> --inner_steps 4
#   * meta methods (CMAML, BCL-Dual, CTN) do 2 rounds x 2 steps   -> --inner_steps 2
#
# AMP: on for all EXCEPT CTN. CTN's config bakes in no_amp:true (bf16 AMP corrupts CTN's
# retention; see v2 doc CTN section), so its row is NOT step-for-step comparable to the
# AMP-on rows -- footnote it in the table. Single-epoch TIL, seeds 0,39,55.
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

run() {
  local cfg="$1"; local isteps="$2"; local name="$3"; shift 3
  echo "[budget-match] ===== START $name (cfg=$cfg lr=0.03 inner_steps=$isteps) ====="
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/$cfg.yaml" \
    --lr 0.03 --n_epochs 1 --inner_steps "$isteps" \
    --expt_name "$name" \
    --seeds "0,39,55" "$@"
  echo "[budget-match] ===== END $name exit=$? ====="
}

# Replay family: 4 inner passes.
run gem      4 gem_bm_se_til
run er_ring  4 er_ring_bm_se_til
run eralg4   4 eralg4_bm_se_til

# Meta family: 2 rounds x 2 steps = 4 SGD steps.
# CMAML (lamaml_cifar) steps its WEIGHTS with opt_wt, not --lr (which is inert here), so
# the lr-match knob is --opt_wt 0.03. alpha_init (the meta-learned inner LR init) is left
# at its tuned value -- it is a learned quantity, not a fixed lr to match.
run cmaml    2 cmaml_bm_se_til --opt_wt 0.03
run bcl_dual 2 bcl_dual_bm_se_til

# CTN: 2 rounds, --no-amp. NOTE: ctn.yaml's `no_amp: true` key is INERT (the parser dest
# is `amp` via --amp/--no-amp; there is no `no_amp` dest, so the YAML key is silently
# dropped and amp stays on). AMP must be disabled on the CLI. Footnoted in the table.
run ctn      2 ctn_bm_se_noamp_til --no-amp

echo "[budget-match] ALL DONE"
echo "[budget-match] results: logs/<model>/<expt_name>-*/{0,39,55}/results.txt"
