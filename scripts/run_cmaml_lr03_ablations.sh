#!/bin/bash
# CMAML MAML-mechanism ablations at the HIGHER lr (opt_wt 0.03), single-epoch TIL, is2,
# seeds 0/39/55. Motivation: in the budget-match, plain ER (Ring-ER/Res-ER) caught and
# passed CMAML once lr + replay weight were equalized, so at the common operating point
# CMAML's meta machinery buys ~nothing. The v2 doc showed the same at CMAML's native
# opt_wt 0.01 (second-order, inner-loop adaptation, multi-step all ~nil; replay is the
# engine). This round re-tests each mechanism at opt_wt 0.03 -- the higher lr where CMAML
# *looked* strongest in the lr=0.03 table -- to see if any MAML mechanism earns its keep
# there. Read each row against the baseline (delta = row - baseline). expt_name suffix
# `_cmaml_lr03abl_se_til`; compare to the v2-doc C-rows (same ablations at opt_wt 0.01).
set -uo pipefail
REPO="/home/lunet/wsmr11/repos/evidential-cl"
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

run() {
  local cfg="$1"; local name="$2"; shift 2
  echo "[cmaml-lr03abl] ===== START $name (cfg=$cfg opt_wt=0.03) ====="
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/$cfg.yaml" \
    --opt_wt 0.03 --n_epochs 1 --inner_steps 2 \
    --expt_name "$name" \
    --seeds "0,39,55" "$@"
  echo "[cmaml-lr03abl] ===== END $name exit=$? ====="
}

# Baseline (second_order:false, alpha 1e-4) at the higher lr.
run cmaml              baseline_cmaml_lr03abl_se_til
# + second-order meta-gradient (the C1 contrast; --second_order is a store_true flag).
run cmaml              so_cmaml_lr03abl_se_til              --second_order
# Zero inner LR: severs MAML inner-loop adaptation (reduces to plain replay SGD via opt_wt).
run cmaml_alpha0       alpha0_cmaml_lr03abl_se_til
# Single inner meta-batch: collapses the multi-step inner loop.
run cmaml_single_inner singleinner_cmaml_lr03abl_se_til
# No replay anchor (memories:0): confirms replay is still the engine at the higher lr.
run cmaml_no_replay    noreplay_cmaml_lr03abl_se_til

echo "[cmaml-lr03abl] ALL DONE"
echo "[cmaml-lr03abl] results: logs/lamaml_cifar/<expt_name>-*/{0,39,55}/results.txt"
