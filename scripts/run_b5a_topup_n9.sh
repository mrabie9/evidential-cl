#!/bin/bash
# Top the B5a-equivalence arms from n=6 to n=9 (adds seeds 1,2,3).
#
# WHY: at n=6 the equivalence held tightly (B5a - RingER+distill = -0.19, p 0.562, matching
# on F1, Diag and BWT alike) but the CONTROL was ambiguous: distill - no_distill = +0.57
# +-0.67 (p 0.156). So the data cannot yet distinguish "B5a equals Ring-ER + distillation"
# from the weaker "B5a equals Ring-ER, and distillation does nothing here". Resolving that
# needs the control separated, and this repo's standing rule is not to call a paired effect
# below n=9.
#
# Three arms topped in parallel. B5a itself must be topped too: the equivalence contrast
# pairs on shared seeds, so it is capped by B5a's n=6 pool no matter how many er_ring seeds
# exist. New B5a runs use expt b5a_topup_se_til and are pooled with the existing
# logs/ablations/til/bcl-dual/B5a dirs by analyse_b5a_reduces_to_ering.py.
#
# Configs are unchanged from the n=6 runs: bcl_singlelevel.yaml at inner_steps 2 for B5a,
# er_ring.yaml at lr 0.001 / is2 / memory_strength 1 / temperature 5.0 for the two Ring-ER
# arms, bf16 throughout.
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"
SEEDS="${SEEDS:-1,2,3}"
LOGDIR="$REPO/scripts/logs/b5a_topup_n9"
mkdir -p "$LOGDIR"

echo "[b5a9] ===== START $(date) seeds=$SEEDS ====="

( echo "[b5a9] START b5a $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/bcl_singlelevel.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --expt_name b5a_topup_se_til --seeds "$SEEDS" \
    > "$LOGDIR/b5a.log" 2>&1
  echo "[b5a9] END   b5a exit=$? $(date '+%H:%M:%S')" ) &

( echo "[b5a9] START ering_distill $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/er_ring.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --lr 0.001 --memory_loss_lambda 1 --memory_strength 1 --temperature 5.0 --er_distill \
    --expt_name ering_distill_b5amatch_se_til --seeds "$SEEDS" \
    > "$LOGDIR/ering_distill.log" 2>&1
  echo "[b5a9] END   ering_distill exit=$? $(date '+%H:%M:%S')" ) &

( echo "[b5a9] START ering_nodistill $(date '+%H:%M:%S')"
  "$PY" "$REPO/main.py" \
    --config "$REPO/configs/base.yaml" \
    --config "$REPO/configs/models/til/er_ring.yaml" \
    --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
    --lr 0.001 --memory_loss_lambda 1 --memory_strength 1 --temperature 5.0 \
    --expt_name ering_nodistill_b5amatch_se_til --seeds "$SEEDS" \
    > "$LOGDIR/ering_nodistill.log" 2>&1
  echo "[b5a9] END   ering_nodistill exit=$? $(date '+%H:%M:%S')" ) &

wait
echo "[b5a9] ===== ALL DONE $(date) ====="
