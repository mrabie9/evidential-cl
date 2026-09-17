#!/bin/bash
# CIL probe: B0 (BCL-Dual baseline) + GLOBAL RESERVOIR replay pool (--bcl_global_reservoir).
#
# Single-flag change from B0. The per-task replay buffers are replaced by one flat
# n_memories pool admitted by Vitter reservoir over the whole stream (eralg4/Res-ER's E0
# mechanism), with a per-slot task id so the per-task distillation freeze still works. The
# dual-memory validation buffer is UNTOUCHED (val_fraction stays at its 0.2 default and the
# outer loss still samples the per-task val buffer) -- that is what distinguishes this from
# B3+gres (scripts/run_cil_b3_gres.sh), which sets val_fraction 0.
#
# CAVEAT -- replay-capacity confound. B0 splits each task's budget 0.8 replay / 0.2 val, so
# its replay capacity is ~0.8*n_memories = 4096. gres_cap is the full n_memories = 5120, and
# the per-task val buffers (1024) still sit on top. So this arm has ~25% more replay slots
# and a larger total footprint than B0. B3+gres did not have this problem (val_fraction 0 =>
# both arms 5120). If gres wins here, re-run budget-matched before citing the delta.
#
# Regime matches the honest post-mask-fix B0 baseline
# (logs/bcl_dual/fixed-cil-mask_se_cil-2026-07-24_16-58-55-9781, Signal F1 0.0483 +/- 0.0026,
# n=9): CIL loader, single-epoch, inner_steps 2, lr 0.001, beta 1. Seeds 0,39,55 are the
# first three of that run's seed set, so this n=3 probe pairs directly against it.
#
# Usage:  bash scripts/run_cil_b0_gres.sh
#         SEEDS=1,2,3 bash scripts/run_cil_b0_gres.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

SEEDS="${SEEDS:-0,39,55}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/cil_b0_gres}"
mkdir -p "$LOGDIR"

echo "[b0gres] START bcl_b0_gres_se_cil seeds=$SEEDS $(date)"
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/cil/bcl_dual.yaml" \
  --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
  --bcl_global_reservoir \
  --expt_name bcl_b0_gres_se_cil \
  --seeds "$SEEDS" > "$LOGDIR/bcl_b0_gres_se_cil.log" 2>&1
echo "[b0gres] END exit=$? $(date)"
