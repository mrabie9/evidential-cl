#!/bin/bash
# Res-ER (eralg4) in CIL at lr 0.001, inner_steps 2, two-forward replay loop, AMP OFF.
#
# WHY: the MX row of Table~tab:cil_ablation compares a C-MAML arm against Res-ER's E0, but
# E0 runs at lr 0.01 while every C-MAML CIL arm runs at lr 0.001, so the contrast confounds
# method with learning rate -- the same confound that inverted the earlier E1-vs-B5a result.
# This run supplies the lr-matched comparator. inner_steps 2 matches on theta-updates
# (Res-ER and C-MAML both take one optimiser step per inner step; C-MAML's inner_update is a
# functional fast-weight step that never touches an optimiser), and --eralg4_joint_er holds
# E0's other non-default setting fixed so lr and precision are the only changes.
#
# The two pre-existing lr 0.001 Res-ER CIL runs (eralg4_resER_reduce_se_cil-2026-07-24_*) are
# dead: bare seed-0 dirs, no results.txt, and they ran at inner_steps 4. Superseded by this.
#
# CAVEAT -- precision axis. E0 and every C-MAML CIL arm ran under bf16 AMP (the CIL block's
# precision). This run is --no-amp, so it differs from E0 on TWO axes (lr AND precision) and
# is NOT a drop-in lr-only replacement for the E0 baseline row. AMP is known in this repo to
# be worth over a point on some methods and to flip cross-method ordering, so pairing this
# against an AMP-trained C-MAML arm reintroduces a confound on the other side. A matching
# --no-amp C-MAML run is needed before MX can be read as a clean method contrast.
#
# n=9 canonical seeds, matching E0 / M0 / M4 so pairing is complete.
#
# Usage:  bash scripts/run_eralg4_lr001_is2_noamp_cil.sh
#         SEEDS=0,39,55 bash scripts/run_eralg4_lr001_is2_noamp_cil.sh
set -uo pipefail

REPO="/home/lunet/wsmr11/repos/evidential-cl"
cd "$REPO" || exit 1   # main.py writes logs/ relative to cwd; anchor it here
PY="/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python"

SEEDS="${SEEDS:-0,1,2,3,39,55,7,13,21}"
LOGDIR="${LOGDIR:-$REPO/scripts/logs/eralg4_lr001_noamp}"
mkdir -p "$LOGDIR"

echo "[resER] START eralg4_resER_joint_lr001_noamp_se_cil seeds=$SEEDS $(date)"
"$PY" "$REPO/main.py" \
  --config "$REPO/configs/base.yaml" \
  --config "$REPO/configs/models/cil/eralg4.yaml" \
  --n_epochs 1 --inner_steps 2 --no-save_checkpoints \
  --lr 0.001 --eralg4_joint_er --no-amp \
  --expt_name eralg4_resER_joint_lr001_noamp_se_cil \
  --seeds "$SEEDS" > "$LOGDIR/eralg4_resER_joint_lr001_noamp_se_cil.log" 2>&1
echo "[resER] END exit=$? $(date)"
