#!/bin/bash
# PR-3, E0 diagnosis. E0 missed the recorded control:
#   recorded 0.5206 / 0.5008 / -0.0198
#   E0       0.5224 / 0.4984 / -0.0240
# Three candidate causes, three runs to separate them.
#
#   A  exactly the recorded command line on today's code (pre-pass now gated to
#      the frozen arm, --no-save_checkpoints as recorded). Isolates whether the
#      *pre-existing* uncommitted tree changes since 2026-08-18 moved the control.
#   B  a second copy of A. Run-to-run determinism check: cudnn.deterministic is
#      True and benchmark False, so A and B must agree exactly.
#   C  A plus --save_checkpoints. Isolates checkpoint writing.
#
# If A == B == 0.5008 the baseline is intact and the culprit is the pre-pass
# and/or checkpointing. If A == B != 0.5008 the working tree no longer
# reproduces the recorded bar and every comparison drawn against it needs
# re-basing. If A != B the run is not deterministic and "bit-identical" is not
# an available gate at all.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PY=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs/mu_frozen
mkdir -p "$LOGS"

run () {
  local name=$1 ckpt=$2
  WOE_LC_DEBUG=1 PYTHONUNBUFFERED=1 $PY $REPO/main.py \
    --config configs/base.yaml \
    --config configs/models/til/woe_si_lc.yaml \
    --n_epochs 1 --inner_steps 2 --lr 0.003 \
    --woe_omega_transform abs --woe_anchor_mode proximal \
    --woe_omega_accum sum --woe_importance_scalar i2 \
    --woe_lambda 240000.0 \
    --woe_mu_mode ema \
    --single-seed --seed 0 "$ckpt" \
    --expt_name "$name" > "$LOGS/$name.log" 2>&1 &
}

run mufz_e0a_noprepass_nockpt_s0 --no-save_checkpoints
run mufz_e0b_noprepass_nockpt_s0 --no-save_checkpoints
run mufz_e0c_noprepass_ckpt_s0   --save_checkpoints
wait

printf "%-34s %8s %8s %9s\n" RUN DIAG FINAL BWT
for n in mufz_e0a_noprepass_nockpt_s0 mufz_e0b_noprepass_nockpt_s0 mufz_e0c_noprepass_ckpt_s0; do
  r=$(ls -d $REPO/logs/woe_si_lc/${n}-*/0/results.txt 2>/dev/null | tail -1)
  [ -n "$r" ] || { printf "%-34s %8s\n" "$n" MISSING; continue; }
  printf "%-34s %8s %8s %9s\n" "$n" \
    "$(grep -m1 'Diagonal F1:' "$r" | awk '{print $3}')" \
    "$(grep -m1 'Final F1:' "$r" | awk '{print $3}')" \
    "$(grep -m1 'Backward:' "$r" | awk '{print $2}')"
done
echo "recorded control                     0.5206   0.5008   -0.0198"
