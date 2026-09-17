#!/bin/bash
# Serialise the two importance campaigns behind whatever is already in flight.
#
# `ps -C python` rather than `pgrep -f`: this script's own command line contains
# the patterns being searched for, and a pgrep -f on them matches the waiter
# itself, which never exits. Matching only python processes avoids that.
set -u
REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR

wait_for () {
  while ps -C python -o args= 2>/dev/null | grep -q "$1"; do sleep 60; done
}

echo "[chain] waiting for in-flight omguni runs"
wait_for "omguni"
echo "[chain] launching remaining EWC uniform cells"
bash "$REPO/scripts/run_omega_uniform_control.sh"
wait_for "omguni"
echo "[chain] single-epoch controls complete; launching multi-epoch"
bash "$REPO/scripts/run_multiepoch_importance.sh"
echo "[chain] all importance campaigns complete"
