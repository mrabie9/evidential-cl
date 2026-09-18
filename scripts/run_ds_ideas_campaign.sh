#!/bin/bash
# Chain the three Dempster-Shafer follow-up campaigns behind whatever is already
# running, two jobs at a time. Sequential rather than parallel on purpose: the
# box already carries an unrelated training job, and three concurrent 10-task
# runs contend for the GPU badly enough to distort the runtime column.
#
#   1. cautious rule (idea 5) -- the extra lambda cell the measured Omega mass
#      ratio points at. The sweep's own trace put final Omega at 122.6 (sum) vs
#      92.3 (max), a ratio of 1.33, so the re-centred peak is near 3.2e5 and the
#      original grid's first gap (2.4e5 -> 7.2e5) straddles it without landing on
#      it. 3.6e5 is that cell.
#   2. evidence injection (idea 7) -- what a buffer should store.
#   3. I_p family (idea 6) -- p=1 sparsity objective and the i1 tracked scalar.
set -u

REPO=/home/lunet/wsmr11/repos/La-MAML-EUCR
PYTHON=/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python
LOGS=$REPO/scripts/logs

while pgrep -f "run_cautious_rule_benchmark.py" > /dev/null; do
  sleep 60
done

echo "[campaign] cautious rule: re-centred lambda cell"
$PYTHON $REPO/scripts/run_cautious_rule_benchmark.py \
  --arms max --seeds 0 --jobs 2 --omega-trace --lambdas 360000 \
  --csv scripts/logs/cautious_rule_recentred.csv \
  > $LOGS/campaign_cautious_recentred.log 2>&1

echo "[campaign] evidence injection"
$PYTHON $REPO/scripts/run_evidence_injection_benchmark.py \
  --seeds 0 --jobs 2 \
  > $LOGS/campaign_injection.log 2>&1

echo "[campaign] I_p family"
$PYTHON $REPO/scripts/run_ip_family_benchmark.py \
  --seeds 0 --jobs 2 \
  > $LOGS/campaign_ip_family.log 2>&1

echo "[campaign] done"
