#!/usr/bin/env bash
# Full paper-scale study campaign; each study writes results/ independently,
# so the campaign can be resumed by re-running the remaining lines.
set -u
cd "$(dirname "$0")"
mkdir -p results/logs

run() {
  local study=$1 model=$2
  local log="results/logs/${model}-${study}.log"
  if [ -f "results/${model}/${study}/results.json" ]; then
    echo "SKIP ${model}/${study} (results exist)"
    return 0
  fi
  echo "=== $(date '+%F %T') START ${model}/${study} ==="
  if python study_runner.py "${study}" --model "${model}" > "${log}" 2>&1; then
    echo "=== $(date '+%F %T') DONE ${model}/${study} ==="
  else
    echo "=== $(date '+%F %T') FAILED ${model}/${study} (see ${log}) ==="
    return 1
  fi
}

run main elasticity-2d
run q    elasticity-2d
run k    elasticity-2d
run main plate
run q    plate
run k    plate
run main plane-stress
run main elasticity-3d
run nu   elasticity-3d
echo "=== $(date '+%F %T') CAMPAIGN COMPLETE ==="
