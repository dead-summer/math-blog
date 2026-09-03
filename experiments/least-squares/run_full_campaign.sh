#!/usr/bin/env bash
# Full paper-scale study campaign; each study writes results/ independently,
# so the campaign can be resumed by re-running the remaining lines.
set -u
cd "$(dirname "$0")"
mkdir -p results/logs

# run <study> <model> [extra study_runner args...]
# The output directory carries the study name, so a study repeated under a
# different activation power writes to its own directory via --tag.
run() {
  local study=$1 model=$2
  shift 2
  local tag="${study}"
  # A trailing "--tag <name>" pair renames the output directory only.
  if [ "${1:-}" = "--tag" ]; then
    tag="$2"
    shift 2
  fi
  local log="results/logs/${model}-${tag}.log"
  if [ -f "results/${model}/${tag}/results.json" ]; then
    echo "SKIP ${model}/${tag} (results exist)"
    return 0
  fi
  echo "=== $(date '+%F %T') START ${model}/${tag} ==="
  if python study_runner.py "${study}" --model "${model}" --output-name "${tag}" \
      "$@" > "${log}" 2>&1; then
    echo "=== $(date '+%F %T') DONE ${model}/${tag} ==="
  else
    echo "=== $(date '+%F %T') FAILED ${model}/${tag} (see ${log}) ==="
    return 1
  fi
}

# Order (N-convergence) ladders use k=7.  The k=3 ladders are retained only as
# end-to-end endpoint comparisons in the power-study subsections.
run order elasticity-2d
run order elasticity-2d --tag order-k3 --activation-power 3
run q    elasticity-2d
run k    elasticity-2d
run order plate
run order plate --tag order-k3 --activation-power 3
run q    plate
run k    plate
run order plane-stress

# The paper's three-dimensional Hu--Zhang experiment has a denser Q=16(N+1)
# rule, a fixed projection rule, and one fresh process per repetition.  It is
# deliberately kept out of this generic chain; use run_hu_zhang_3d.sh.

if [ -f "results/elasticity-2d/near-incompressible/results.json" ]; then
  echo "SKIP elasticity-2d/near-incompressible (results exist)"
else
  echo "=== $(date '+%F %T') START elasticity-2d/near-incompressible ==="
  if OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
      LS_TORCH_THREADS="${LS_TORCH_THREADS:-4}" \
      python run_near_incompressible_2d.py \
      > "results/logs/elasticity-2d-near-incompressible.log" 2>&1; then
    echo "=== $(date '+%F %T') DONE elasticity-2d/near-incompressible ==="
  else
    echo "=== $(date '+%F %T') FAILED elasticity-2d/near-incompressible (see log) ==="
    exit 1
  fi
fi

# Power sweep (the paper's 关于幂次 $k$ 的收敛实验 subsections).  Two-dimensional
# elasticity uses the capacity-probe configuration that cleanly separates
# k=3,5,7,9 over ten independent repetitions.
run power elasticity-2d --q-ratio 16 --ritz-degree 10 \
  --projection-samples 12808 --direct-rcond 1e-14 --algorithms ball \
  --budgets 1000,3000,10000,30000,inf
run power plate --q-ratio 8 --ritz-degree 10 \
  --projection-samples 12808 --direct-rcond 1e-14 \
  --algorithms ball,ridge,tsvd
echo "=== $(date '+%F %T') CAMPAIGN COMPLETE ==="
