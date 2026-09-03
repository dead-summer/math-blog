#!/usr/bin/env bash
# Stable Hu--Zhang manufactured-solution campaign for the paper's 3-D order study.
#
# Every repetition runs in a fresh process.  This is essential on small-memory
# hosts because the LAPACK/PyTorch SVD workspace is not necessarily returned to
# the operating system before the next configuration is assembled.
set -u
set -o pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
ROOT_OUT=${ROOT_OUT:-results-remote/2026-08-30-3d-hu-zhang-k7-q16-qr16384}
WIDTHS=${LS_3D_WIDTHS:-"200 400 600 800 1000"}
REPEATS=${LS_3D_REPEATS:-10}
Q_RATIO=${LS_3D_Q_RATIO:-16}
PROJECTION_SAMPLES=${LS_3D_PROJECTION_SAMPLES:-16384}
DIRECT_RCOND=${LS_3D_DIRECT_RCOND:-1e-13}
THREADS=${LS_PARALLEL_THREADS:-4}
SVD_BACKEND=${LS_SVD_BACKEND:-scipy}

PARTS="$ROOT_OUT/parts/order"
LOG_DIR="$ROOT_OUT/logs/order"
mkdir -p "$PARTS" "$LOG_DIR"

is_complete() {
  local out=$1
  [ -f "$out/results.json" ] && [ "$(cat "$out/.exit" 2>/dev/null || true)" = 0 ]
}

for n in $WIDTHS; do
  for run in $(seq 0 $((REPEATS - 1))); do
    out="$PARTS/N-${n}/run-${run}"
    log="$LOG_DIR/N-${n}-run-${run}.log"
    if is_complete "$out"; then
      echo "SKIP N=${n} run=${run}"
      continue
    fi
    mkdir -p "$out"
    echo "=== $(date '+%F %T') START N=${n} run=${run} ===" | tee -a "$ROOT_OUT/status.log"
    env LS_TORCH_THREADS="$THREADS" LS_SVD_BACKEND="$SVD_BACKEND" LS_SVD_THREADS="$THREADS" \
      OPENBLAS_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      "$PYTHON" -u study_runner.py order --model elasticity-3d \
        --widths "$n" --q-ratio "$Q_RATIO" --repeats 1 --run-offset "$run" \
        --algorithms ball --budgets inf --activation-power 7 --ritz-degree 7 \
        --manufactured-solution hu_zhang --projection-samples "$PROJECTION_SAMPLES" \
        --validation-points 4096 --test-points 32768 --direct-rcond "$DIRECT_RCOND" \
        --direct-solver streaming_tsqr --direct-batch-size 2048 \
        --direct-qr-block-size 128 --evaluation-batch-size 8192 \
        --projection-batch-size 4096 --body-force-batch-size 5000 \
        --output-dir "$out" > "$log" 2>&1
    status=$?
    echo "$status" > "$out/.exit"
    echo "=== $(date '+%F %T') END N=${n} run=${run} status=${status} ===" \
      | tee -a "$ROOT_OUT/status.log"
    if [ "$status" -ne 0 ]; then
      exit "$status"
    fi
  done
done

"$PYTHON" merge_3d_final.py --source "$ROOT_OUT" --source-study order \
  --destination results --destination-study order
echo "=== $(date '+%F %T') HU--ZHANG ORDER CAMPAIGN COMPLETE ===" | tee -a "$ROOT_OUT/status.log"
