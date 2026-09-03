#!/usr/bin/env bash
# Regeneration chain for the direct_rcond 1e-12 -> 1e-14 change.
#
# Why: the measured/floor ratio of the order study grew with N (elasticity-2d
# u_H1 2.07 -> 20.8 across N=200..1000) because the truncation level, not the
# dictionary, limited the k=7 solve.  A probe at rcond 1e-14 puts every ratio
# back in 1.3-2.4 with no trend in N.  The two-dimensional elasticity,
# plane-stress, and plate defaults now carry 1e-14, so their solver-side
# studies have to be measured again.  The final three-dimensional Hu--Zhang
# campaign is separate and uses rcond 1e-13.
#
# The best-approximation floors do not involve a solver and are unaffected.
#
# Every step is individually resumable; a failure only costs its own step.
set -u
cd "$(dirname "$0")"
FLOORS=approximation-floors
STATUS="results/logs/rcond14-status.txt"
# The displaced rcond 1e-12 runs are kept, not discarded: they are the only
# measurement behind the truncation-level ceiling reported in the paper.  The
# guard therefore refuses to overwrite an existing archive.
ARCHIVE="results-archive/rcond12-comparison"

if [ -d results ] && [ ! -d "$ARCHIVE" ]; then
  mkdir -p "$ARCHIVE"
  for model in elasticity-2d plane-stress plate; do
    if [ -d "results/$model" ]; then
      mv "results/$model" "$ARCHIVE/"
    fi
  done
fi
mkdir -p results/logs "$FLOORS/results/logs"
: > "$STATUS"

step() {
  local name=$1
  shift
  echo "=== $(date '+%F %T') START ${name} ===" >> "$STATUS"
  if "$@"; then
    echo "=== $(date '+%F %T') DONE ${name} ===" >> "$STATUS"
  else
    echo "=== $(date '+%F %T') FAILED ${name} (exit $?) ===" >> "$STATUS"
  fi
}

step campaign bash -c "bash run_full_campaign.sh > results/logs/campaign.log 2>&1"

step plots bash -c "python plot_convergence.py > results/logs/plot-convergence.log 2>&1"
step solver-gap bash -c \
  "cd $FLOORS && python run_solver_gap.py --study order > results/logs/solver-gap.log 2>&1"
step solver-gap-k3 bash -c \
  "cd $FLOORS && python run_solver_gap.py --study order-k3 > results/logs/solver-gap-k3.log 2>&1"

echo "=== $(date '+%F %T') CHAIN COMPLETE ===" >> "$STATUS"
