#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
JOBS="${JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" PY
mkdir -p outputs/paper/logs
for n in 80 100 150; do for s in $(seq 1 12); do echo "$n $s"; done; done |
  xargs -P "$JOBS" -n 2 sh -c '"$PY" -u experiment.py --output-dir outputs/paper/parts/n$0_s$1 --agents $0 --epochs 100 --seeds $1 --device cpu --no-live-state > outputs/paper/logs/n$0_s$1.log 2>&1'
for n in 80 100 150; do
  mkdir -p outputs/paper/n$n
  mv outputs/paper/parts/n${n}_s*/*_seed_* outputs/paper/n$n/
  "$PY" analyzer.py --experiment-dir outputs/paper/n$n --quiet
done
"$PY" paper/analysis.py > /dev/null
echo "sweep, analysis, tables and figures are up to date"
