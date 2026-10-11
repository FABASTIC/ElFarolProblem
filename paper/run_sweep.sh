#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
JOBS="${JOBS:-4}"
NIGHTS="${NIGHTS:-100}"
LATE="${LATE:-$((NIGHTS / 5))}"
SIZES="${SIZES:-80 100 150}"
SEEDS="${SEEDS:-$(seq 1 12)}"
OUT="outputs/paper/h${NIGHTS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" PY NIGHTS OUT
if [ -e "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "$OUT already exists; move it away or set FORCE=1 to overwrite" >&2
  exit 1
fi
rm -rf "$OUT"
mkdir -p "$OUT/logs"
for n in $SIZES; do for s in $SEEDS; do echo "$n $s"; done; done |
  xargs -P "$JOBS" -n 2 sh -c '"$PY" -u experiment.py --output-dir "$OUT/parts/n$0_s$1" --agents $0 --epochs "$NIGHTS" --seeds $1 --device cpu --no-live-state > "$OUT/logs/n$0_s$1.log" 2>&1'
for n in $SIZES; do
  mkdir -p "$OUT/n$n"
  mv "$OUT"/parts/n${n}_s*/*_seed_* "$OUT/n$n/"
  "$PY" analyzer.py --experiment-dir "$OUT/n$n" --quiet
  "$PY" analyzer.py --experiment-dir "$OUT/n$n" --epoch-min $((NIGHTS - LATE)) --output "$OUT/n$n/analytics_report_late.json" --quiet
done
rm -rf "$OUT/parts"
echo "$OUT: $NIGHTS nights, full-run and last-$LATE-night reports written"
