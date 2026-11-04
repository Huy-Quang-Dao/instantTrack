#!/bin/bash
# Main comparison: SuperMimic + EMA (3 seeds, 0.75 h) and InstantTrack (5 seeds, 0.5 h), one run at a time so the
# GPU is never shared (training and evaluation are sequential), then the comparison table (~6 h on one GPU).
# Add "supermimic" to METHODS for the plain baseline (needs ~2 h per run to reach the target).
set -euo pipefail
cd "$(dirname "$0")/.."
METHODS=${METHODS:-"supermimic_ema instanttrack"}
declare -A SEEDS=([supermimic]="0" [supermimic_ema]="0 1 2" [instanttrack]="0 1 2 3 4")
declare -A HOURS=([supermimic]=2 [supermimic_ema]=0.75 [instanttrack]=0.5)
RUNS=()
for m in $METHODS; do
  for s in ${SEEDS[$m]}; do
    [ -f "logs/${m}_s$s/eval.csv" ] || scripts/train.sh "$m" "${m}_s$s" --seed "$s" --max_hours "${HOURS[$m]}"
    RUNS+=("logs/${m}_s$s")
  done
done
uv run python tools/compare.py "${RUNS[@]}" --out results/reproduce
