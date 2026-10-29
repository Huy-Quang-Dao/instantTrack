#!/bin/bash
# Main comparison: SuperMimic + EMA vs InstantTrack, 3 seeds each, 0.75 h per run (~5 h on one GPU,
# training and evaluation are sequential so the GPU is never shared), then the comparison table.
# Add "supermimic" to METHODS for the plain baseline (needs ~2 h per run to reach the target).
set -euo pipefail
cd "$(dirname "$0")/.."
METHODS=${METHODS:-"supermimic_ema instanttrack"}
for m in $METHODS; do
  for s in 0 1 2; do
    [ -f "logs/${m}_s$s/eval.csv" ] || scripts/train.sh "$m" "${m}_s$s" --seed "$s" --max_hours 0.75
  done
done
uv run python tools/compare.py $(for m in $METHODS; do for s in 0 1 2; do echo "logs/${m}_s$s"; done; done) --out results/reproduce
