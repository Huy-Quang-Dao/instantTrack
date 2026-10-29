#!/bin/bash
# Train one preset, then evaluate every checkpoint on all 40 clips.
#   scripts/train.sh <preset> <run_name> [extra train.py args...]
#   scripts/train.sh instanttrack it_s0 --seed 0 --max_hours 0.75
# Presets: supermimic, supermimic_ema, instanttrack_track, instanttrack (see scripts/presets.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/presets.sh
P=${1:?preset}; RUN=${2:?run name}; shift 2
[ -n "${PRESET[$P]+x}" ] || { echo "unknown preset $P (${!PRESET[*]})"; exit 1; }
mkdir -p "logs/$RUN"
uv run python -m instanttrack.train --log_dir "logs/$RUN" ${PRESET[$P]} "$@" 2>&1 | tee -a "logs/$RUN/stdout.log"
uv run python tools/eval_batch.py --run_dir "logs/$RUN" --group 16 2>&1 | tee -a "logs/$RUN/eval_stdout.log"
