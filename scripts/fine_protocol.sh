#!/bin/bash
# Fine-grained time to target: the instanttrack preset with a checkpoint every 100 iterations (no wall-clock
# checkpoints), every checkpoint evaluated with 4 replicas (eval_r4.csv), then tools/time_to_target.py.
# Usage: bash scripts/fine_protocol.sh <run_name> <seed> [hours, default 0.1]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/presets.sh
RUN=$1; SEED=$2; H=${3:-0.1}
mkdir -p "logs/$RUN"
uv run python -m instanttrack.train --log_dir "logs/$RUN" ${PRESET[instanttrack]} --seed "$SEED" --max_hours "$H" \
  --ckpt_minutes 0 --ckpt_every 100 > "logs/$RUN/stdout.log" 2>&1
uv run python tools/eval_batch.py --run_dir "logs/$RUN" --group 26 --reps 4 > "logs/$RUN/eval_stdout.log" 2>&1
uv run python tools/time_to_target.py "logs/$RUN"
