#!/bin/bash
# v1.4 dev: fine protocol (scripts/fine_protocol.sh, 0.1 h) with extra flags. bash scripts/fine_exp.sh <run> <seed> [flags...]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/presets.sh
RUN=$1; SEED=$2; shift 2
mkdir -p "logs/$RUN"
uv run python -m instanttrack.train --log_dir "logs/$RUN" ${PRESET[instanttrack]} --seed "$SEED" --max_hours 0.1 \
  --ckpt_minutes 0 --ckpt_every 100 "$@" > "logs/$RUN/stdout.log" 2>&1
uv run python tools/eval_batch.py --run_dir "logs/$RUN" --group 26 --reps 4 > "logs/$RUN/eval_stdout.log" 2>&1
uv run python tools/time_to_target.py "logs/$RUN"
