#!/bin/bash
# v1.4 dev: training speed only (no evaluation). bash scripts/speed.sh <run> <iterations> [extra flags]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/presets.sh
RUN=$1; N=$2; shift 2
mkdir -p "logs/$RUN"
uv run python -m instanttrack.train --log_dir "logs/$RUN" ${PRESET[instanttrack]} --seed 0 --iterations "$N" --ckpt_minutes 0 --ckpt_every 100000 "$@" > "logs/$RUN/stdout.log" 2>&1
