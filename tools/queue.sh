#!/bin/bash
# Sequential experiment queue for one GPU: train, then batch-evaluate, one run at a time.
# Each non-empty, non-comment line of the queue file is "<run_name> <train.py args...>". Lines are
# read one at a time, so the file can be edited while the queue runs; finished runs are skipped
# (logs/<run>/eval.csv exists). Waits for any training tmux session that is already running.
#   tmux new -d -s queue "tools/queue.sh experiments/all_runs.txt"
QUEUE=${1:-experiments/all_runs.txt}
cd "$(dirname "$0")/.."
while true; do
  while tmux ls 2>/dev/null | grep -qE '^e[0-9]+_' || pgrep -f 'python3 -m instanttrack.train' >/dev/null; do sleep 30; done
  line=$(grep -vE '^\s*(#|$)' "$QUEUE" | while read -r name rest; do [ -f "logs/$name/eval.csv" ] || { echo "$name $rest"; break; }; done)
  [ -z "$line" ] && { echo "queue empty"; break; }
  name=${line%% *}; rest=${line#* }; [ "$rest" = "$name" ] && rest=""
  if [[ "$name" == RUN:* ]]; then  # "RUN:<tag> <shell command>": run once, then mark logs/<tag>/eval.csv done
    tag=${name#RUN:}; mkdir -p "logs/$tag"; echo "[$(date '+%F %T')] run $tag: $rest"
    bash -c "$rest" > "logs/$tag/stdout.log" 2>&1; touch "logs/$tag/eval.csv"; continue
  fi
  mkdir -p "logs/$name"
  if grep -q "time budget" "logs/$name/stdout.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] $name already trained"
  else
    echo "[$(date '+%F %T')] train $name: $rest"
    uv run python -m instanttrack.train --log_dir "logs/$name" $rest >> "logs/$name/stdout.log" 2>&1
  fi
  echo "[$(date '+%F %T')] eval $name"
  uv run python tools/eval_batch.py --run_dir "logs/$name" --group 16 >> "logs/$name/eval_stdout.log" 2>&1
done
