"""Time to target of one or more evaluated runs (tools/eval_batch.py output).

For each run directory, reads ``eval_r<reps>.csv`` (``--file``; ``eval.csv`` without replicas): one row per checkpoint
with the training wall-clock ``train_wall_h``, the normalized return and the number of failed clips (averaged over
the replicas). Reports, in minutes of training wall-clock:

  first >= thr      first checkpoint with normalized return >= thr (SuperMimic's metric)
  stable k          first checkpoint from which k consecutive checkpoints all have return >= thr and 0 failed clips
                    (k = 3: InstantTrack's rule; k = 6: a stricter rule)

Consecutive checkpoints are ``ckpt_every`` iterations apart (from the run's args.json): checkpoints off that grid
(e.g. the last one of a --max_hours run) count for "first" only, and a missing checkpoint breaks a stable run.

    python tools/time_to_target.py logs/reproduce/s0 logs/reproduce/s1 logs/reproduce/s2
"""

import argparse
import csv
import json
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--file", type=str, default="eval_r4.csv", help="result file in each run directory")
ap.add_argument("--thr", type=float, default=0.82)
ap.add_argument("--k", type=int, nargs="+", default=[3, 6])
args = ap.parse_args()


def read(run):
    path = os.path.join(run, args.file)
    rows = [(int(r["iteration"]), float(r["train_wall_h"]) * 60, float(r["norm_return"]), float(r["n_failed"]))
            for r in csv.DictReader(open(path))]  # fmt: skip
    step = json.load(open(os.path.join(run, "args.json")))["ckpt_every"]
    return sorted(rows), step


def first(rows, thr):
    return next(((it, m) for it, m, ret, _ in rows if ret >= thr), None)


def stable(rows, step, thr, k):
    ok = {it: ret >= thr and nf == 0 for it, _, ret, nf in rows}
    return next(((it, m) for it, m, _, _ in rows if it % step == 0 and all(ok.get(it + j * step, False) for j in range(k))), None)


fmt = lambda t: "   --   " if t is None else f"{t[1]:5.2f} min (it {t[0]})"  # noqa: E731
res = {"first": []} | {k: [] for k in args.k}
for run in args.runs:
    rows, step = read(run)
    f = first(rows, args.thr)
    st = {k: stable(rows, step, args.thr, k) for k in args.k}
    res["first"].append(f)
    for k in args.k:
        res[k].append(st[k])
    print(f"{run}: first >= {args.thr} {fmt(f)} | " + " | ".join(f"stable {k} {fmt(st[k])}" for k in args.k))
if len(args.runs) > 1:
    for key, vals in res.items():
        got = [v[1] for v in vals if v is not None]
        name = f"first >= {args.thr}" if key == "first" else f"stable {key}"
        mean = f"{statistics.mean(got):.2f} min" if len(got) == len(vals) else f"not reached in {len(vals) - len(got)} run(s)"
        print(f"mean {name}: {mean}")
