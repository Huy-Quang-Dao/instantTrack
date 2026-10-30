# Per-run logs

One directory per training run (`logs/<run>/` copied without checkpoints):

* `args.json` — every `train.py` flag of the run
* `train_log.csv` — training metrics every 50 iterations (losses, imagined / simulated return, it/s)
* `eval.csv` — offline evaluation of every checkpoint on all 40 clips (`tools/eval_batch.py`):
  iteration, training wall-clock (h), normalized return, number and names of failed clips

Run names are `e<id>_<short description>_s<seed>`; [`tools/make_report.py`](../../tools/make_report.py)
lists which runs form each table row. Two caveats:

* `e3_supermimic_ema_s1` shared the GPU with another job for part of its run (2.1 it/s instead of
  3.1), so its wall-clock is not valid; it was re-run as `e58_sm_ema_s1`, which is the one reported.
* `e42_gpu_s2` was restarted after a machine crash (the partial run was discarded); `e54_wc01_s0` ran
  right after a reboot with a cold torch.compile cache (~0.05 h of extra compilation).
* v1.1 round (2026-10-29/30): `e68`–`e70` root reward, `e71`–`e74`, `e78`, `e79` reference residual
  (v1.1, seeds 0–4; `e72` is β = 0.5), `e75`/`e76` world-model ensemble, `e77` mixture-of-experts
  policy, `e80` the 1.5 h v1.1 run. Nothing else used the GPU during these runs; offline evaluation
  and video rendering ran only between runs.
