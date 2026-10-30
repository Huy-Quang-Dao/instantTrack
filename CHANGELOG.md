# Changelog

All notable changes to InstantTrack are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All results are measured on one RTX 5090
with the protocol in the README ("Time to target"); every number below can be regenerated from
`results/runs/` with `tools/make_report.py`.

## [1.1.0] - 2026-10-30

### Results

| | v1.0.0 | 1.1.0 |
|---|---|---|
| time to return ≥ 0.82 with 0/40 failed clips, held ×3 (mean) | 0.25 h (3 seeds) | **0.19 h** (5 seeds: 0.15 / 0.16 / 0.25 / 0.20 / 0.20) |
| speed-up over SuperMimic + EMA (0.55 h) | 2.2× | **2.8×** |
| iterations to target | 5900–6000 | 3687–5847 |
| iterations per second | 6.32 | 6.28 |
| body position / orientation / joint error, 0.75 h | 3.53 cm / 7.5° / 4.56° | 3.20 cm / 6.5° / 4.12° |
| contact agreement / foot slip, 0.75 h | 0.812 / 6.62 cm/s | 0.817 / 6.59 cm/s |
| penetration / root drift, 0.75 h | 0.052 mm / 61 cm | 0.055 mm / 61 cm |
| normalized return at 1.5 h (1 run) | 0.877 | 0.881 (SuperMimic + EMA: 0.879–0.880) |

### Added
- `--ref_residual β` (policy): the decoder outputs a correction on top of
  β · (q_ref(t+1) − q_default) / α, the next reference joint pose read from the goal input. With
  β = 1 the joint targets are q_ref(t+1) + α·decoder(z, o), so an untrained policy commands the
  reference instead of the standing pose. q_default and α are checkpoint buffers. Part of the
  `instanttrack` preset.
- `--root_reward w`, `--root_reward_std s` (imagination reward): a wide-kernel global root-position
  term. Kept for reproducibility; not used by any preset (no gain, see below).
- `--wm_ens K`, `--wm_ens_pen λ` (world model): K increment heads on a shared trunk, a bootstrapped
  one-step loss per head, and an imagination penalty on their disagreement. The disagreement is
  logged as `ens_u`. Kept for reproducibility; not used by any preset.
- `--pi_experts E` (policy): E decoders mixed by a softmax gate on the motion latent. Kept for
  reproducibility; not used by any preset.
- `instanttrack_v1.0` preset: the exact v1.0 configuration.
- Logs of the 13 runs of this round in `results/runs/` (`e68`–`e80`), their commands in
  `experiments/all_runs.txt`, and metrics in `results/metrics_a1_075h.md`,
  `results/metrics_a1_150h.md` and `results/metrics_r2a_075h.md`.
- `media/equal_time_9min.gif` (SuperMimic + EMA, v1.0 and v1.1 after 9 minutes of training) and new
  videos on the project page.
- `CHANGELOG.md`.

### Changed
- `instanttrack` preset = v1.0 + `--ref_residual 1.0`.
- `tools/eval_batch.py`, `tools/eval_metrics.py`, `instanttrack.evaluate` and `tools/export_onnx.py`
  read `ref_residual` and `pi_experts` from the checkpoint's `args` (default: v1.0 behaviour).
- `tools/make_report.py`, README tables and figures report v1.0 and v1.1 side by side.
- `WorldModel.step_c` is split into `step_full` and `integrate`. Outputs are unchanged when the new
  options are off.

### Fixed
- `tools/queue.sh`: `RUN:<tag>` lines were never marked done, so they ran again on every pass of the
  queue.

### Tried and not adopted (single seed each, same seed as the configuration they modify)
| change | on top of | held ×3 | note |
|---|---|---|---|
| root reward 0.5 / 1.0 (std 1 m), 0.5 (std 0.5 m) | v1.0 (0.25 h) | 0.28 / 0.35 / 0.25 h | root drift 56 / 57 / 60 cm vs 53 cm |
| reference residual β = 0.5 | v1.0 (0.25 h) | 0.17 h | β = 1 gives 0.15 h |
| WM ensemble, 4 heads, penalty 100 / 300 | v1.1 (0.15 h) | 0.25 / 0.35 h | head disagreement collapses to ~1e-4 |
| mixture-of-experts policy, 2 decoders | v1.1 (0.15 h) | 0.20 h | 5.18 it/s (−18 %) |

### Known limitations
- Root drift is still larger than SuperMimic + EMA (55–64 cm vs 49–54 cm).
- In 3 of 5 seeds, one or two hard clips (`fallAndGetUp*`, `jumps1_subject2`) fail at a single
  checkpoint after the first hit, which delays "held ×3" by 0.05–0.10 h.
- Accuracy at 0.75 h is measured on 3 of the 5 seeds, and the 1.5 h comparison is one run.
- Re-evaluating the same checkpoint moves return by about ±0.001 and root drift by a few cm.
- Simulation only (Unitree G1, LAFAN1). No sim-to-real experiment.

### Compatibility
No breaking changes. v1.0 checkpoints and logs load and evaluate unchanged. To reproduce v1.0 exactly,
use the `instanttrack_v1.0` preset.

## [1.0.0] - 2026-10-29

First public release.

- InstantTrack: SuperMimic plus a policy EMA, no host syncs, fused Adam, fused 3×3 rotations, extra
  `torch.compile` coverage, lr 4e-4 with 5 world-model batches, a 1024³ world model,
  failure-focused resets and a contact reward in imagination.
- 0.25 h to return ≥ 0.82 with 0/40 failed clips (3 seeds), against 0.55 h for SuperMimic + EMA and
  1.50 h for SuperMimic on the same machine.
- Logs of all 60+ runs, report generator, presets, reproduction script, project page.

[1.1.0]: https://github.com/Huy-Quang-Dao/instantTrack/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Huy-Quang-Dao/instantTrack/releases/tag/v1.0.0
