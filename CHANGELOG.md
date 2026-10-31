# Changelog

All notable changes to InstantTrack are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All results are measured on one RTX 5090
with the protocol in the README ("Time to target"); every number below can be regenerated from
`results/runs/` with `tools/make_report.py`.

## [Unreleased]

## [1.2.0] - 2026-11-01

### Results

| | 1.1.0 | 1.2.0 |
|---|---|---|
| time to return ≥ 0.82 with 0/40 failed clips, held ×3 (mean of 5 seeds) | 0.19 h (0.15 / 0.16 / 0.25 / 0.20 / 0.20) | **0.12 h** (0.10 / 0.10 / 0.10 / 0.10 / 0.18) |
| speed-up over SuperMimic + EMA (0.55 h) | 2.8× | **4.7×** |
| iterations to target | 3687–5847 | 3270–3370 (seed 4: 6000) |
| iterations per second | 6.28 | **8.74** |
| return / failed clips after 0.15 h (mean) | 0.822 / 0.2 | 0.855 / 0.2 |
| return, 0.5 h (seeds 0–2, same session) | 0.862 | 0.876 |
| body position / orientation / joint error, 0.5 h | 3.44 cm / 7.2° / 4.39° | 3.24 cm / 6.7° / 4.36° |
| contact agreement / foot slip, 0.5 h | 0.804 / 6.76 cm/s | 0.818 / 6.63 cm/s |
| penetration / root drift, 0.5 h | 0.059 mm / 69 cm | 0.055 mm / 60 cm |
| return at 1.5 h (1 run; SuperMimic + EMA: 0.879) | 0.880 | 0.886 |

The v1.2 runs were checkpointed every 1.5 minutes. The table uses only the checkpoints on the
3-minute schedule of earlier releases; with all checkpoints the times are identical.

### Added
- `--muon`, `--muon_on {pi,wm}` (optimizer): `torch.optim.Muon` with `match_rms_adamw` scaling for
  the hidden weight matrices of the world model and the policy (all `Linear` layers of each MLP
  except the first and last); Adam for the remaining parameters and for the critic. Part of the
  `instanttrack` preset.
- `--cuda_graph` (systems): captures one world-model update and one policy + critic update as CUDA
  graphs and replays them. Uses capturable fused Adam with tensor learning rates, an in-place
  foreach EMA, a loop instead of `cumprod` for the survival weights, and a device copy of the
  replay buffer's write pointer. Requires `--no_sync --fused_adam`. Part of the `instanttrack` preset.
- `--batch_goal_enc` (systems): the motion encoder runs once per imagined rollout on all H × B goals.
  `Policy.forward` takes an optional precomputed latent `z`, and `Policy.encode` is new. Part of the
  `instanttrack` preset.
- `--pi_simba WxB` (policy as SimBa residual stacks) and `--wm_specnorm` (spectral norm on the world
  model's hidden layers). Kept for reproducibility; not used by any preset.
- `instanttrack_v1.1` preset: the exact v1.1 configuration.
- Logs of the 33 runs of this round in `results/runs/` (`e81`–`e113`), their commands in
  `experiments/all_runs.txt`, and metrics in `results/metrics_v12_05h.md` and
  `results/metrics_v12_150h.md`.
- `media/equal_time_6min.gif` (SuperMimic + EMA, v1.1 and v1.2 after 6 minutes of training) and new
  videos on the project page.

### Changed
- `instanttrack` preset = v1.1 + `--muon --cuda_graph --batch_goal_enc --enc_hidden 1024,512,256
  --dec_hidden 1024,512,256`.
- `tools/make_report.py` compares all methods on the 3-minute checkpoint schedule, reports v1.2,
  extends the ablation ladder with the v1.2 steps, and marks runs shorter than a table column with "—".
- `eval_batch.py`, `eval_metrics.py`, `instanttrack.evaluate` and `export_onnx.py` read `pi_simba`
  from the checkpoint's `args`.
- Result figures renamed to `results/time_to_target.png` and `results/learning_curves.png` (after
  1.1.0), so that cached v1.0 images are not shown in place of newer ones.

### Tried and not adopted (single seed unless stated; compared with v1.1 re-run for 0.5 h with 1.5-minute checkpoints, seeds 0 / 1 / 2: 0.15 / 0.15 / 0.16 h)
| change | held ×3 | note |
|---|---|---|
| Muon on the policy only / on the world model only | 0.15 / 0.20 h | the gain needs both |
| spectral norm on the world model | 0.20 h | a smoother model alone does not help |
| SimBa residual policy 512×2 (2 seeds) | 0.20 / 0.15 h | 6.07 it/s |
| CUDA-graph updates without Muon | 0.15 h | +10 % it/s, same time to target |
| CUDA-graph updates + policy 1024-512-256 without Muon (2 seeds) | 0.15 / 0.15 h | 8.97 it/s, but more iterations |
| horizon 24 / 16 | 0.15 / 0.20 h | |
| 200 world-model-only iterations | 0.15 h | |
| lr 6e-4 (3 seeds) | 0.15 / 0.30 / 0.15 h | hyperparameter tuning |
| lr 6e-4 + horizon 24 / + failure-focused resets 0.5 | 0.20 / 0.15 h | |
| EMA 0.997 (seed 2, 0.75 h, 3-minute checkpoints; v1.1 seed 2: 0.25 h) | 0.40 h | |
| failure-focused resets 0.5 (seed 2, same protocol) | 0.15 h | tuning, one seed |

### Known limitations
- The smaller policy (1024-512-256) is a network-size choice (a hyperparameter), not a method
  contribution. With SuperMimic's policy size (2048-1024-512-256), Muon + CUDA graphs + batched goal
  encoding give 0.10 / 0.15 / 0.10 h on seeds 0–2 at 6.98 it/s (v1.2: 0.10 / 0.10 / 0.10 h at 8.74 it/s).
  That configuration is measured on 3 seeds only.
- Seed 4 holds the target only at 0.18 h: after first meeting it at 0.12 h, `jumps1_subject2`
  fails at the 0.15 h and 0.175 h checkpoints, and 3 re-evaluations of each confirm one failed clip.
- At 1.5 h (one run) v1.1 has slightly lower pose errors than v1.2 (body position 2.96 vs 3.01 cm,
  orientation 5.8° vs 6.0°, joints 3.76° vs 3.96°).
- Root drift at 1.5 h is 52 cm against 50 cm for SuperMimic + EMA (one run each).
- Under `--cuda_graph` Muon's learning rate is fixed at capture and does not follow the cosine
  schedule; in a 1.5 h run that is at most 8 % above the schedule.
- Accuracy at 0.5 h is measured on 3 of the 5 seeds, and the 1.5 h comparison is one run.
- Simulation only (Unitree G1, LAFAN1). No sim-to-real experiment.

### Compatibility
No breaking changes. v1.0 and v1.1 checkpoints and logs load and evaluate unchanged. To reproduce
v1.1 exactly, use the `instanttrack_v1.1` preset. `--cuda_graph` needs a CUDA GPU.

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

[Unreleased]: https://github.com/Huy-Quang-Dao/instantTrack/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/Huy-Quang-Dao/instantTrack/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Huy-Quang-Dao/instantTrack/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Huy-Quang-Dao/instantTrack/releases/tag/v1.0.0
