A second v1.1 run of seed 0 for 1.5 h (`e80_rres10_long_s0`) held the target from 0.20 h, and the 1.5 h v1.2 run of seed 0 (`e111_v12_long_s0`) from 0.12 h.

v1.2 runs were checkpointed every 1.5 minutes; all tables use only the checkpoints of the 3-minute protocol (every 0.05 h and every 2000 iterations). With every 1.5-minute checkpoint, v1.2 holds the target at 0.10 / 0.10 / 0.10 / 0.10 / 0.18 h and v1.1 (`e83`, `e106`, `e107`) at 0.15 / 0.15 / 0.16 h.

## Time to target (return ≥ 0.82 with 0/40 failed clips, held for 3 checkpoints)

| method | seeds | held at (h) | mean (h) | iteration at held | it/s | speed-up vs SuperMimic + EMA |
|---|---|---|---|---|---|---|
| SuperMimic | 1 | 1.50 | 1.50 | 16665 | 3.05 | 0.4× |
| SuperMimic + EMA | 3 | 0.50 / 0.54 / 0.60 | 0.55 | 5749 / 6000 / 6971 | 3.08 | 1.0× |
| InstantTrack v1.0 | 3 | 0.25 / 0.25 / 0.25 | 0.25 | 5900 / 6000 / 5956 | 6.32 | 2.2× |
| InstantTrack v1.1 | 5 | 0.15 / 0.16 / 0.25 / 0.20 / 0.20 | 0.19 | 3687 / 4000 / 5847 / 4868 / 4787 | 6.28 | 2.8× |
| InstantTrack v1.2 | 5 | 0.10 / 0.10 / 0.10 / 0.10 / 0.18 | 0.12 | 3322 / 3270 / 3336 / 3370 / 6000 | 8.74 | 4.7× |

## Equal training time (mean over seeds: normalized return / failed clips)

| method | 0.15 h | 0.25 h | 0.50 h | 0.75 h |
|---|---|---|---|---|
| SuperMimic | 0.545 / 12.0 | 0.646 / 10.0 | 0.775 / 2.0 | 0.757 / 5.0 |
| SuperMimic + EMA | 0.540 / 12.3 | 0.630 / 12.0 | 0.819 / 0.3 | 0.847 / 0.0 |
| InstantTrack v1.0 | 0.791 / 1.0 | 0.831 / 0.0 | 0.857 / 0.0 | 0.867 / 0.0 |
| InstantTrack v1.1 | 0.822 / 0.2 | 0.844 / 0.0 | 0.865 / 0.0 | 0.871 / 0.0 |
| InstantTrack v1.2 | 0.855 / 0.2 | 0.866 / 0.0 | 0.875 / 0.2 | — |

## Ablation ladder (each row adds one change to the row above)

| configuration | seeds | held at (h) | mean (h) | it/s | best return |
|---|---|---|---|---|---|
| SuperMimic (paper hyperparameters) | 1 | 1.50 | 1.50 | 3.05 | 0.863 |
| + EMA policy 0.995 | 3 | 0.50 / 0.54 / 0.60 | 0.55 | 3.08 | 0.879 / 0.850 / 0.854 |
| + lr 4e-4, 5 WM batches, no host syncs, fused Adam | 3 | 0.29 / 0.30 / 0.35 | 0.31 | 3.51 | 0.872 / 0.867 / 0.873 |
| + fused 3×3 rotations (`--fused_rot`) | 1 | 0.30 | 0.30 | 4.14 | 0.872 |
| + compiled reward / obs / goal (`--compile_more`) | 3 | 0.25 / 0.25 / 0.49 | 0.33 | 4.88 | 0.878 / 0.878 / 0.875 |
| + world model 1024³ (was 2048³) | 3 | 0.30 / 0.20 / 0.30 | 0.27 | 6.35 | 0.878 / 0.880 / 0.881 |
| + failure-focused resets (`--fail_reset 0.3`) | 3 | 0.20 / 0.20 / 0.25 | 0.22 | 6.29 | 0.875 / 0.874 / 0.874 |
| + contact reward 0.25 = **InstantTrack v1.0** | 3 | 0.25 / 0.25 / 0.25 | 0.25 | 6.32 | 0.869 / 0.869 / 0.864 |
| + reference-residual policy (`--ref_residual 1.0`) = **InstantTrack v1.1** | 5 | 0.15 / 0.16 / 0.25 / 0.20 / 0.20 | 0.19 | 6.28 | 0.873 / 0.873 / 0.871 / 0.874 / 0.873 |
| v1.1 again, 1.5-minute checkpoints, 0.5 h (reference for the rows below) | 3 | 0.15 / 0.15 / 0.16 | 0.15 | 6.29 | 0.863 / 0.862 / 0.863 |
| + Muon for hidden weight matrices (`--muon`) | 3 | 0.10 / 0.15 / 0.17 | 0.14 | 6.13 | 0.875 / 0.875 / 0.877 |
| + CUDA-graph updates, goals encoded once per rollout (`--cuda_graph --batch_goal_enc`) | 3 | 0.10 / 0.15 / 0.10 | 0.12 | 6.98 | 0.878 / 0.879 / 0.877 |
| + policy encoder / decoder 1024-512-256 = **InstantTrack v1.2** | 5 | 0.10 / 0.10 / 0.10 / 0.10 / 0.18 | 0.12 | 8.74 | 0.877 / 0.876 / 0.876 / 0.875 / 0.875 |

## Variants not adopted

| change | on top of | seeds | held at (h) | it/s |
|---|---|---|---|---|
| Smaller policy and WM (1024-512-256) | SuperMimic | 1 | 1.20 | 4.20 |
| Demonstration replay (DiffMimic) | SuperMimic + EMA | 1 | 0.59 | 3.18 |
| Policy batch 2048 | SuperMimic + EMA | 1 | 0.55 | 3.66 |
| EMA 0.999 | SuperMimic + EMA | 1 | 0.55 | 3.13 |
| lr 6e-4 | + lr 4e-4, wm5 | 1 | 0.40 | 3.27 |
| 3 WM batches | + lr 4e-4 | 1 | 0.45 | 3.93 |
| Horizon 24 | + lr 4e-4, wm5 | 1 | 0.30 | 4.31 |
| Horizon 16 | + compile_more | 1 | 0.30 | 7.02 |
| Foot-contact WM loss 0.05 (PhysMamba-style) | + lr 4e-4, wm5 | 1 | 0.35 | 3.30 |
| Anchored imagination (on-policy corrections) | + lr 4e-4, wm5 | 1 | 0.31 | 3.02 |
| 2 critic epochs | + lr 4e-4, wm5 | 1 | 0.40 | 3.81 |
| UTD 0.5 | + lr 4e-4, wm5 | 1 | 0.40 | 3.72 |
| CUDA graphs (`--cudagraphs`) | + compile_more | 1 | 0.25 | 5.11 |
| WM window 4 | + compile_more | 1 | 0.25 | 5.45 |
| Contact-mode world model (`--wm_contact 0.1`) | + fail_reset | 3 | 0.30 / 0.20 / 0.20 | 6.12 |
| Contact reward 0.5 | + fail_reset | 3 | 0.25 / 0.30 / 0.35 | 6.21 |
| Wide-kernel root reward 0.5, std 1 m (`--root_reward`) | v1.0 | 1 | 0.28 | 6.30 |
| Wide-kernel root reward 1.0, std 1 m | v1.0 | 1 | 0.35 | 6.28 |
| Wide-kernel root reward 0.5, std 0.5 m | v1.0 | 1 | 0.25 | 6.35 |
| Reference residual β = 0.5 (v1.1 uses 1.0; same seed: 0.15 h) | v1.0 | 1 | 0.17 | 6.20 |
| WM ensemble, 4 heads, disagreement penalty 100 (`--wm_ens`) | v1.1 | 1 | 0.25 | 6.12 |
| WM ensemble, 4 heads, disagreement penalty 300 | v1.1 | 1 | 0.35 | 6.09 |
| Mixture-of-experts policy, 2 decoders (`--pi_experts 2`) | v1.1 | 1 | 0.20 | 5.18 |
| EMA 0.997 (seed 2; v1.1 seed 2: 0.25 h) | v1.1 | 1 | 0.40 | 6.29 |
| Failure-focused resets 0.5 (seed 2; v1.1 seed 2: 0.25 h; tuning, one seed) | v1.1 | 1 | 0.15 | 6.36 |
| Horizon 24 | v1.1, 1.5-min ckpts | 1 | 0.15 | 7.40 |
| Horizon 16 | v1.1, 1.5-min ckpts | 1 | 0.20 | 8.99 |
| lr 6e-4 (tuning) | v1.1, 1.5-min ckpts | 3 | 0.15 / 0.30 / 0.15 | 6.30 |
| lr 6e-4 + horizon 24 | v1.1, 1.5-min ckpts | 1 | 0.20 | 7.49 |
| lr 6e-4 + failure-focused resets 0.5 | v1.1, 1.5-min ckpts | 1 | 0.15 | 6.38 |
| 200 world-model-only iterations (was 100) | v1.1, 1.5-min ckpts | 1 | 0.15 | 6.23 |
| CUDA-graph updates without Muon | v1.1, 1.5-min ckpts | 1 | 0.15 | 6.95 |
| CUDA-graph updates + policy 1024-512-256 without Muon | v1.1, 1.5-min ckpts | 2 | 0.15 / 0.15 | 8.97 |
| SimBa residual policy 512×2 (`--pi_simba`) | v1.1, 1.5-min ckpts | 2 | 0.20 / 0.15 | 6.07 |
| Muon on the policy only (`--muon_on pi`) | v1.1, 1.5-min ckpts | 1 | 0.15 | 6.32 |
| Muon on the world model only (`--muon_on wm`) | v1.1, 1.5-min ckpts | 1 | 0.20 | 6.09 |
| Spectral norm on the world model (`--wm_specnorm`) | v1.1, 1.5-min ckpts | 1 | 0.20 | 6.10 |
