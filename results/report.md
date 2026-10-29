## Time to target (return ≥ 0.82 with 0/40 failed clips, held for 3 checkpoints)

| method | seeds | held at (h) | mean (h) | iteration at held | it/s | speed-up vs SuperMimic + EMA |
|---|---|---|---|---|---|---|
| SuperMimic | 1 | 1.50 | 1.50 | 16665 | 3.05 | 0.4× |
| SuperMimic + EMA | 3 | 0.50 / 0.54 / 0.60 | 0.55 | 5749 / 6000 / 6971 | 3.08 | 1.0× |
| InstantTrack | 3 | 0.25 / 0.25 / 0.25 | 0.25 | 5900 / 6000 / 5956 | 6.32 | 2.2× |

## Equal training time (mean over seeds: normalized return / failed clips)

| method | 0.15 h | 0.25 h | 0.50 h | 0.75 h |
|---|---|---|---|---|
| SuperMimic | 0.545 / 12.0 | 0.646 / 10.0 | 0.775 / 2.0 | 0.757 / 5.0 |
| SuperMimic + EMA | 0.540 / 12.3 | 0.630 / 12.0 | 0.819 / 0.3 | 0.847 / 0.0 |
| InstantTrack | 0.791 / 1.0 | 0.831 / 0.0 | 0.857 / 0.0 | 0.867 / 0.0 |

## Ablation ladder (each row adds one change to the row above)

| configuration | seeds | held at (h) | mean (h) | it/s | best return (0.75 h) |
|---|---|---|---|---|---|
| SuperMimic (paper hyperparameters) | 1 | 1.50 | 1.50 | 3.05 | 0.863 |
| + EMA policy 0.995 | 3 | 0.50 / 0.54 / 0.60 | 0.55 | 3.08 | 0.879 / 0.850 / 0.854 |
| + lr 4e-4, 5 WM batches, no host syncs, fused Adam | 3 | 0.29 / 0.30 / 0.35 | 0.31 | 3.51 | 0.872 / 0.867 / 0.873 |
| + fused 3×3 rotations (`--fused_rot`) | 1 | 0.30 | 0.30 | 4.14 | 0.872 |
| + compiled reward / obs / goal (`--compile_more`) | 3 | 0.25 / 0.25 / 0.49 | 0.33 | 4.88 | 0.878 / 0.878 / 0.875 |
| + world model 1024³ (was 2048³) | 3 | 0.30 / 0.20 / 0.30 | 0.27 | 6.35 | 0.878 / 0.880 / 0.881 |
| + failure-focused resets (`--fail_reset 0.3`) | 3 | 0.20 / 0.20 / 0.25 | 0.22 | 6.29 | 0.875 / 0.874 / 0.874 |
| + contact reward 0.25 = **InstantTrack** | 3 | 0.25 / 0.25 / 0.25 | 0.25 | 6.32 | 0.869 / 0.869 / 0.864 |

## Variants without a time-to-target gain

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
