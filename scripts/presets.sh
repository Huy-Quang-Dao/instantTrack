# Training presets (flags of `python -m instanttrack.train`). With no flags, train.py is the SuperMimic
# baseline; every change is a flag, so all methods share simulator, MDP and evaluator.
# The presets of InstantTrack v1.0-v1.2 use flags of the earlier code base: check out tag v1.2.0 to run them.
declare -A PRESET=(
  # SuperMimic (paper hyperparameters, our mjlab reimplementation)
  [supermimic]=""
  # SuperMimic + Polyak-averaged policy (EMA 0.995): the fair baseline (EMA alone gives ~3x)
  [supermimic_ema]="--pi_ema 0.995"
  # InstantTrack v1.4 (main configuration)
  #   v1.2 base: EMA policy, lr 4e-4, 5 world-model batches, no host syncs, fused Adam / rotations, compiled
  #     reward / obs / goal, world model 1024^3, failure-weighted resets, contact reward, Muon, CUDA-graph updates
  #   from BlinkMimic: structured world model with mirror augmentation, Muon on the policy and the world model
  #     (--pi_opt/--wm_opt muon), reference-action skip, evaluation-guided reset allocation (--eprobe_*), policy
  #     batch 2048 with a 4x policy learning rate decayed to 1x once the training return reaches 0.75, CUDA graphs
  #     for the collection step and the probe sweep
  #   new in v1.3: static-shape compilation, collection and world-model updates overlapped with the policy update
  #     on side CUDA streams, policy learning-rate annealing after the target return
  #   new in v1.4: goals of all imagined steps encoded in one batch, critic updates on a side CUDA stream, evaluation
  #     sweep on a side stream alongside training and limited to 372 worlds that visit failing tiles first
  [instanttrack]="--pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 --fail_reset 0.3 --contact_reward 0.25 --wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon --pi_ref_skip --wm_only_iters 0 --bin_fix --ref_contact_fix --cuda_graph --eprobe_share 0.15 --graph_early --collect_graph --reset_masked --pi_batch 2048 --pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300 --ckpt_light --eprobe_graph --eprobe_async 10 --eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7 --lr_decay_by 1400 --static_compile --overlap_collect --overlap_wm --pi_anneal_len 3000 --batch_goal_enc --overlap_critic --eprobe_active 372"
)
