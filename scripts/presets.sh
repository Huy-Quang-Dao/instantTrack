# Training presets (flags of `python -m instanttrack.train`). With no flags, train.py is the SuperMimic
# baseline; every InstantTrack change is a flag, so all methods share simulator, MDP and evaluator.
declare -A PRESET=(
  # SuperMimic (paper hyperparameters, our mjlab reimplementation)
  [supermimic]=""
  # SuperMimic + Polyak-averaged policy (EMA 0.995): the fair baseline (EMA alone gives ~3x)
  [supermimic_ema]="--pi_ema 0.995"
  # InstantTrack without the contact reward: best normalized return / pose accuracy
  [instanttrack_track]="--pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 --fail_reset 0.3"
  # InstantTrack (main configuration): + contact-matching / anti-slip reward in imagination
  [instanttrack]="--pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 --fail_reset 0.3 --contact_reward 0.25"
)
