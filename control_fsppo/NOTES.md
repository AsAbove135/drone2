# control_fsppo/ — Frame-Stacked Feedforward PPO

## Contents
- `train_gpu_fsppo.py` — Frame-stacked PPO training loop with feedforward MLP

## Architecture
- Feedforward MLP: shared encoder (input → 256), separate actor/critic branches (256 → 256 → output)
- Input: last N observations concatenated (default N=16, configurable via --frame-stack)
- Input dim = OBS_DIM * frame_stack (e.g., 25 * 16 = 400)
- Standard minibatch PPO (no sequence-aware batching needed)
- Expected FPS: ~50-70K (no recurrent overhead)

## Motivation
- RPPO trains at ~27K FPS, limited by LSTM forward/backward passes
- Frame stacking provides temporal context without recurrence
- At 100Hz control, 16 frames = 160ms of history — enough to capture turns
- 2-3x faster training enables rapid hyperparameter/reward iteration
- Tradeoff: fixed temporal window vs LSTM's unbounded memory

## Experiments Run
*No experiments run yet — trainer created Mar 12.*

## Planned Experiments
- Baseline: frame_stack=16, same reward/env config as RPPO Exp #22
- Frame stack sweep: 8, 16, 32, 64 frames
- Reward function experiments (angular acceleration penalty, etc.)
- Observation ablation: which features matter most?

## Key Questions to Answer
1. Does frame-stacking match RPPO performance within ~90%? If so, the faster iteration is worth it.
2. What's the optimal frame stack depth? Too few = not enough context, too many = large input, slow.
3. Can the policy learn multi-gate navigation without recurrence? The obs includes gate_idx and dist_gate, so explicit memory of past gates may not be needed.
