# control/ — Feedforward PPO & Shared Environment

## Contents
- `gpu_env.py` — GPU-vectorized BatchedDroneEnv (used by all trainers)
- `train_gpu.py` — Original feedforward PPO training loop
- `dynamics.py` — PyTorch batched quadcopter dynamics (17-state)
- `plot_training.py` — Chart generation from training CSVs
- `save_run.py` — Utility to copy training runs to saved/

## Experiments Run (Feedforward PPO)

### Experiments 1-5: SB3 Feedforward PPO (Mar 8)
- 3x64 or 3x128 MLP, ~56K FPS
- Best result: ~8% >=1 gate (multi-track, Exp #2)
- Gate passage fundamentally limited without memory — policy can't track gate sequence
- Attention (GLAM) added complexity without benefit

### Experiment 6: Custom GPU PPO Loop (Mar 9)
- Moved from SB3 to custom loop for more control
- 3x128 MLP, escalating gate rewards
- Peaked at ~15% >=1 at 18M steps, then declined due to linear LR decay to 0

## Key Observations
- **Feedforward policies cap at ~5-15% >=1 gate** — without recurrence, the policy cannot track which gate to head toward next
- ~56K FPS — fastest of all architectures (no recurrent overhead)
- Linear LR decay to 0 kills learning — discovered here, fixed in later experiments with cosine LR
- Escalating gate rewards (lambda_gate_base + lambda_gate_inc * gates_passed) helped somewhat
- This architecture was abandoned in favor of recurrent models after Exp #7 proved LSTM breakthrough

## gpu_env.py — Shared Environment Notes
- Used by all trainers (RPPO, Mamba, FSPPO)
- Configurable reward lambdas: lambda_prog, lambda_gate, lambda_gate_inc, lambda_rate
- Yaw jitter: currently ±90° (was ±5° initially, then ±30°, now ±90°)
- Roll/pitch jitter: ±20°
- Per-segment tracking for identifying hard gate transitions
- Gate size curriculum: conditional on --gate-size flag (skipped when fixed size provided)
- DT_SIM = 0.01 (100Hz control)
