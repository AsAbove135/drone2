# control_mamba/ — Mamba SSM Policy

## Contents
- `train_gpu_mamba.py` — Mamba training loop with SSM actor-critic, sequence-aware minibatching

## Architecture
- Mamba SSM: d_model=64, d_state=16, d_inner=128
- 75K parameters (1/3 of RPPO's 217K)
- Separate actor/critic Mamba blocks
- ~4-5K FPS (without custom CUDA kernels)
- Matches RPPO per-step learning efficiency with far fewer parameters

## Experiments Run

### Experiment 8: Mamba Baseline (Mar 10)
- Same config as RPPO Exp #7: n_steps=512, seq=16, lambda_gate=15
- Matched RPPO within 2-3% at same step counts
- Killed at 36.7M to try aggressive config (~64% >=1)

### Experiment 9: Aggressive seq=128 (Mar 10)
- Hypothesis: longer BPTT helps multi-gate credit assignment
- n_steps=256, seq=128, batch=64
- **Result: 22.3% >=1 at 18.4M — significantly behind RPPO pace**
- Fewer gradient updates per iteration hurt early learning

### Experiments 10-11: Cosine LR A/B Test (Mar 10)
- First runs with cosine LR warm restarts, doubled gate rewards (lambda_gate=30, inc=20)
- **Seq64 (#11): 92.0% >=1, 35.9% >=2, avg 1.49** — nearly matched RPPO
- **Seq128 (#10): 85.2% >=1, 33.6% >=2, avg 1.42** — behind seq64
- Cosine LR kept both learning through 100M (no plateau like RPPO linear decay)

### Experiments 12-13: Bootstrap Failures (Mar 10)
- Attempted resume from #10/#11 with halved lambda_prog and ±30° yaw
- **Both collapsed**: entropy blew up, PG loss collapsed
- #12: Loaded optimizer state — Adam momentum shock
- #13: Fresh optimizer — policy couldn't unlearn "fly straight"
- **Lesson: bootstrapping across major task/reward changes doesn't work**

### Experiment 14: Best Mamba — seq64 + ±30° Yaw (Mar 11)
- Fresh training with ±30° yaw, ±20° roll/pitch
- lambda_prog=10, lambda_gate=30, lambda_gate_inc=20, cosine LR
- **Result: 86.0% >=1, 42.2% >=2, 18.0% >=3, avg 1.53 gates**
- Yaw jitter improved multi-gate: 42% vs 36% >=2 without jitter
- Per-segment analysis: Seg 4 (gate 3→4) = 0.67 avg gates (hardest)

### Experiment 15: seq128 + ±30° Yaw (Mar 11)
- Same config as #14 but seq=128
- **Result: 83.5% >=1, 32.9% >=2 — behind seq64 everywhere**
- Extra BPTT not worth fewer gradient updates

### Experiment 16: seq16 + ±30° Yaw (Mar 11)
- **Result: 88.6% >=1, 42.9% >=2, 13.1% >=3, avg 1.46**
- Best >=1 and >=2 of all Mamba, but worse on >=3/>=4 than seq64
- Entropy rose to ~7.0 over training — broader policy, less precise

### Experiment 17: seq32 + ±30° Yaw (Mar 11)
- **Result: 38.8% >=1, 12.5% >=2 — anomalously bad**
- Worst of all seq_lens, possible "bad middle ground"
- Neither enough gradient updates (like seq16) nor enough BPTT (like seq64)

## Seq_len Sweep Summary (all ±30° yaw, 100M steps)
| seq_len | >=1 | >=2 | >=3 | Avg Gates |
|---------|-----|-----|-----|-----------|
| 16 | 88.6% | 42.9% | 13.1% | 1.46 |
| 32 | 38.8% | 12.5% | 5.3% | 0.57 |
| 64 | 86.0% | 42.2% | 18.0% | 1.53 |
| 128 | 83.5% | 32.9% | 13.4% | 1.33 |

## Key Findings
1. **Mamba matches RPPO with 1/3 params** — 75K vs 217K, same learning curve per step
2. **seq64 best overall for Mamba** — best on >=3/>=4; seq16 best on >=1/>=2
3. **seq32 is anomalously bad** — avoid this middle ground
4. **4-5K FPS** — 5x slower than RPPO due to lack of custom CUDA kernels
5. **Yaw jitter improves multi-gate** — same finding as RPPO
6. **Bootstrapping across task changes fails** — confirmed independently from RPPO
7. **Not yet tested with ±90° yaw** — would be interesting to compare with RPPO ±90° results
