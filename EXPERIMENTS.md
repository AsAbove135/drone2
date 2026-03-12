# MonoRace Drone Racing — Experiment Log

## Summary Table

| # | Date | Model | Steps | Best >=1 | Best >=2 | Best >=3 | Avg Gates | Key Config | Outcome |
|---|------|-------|-------|----------|----------|----------|-----------|------------|---------|
| 1 | Mar 8 | FF PPO (SB3) | 50M | ~5% | <1% | 0% | ~0.05 | expert track, 3x64, ent=0.01 | Baseline — learns to fly, rarely passes gates |
| 2 | Mar 8 | FF PPO (SB3) | 50M | ~8% | <1% | 0% | ~0.08 | multitrack (kidney+fig8+expert) | Slightly better, but multitrack didn't help much |
| 3 | Mar 8 | FF PPO (SB3) | 50M | ~3% | 0% | 0% | ~0.03 | easy track, 3x64, scaling+GLAM | Worse — attention mechanism didn't help |
| 4 | Mar 8 | FF PPO (SB3) | 2M | - | - | - | - | kidney, 3x128, old curriculum | Too short to learn anything meaningful |
| 5 | Mar 8 | FF PPO (SB3) | 18M | ~5% | <1% | 0% | ~0.05 | kidney, 3-phase curriculum | Failed — curriculum transitions too abrupt |
| 6 | Mar 9 | FF PPO (GPU) | 50M | ~15% | ~1% | 0% | ~0.15 | kidney, 3x128, escalating gates, ent=0.01 | Peaked at 18M then declined — LR decay issue |
| 7 | Mar 10 | **RPPO (LSTM)** | **100M** | **93.5%** | **65.7%** | **29.0%** | **2.05** | n_steps=512, seq=16, lambda_gate=15, inc=10, linear LR | **Best run.** Recurrence breakthrough. LR hit 0 and learning stopped |
| 8 | Mar 10 | Mamba (baseline) | 36.7M | ~64% | ~2% | 0% | ~0.64 | n_steps=512, seq=16, lambda_gate=15 | Matched RPPO step-for-step, killed early |
| 9 | Mar 10 | Mamba (aggressive) | 18.4M | 22.3% | 0.26% | 0% | 0.23 | n_steps=256, seq=128, batch=64, 4 epochs, lambda_gate=15 | Slower learning — long BPTT didn't help early. Killed |
| 10 | Mar 10 | Mamba seq128+cosine | 100M | 85.2% | 33.6% | 16.1% | 1.42 | n_steps=256, seq=128, cosine LR, lambda_gate=30, inc=20 | Solid but behind seq64 on >=1, ahead on >=3/>=4 |
| 11 | Mar 10 | Mamba seq64+cosine | 100M | 92.0% | 35.9% | 16.0% | 1.49 | n_steps=256, seq=64, cosine LR, lambda_gate=30, inc=20 | Nearly matched RPPO on >=1 (92% vs 93.5%) |
| 12 | Mar 10 | Mamba bootstrap (failed) | 120M | 10.5% | 0.5% | 0% | 0.10 | Resumed from #10/#11, lambda_prog=5, ±30° yaw | Collapsed — optimizer state + reward shift + yaw jitter too much |
| 13 | Mar 10 | Mamba bootstrap v2 (failed) | 120M | 9.6% | 0.4% | 0% | 0.10 | Fresh optimizer, same config as #12 | Entropy blew up to near-random, policy couldn't unlearn old behavior |
| 14 | Mar 11 | **Mamba seq64+yaw** | **100M** | **86.0%** | **42.2%** | **18.0%** | **1.53** | n_steps=256, seq=64, cosine LR, ±30° yaw, ±20° roll/pitch | **Best Mamba.** Yaw jitter improved multi-gate (42% vs 36% >=2) |
| 16 | Mar 11 | Mamba seq16+yaw | 100M | 88.6% | 42.9% | 13.1% | 1.46 | n_steps=256, seq=16, cosine LR, ±30° yaw, ±20° roll/pitch | Best >=1 and >=2, but worse on >=3/>=4 than seq64 |
| 17 | Mar 11 | Mamba seq32+yaw | 100M | 38.8% | 12.5% | 5.3% | 0.57 | n_steps=256, seq=32, cosine LR, ±30° yaw, ±20° roll/pitch | Anomalously bad — worst of all seq_lens. Possible bad middle ground |
| 15 | Mar 11 | Mamba seq128+yaw | 100M | 83.5% | 32.9% | 13.4% | 1.33 | n_steps=256, seq=128, cosine LR, ±30° yaw, ±20° roll/pitch | Behind seq64 everywhere — extra BPTT not worth the cost |
| 18 | Mar 11 | **RPPO ±90° yaw** | **100M** | **91.9%** | **73.8%** | **47.4%** | **2.83** | n_steps=256, seq=16, batch=256, ±90° yaw, fixed 1.5m gates, cosine LR | **New best at 100M.** ±90° yaw massively improved multi-gate |
| 19 | Mar 11 | RPPO cost-to-go | 100M | 87.8% | 46.5% | 20.2% | 1.66 | Same as #18 + heading-aware r_prog (alpha=0.5) | Cost-to-go shaping hurt — simpler distance reward wins |
| 20 | Mar 11 | **RPPO ±90° resume52M** | **147M** | **91.5%** | **80.6%** | **65.0%** | **4.06** | Resumed #18 from 52M checkpoint with optimizer state | **Best median.** 50% >=4, 5.3% >=10 (half lap) |
| 21 | Mar 11 | **RPPO ±90° resume100M** | **200M** | **88.6%** | **74.3%** | **56.3%** | **3.97** | Resumed #18 from 100M with fresh optimizer | **Best tail.** 8.2% >=10, 0.1% >=20 (2 full laps!) |
| 22 | Mar 12 | **RPPO ±90° 500M** | **500M** | **74.3%** | **59.0%** | **52.7%** | **5.82** | n_steps=256, seq=16, batch=256, from scratch | **Best avg gates (5.82).** 29% full laps, 2.3% >=20. Entropy rose to 6.86 |
| 23 | Mar 12 | RPPO ±90° 512/64 | 500M | 78.1% | 59.4% | 42.2% | 2.82 | n_steps=512, seq=64, batch=512, from scratch | Longer BPTT hurt — fewer gradient steps. seq16 confirmed better for RPPO |
| 24 | Mar 12 | FSPPO fs4 | 100M | 91.1% | 80.1% | 59.5% | 3.77 | frame_stack=4, 66K FPS | Eval: kidney 8.0 avg (caps at gate 8) |
| 25 | Mar 12 | FSPPO fs8 | 100M | 91.3% | 79.1% | 60.0% | 3.65 | frame_stack=8, 64K FPS | Eval: kidney 14.3 avg (bimodal 8 or 19) |
| 26 | Mar 12 | **FSPPO fs16** | **100M** | **90.5%** | **79.3%** | **63.2%** | **4.21** | frame_stack=16, 65K FPS | **Eval: kidney 19.0 avg (10/10 hit 19 gates)** |

| 27 | Mar 12 | **FSPPO fs16 multi** | **200M** | - | - | - | - | frame_stack=16, kidney+fig8 | **10/10 both tracks. Kidney 8.61s, Fig8 11.64s** |
| 28 | Mar 12 | FSPPO fs64 multi | 200M | - | - | - | - | frame_stack=64, kidney+fig8 | 10/10 both tracks. Kidney 9.36s, Fig8 11.83s |

## Per-Segment Analysis

### Experiment 20 (RPPO ±90° yaw, 147M steps) — Current Best
```
Seg 0 (start→gate 0):  4.27
Seg 1 (gate 0→1):      3.48
Seg 2 (gate 1→2):      2.70
Seg 3 (gate 2→3):      2.57
Seg 4 (gate 3→4):      2.72  ← was 0.67 in Exp 14!
Seg 5 (gate 4→5):      3.50
Seg 6 (gate 5→6):      3.35
Seg 7 (gate 6→7):      5.68
Seg 8 (gate 7→8):      5.34  ← was 0.86 in Exp 14!
Seg 9 (gate 8→9):      4.92
```
±90° yaw jitter eliminated the previous bottlenecks. Seg 4 improved 4x (0.67→2.72), Seg 8 improved 6x (0.86→5.34).

### Experiment 14 (Mamba ±30° yaw, 100M steps) — Previous Best
```
Seg 0 (start→gate 0):  3.20 avg gates  ← easiest
Seg 1 (gate 0→1):      2.70
Seg 2 (gate 1→2):      1.88
Seg 3 (gate 2→3):      1.01
Seg 4 (gate 3→4):      0.67  ← hardest
Seg 5 (gate 4→5):      1.33
Seg 6 (gate 5→6):      1.16
Seg 7 (gate 6→7):      1.44
Seg 8 (gate 7→8):      0.86
Seg 9 (gate 8→9):      1.11
```

---

## Detailed Notes

### Experiment 1-5: Feedforward PPO (SB3)
- Used Stable Baselines 3 with feedforward 3x64 or 3x128 MLP
- ~56K FPS on GPU, 50M steps in ~15 min
- Gate passage fundamentally limited without memory — policy can't track which gate it's heading to
- Escalating gate rewards (lambda_gate_base + lambda_gate_inc × gates_passed) helped somewhat
- Attention (GLAM) added complexity without benefit

### Experiment 6: GPU Feedforward PPO (Custom Loop)
- Moved from SB3 to custom PPO loop for more control
- Peaked at ~18M steps then reward declined
- Root cause: linear LR decay to 0 killed learning

### Experiment 7: RPPO — The Breakthrough
- **Config**: LSTM 128 hidden, separate actor/critic, 217K params
- **Training**: n_steps=512, seq_len=16, lr=3e-4 linear decay, ent_coef=0.005
- **Rewards**: lambda_prog=10, lambda_gate=15, lambda_gate_inc=10, gate_size 1.5→0.5m curriculum
- **Result**: 93.5% >=1 gate, 65.7% >=2, 29.0% >=3, 12.3% >=4, 3.6% >=5, 0.37% >=6
- Recurrence was the key — LSTM state lets policy track gate sequence and flight history
- Learning plateaued after ~70M steps as LR decayed to near 0
- **Saved to**: `D:/drone2_training/saved/2026-03-10_08-05_rppo_100M_escalating_gates_gate_curriculum/`

### Experiment 8: Mamba Baseline
- **Config**: Mamba SSM, d_model=64, d_state=16, d_inner=128, 75K params (1/3 of RPPO)
- Same hyperparameters as RPPO experiment 7
- Matched RPPO within 2-3% on all metrics at same step counts
- Killed at 36.7M to try aggressive config

### Experiment 9: Mamba Aggressive (seq_len=128)
- Hypothesis: longer BPTT (128 vs 16) would help multi-gate credit assignment
- Reality: fewer gradient updates per iteration hurt early learning
- 22.3% >=1 at 18.4M steps vs RPPO's 37% at 15.7M — significantly behind
- Killed to try with cosine LR + doubled gate rewards

### Experiments 10-11: Mamba Cosine LR A/B Test (100M)
- First runs with cosine LR warm restarts (3e-4 ↔ 1e-5, 50M cycle)
- Doubled gate reward (lambda_gate=30, lambda_gate_inc=20)
- Seq64 nearly matched RPPO on >=1 gate (92% vs 93.5%)
- Seq128 slightly better on >=3/>=4 but worse on >=1 — not worth the FPS cost
- Cosine LR kept both learning through 100M (no plateau like RPPO)
- **Saved to**: `D:/drone2_training/saved/2026-03-10_mamba_seq128_100M_cosLR_gate30/` and `mamba_seq64`

### Experiments 12-13: Bootstrap Failures
- Attempted to resume from #10/#11 with lambda_prog halved (10→5) and ±30° yaw jitter
- **Exp 12**: Loaded optimizer state — Adam momentum from old reward scale caused entropy blowup, PG loss collapsed to near 0
- **Exp 13**: Fresh optimizer — better initial response but same entropy blowup pattern. Policy couldn't unlearn "fly straight" to learn "turn then fly"
- **Lesson**: Bootstrapping across major task/reward changes doesn't work well. Better to train from scratch

### Experiments 14-15: Fresh Training with Yaw Jitter
- Trained from scratch with ±30° yaw jitter and ±20° roll/pitch (closer to paper's initialization)
- lambda_prog=10 (kept strong), lambda_gate=30, lambda_gate_inc=20, cosine LR
- **Seq64 (#14) beat all previous Mamba runs on multi-gate** — 42.2% >=2 vs 35.9% without jitter
- Yaw jitter forces the policy to learn turning, which transfers to better gate-to-gate transitions
- Seq128 (#15) behind seq64 everywhere — extra BPTT length not worth the fewer gradient updates
- Per-segment logging reveals Seg 4 (gate 3→4) as the hardest transition
- **Saved to**: `D:/drone2_training/saved/2026-03-11_mamba_seq64_100M_yawjitter_cosLR_gate30/` and `mamba_seq128`

### Experiment 18: RPPO ±90° Yaw — New Best at 100M
- **Config**: LSTM 128 hidden, n_steps=256, seq=16, batch=256, 4096 envs, 4 epochs
- **Spawn**: ±90° yaw jitter, ±20° roll/pitch, fixed 1.5m gates (no curriculum)
- **Rewards**: lambda_prog=10, lambda_gate=30, lambda_gate_inc=20, cosine LR
- **Result**: 91.9% >=1, 73.8% >=2, 47.4% >=3, 30.2% >=4, 18.4% >=5, 11.8% >=6
- FPS ~19-27K (varies with GPU contention), 100M in ~1hr
- ±90° yaw forces robust turning — massive improvement on multi-gate vs Exp #7 (±5° yaw)
- **Saved to**: `D:/drone2_training/saved/2026-03-11_rppo_yaw90_fixed15_100M_best/`

### Experiment 19: RPPO Cost-to-Go Shaping (Failed)
- Same config as #18 but r_prog uses heading-aware cost: `cost = dist * (1 + 0.5 * (1 - cos_angle))`
- Hypothesis: rewarding heading alignment would help turning
- Result: 87.8% >=1, 46.5% >=2, 20.2% >=3 — significantly worse than baseline
- Lower entropy (5.06 vs 5.43) suggests policy converged to narrow local optimum
- Heading term added noise at low speeds (spawn with near-zero velocity makes cos_angle meaningless)
- **Lesson**: Simpler distance-based progress reward wins. Don't over-shape.

### Experiments 20-21: RPPO ±90° Extended Training (200M)
- Resumed Exp #18 with two strategies:
  - **#20**: From 52M checkpoint (optimizer state intact), ran to 147M
  - **#21**: From 100M final (fresh optimizer), ran to 200M
- **#20 wins on median**: 91.5% >=1, 80.6% >=2, 65.0% >=3, 50.3% >=4 — optimizer continuity helps
- **#21 wins on tail**: 8.2% >=10, 1.7% >=15, 0.1% >=20 (2 full laps!) — higher entropy explores more
- Cosine LR warm restarts kept both learning well past 100M — no plateau
- Per-segment analysis shows ±90° yaw eliminated previous bottlenecks (Seg 4: 0.67→2.72)
- **Saved to**: `D:/drone2_training/saved/2026-03-11_rppo_yaw90_resume52M_147M/` and `rppo_yaw90_resume100M_200M`

---

### Experiments 22-23: RPPO 500M — Seq Length Comparison
- Both from scratch with ±90° yaw, fixed 1.5m gates, cosine LR, 4096 envs
- **#22 (256/16/256)**: avg 5.82 gates, 29% full laps (>=10), 2.3% >=20 (2 laps). Entropy rose to 6.86 — very exploratory
- **#23 (512/64/512)**: avg 2.82 gates, 4.1% full laps. Fewer gradient steps (256/epoch vs 1024) hurt despite better GAE
- Confirms seq_len=16 > seq_len=64 for RPPO — more gradient updates wins over longer BPTT
- 500M training shows continued improvement but entropy drift — may benefit from higher ent_coef or entropy cap
- **Saved to**: `D:/drone2_training/saved/2026-03-12_rppo_yaw90_500M_baseline_256_16_256/` and `rppo_yaw90_500M_longbptt_512_64_512`

---

## Key Insights

1. **Frame stacking rivals recurrence** — FSPPO with 16 frames matches RPPO training metrics and dominates on eval (19/20 gates consistently vs untested RPPO)
2. **LR decay kills learning** — linear decay to 0 stops improvement. Cosine warm restarts keep the policy learning
3. **Gate reward is sparse** — r_prog dominates r_gate by 4-20x depending on training stage. Doubled lambda_gate (30) helps
4. **Escalating gate rewards help** — increasing reward per successive gate encourages multi-gate runs
5. **seq_len=16 is optimal for both RPPO and Mamba** — more gradient updates per rollout beats longer BPTT consistently
6. **Mamba matches RPPO with 1/3 the parameters** — 75K vs 217K, same learning curve
7. **±90° yaw jitter is transformative** — forces robust turning, eliminated segment bottlenecks, enabled 2-lap completions
8. **Bootstrapping same-task works, cross-task fails** — resuming with same reward/config + optimizer state gives best results
9. **Optimizer continuity vs fresh optimizer**: intact optimizer → better median; fresh optimizer → better tail (more exploration)
10. **Cost-to-go heading shaping hurts** — heading-aware r_prog adds noise at low speeds. Simpler distance reward wins
11. **500M training keeps improving** — avg gates: 2.83 (100M) → 4.06 (147M) → 5.82 (500M). No plateau with cosine LR
12. **Entropy drift at 500M** — policy becomes very exploratory (entropy 5.4→6.9), >=1 drops (92%→74%) while multi-gate improves
13. **FPS**: feedforward ~56K, RPPO ~19-27K (batch=256), Mamba ~4K (without custom CUDA kernels)
14. **Batch size 256 >> 64 for RPPO** — stabler gradients, fewer but higher-quality updates, 3x FPS improvement
15. **Training metrics can be misleading** — FSPPO fs4/fs8/fs16 all show ~91% >=1 in training, but eval reveals 8 vs 14 vs 19 avg gates. Always eval deterministically
16. **FSPPO FPS ~65K** — 2.4x faster than RPPO (~27K), enabling rapid iteration on reward/hyperparams

### Experiments 24-26: Frame-Stacked PPO — Frame Stack Sweep (100M)
- **Architecture**: Feedforward MLP with concatenated last N observations (no recurrence)
  - Shared encoder (input → 256), separate actor/critic branches (256 → 256 → output)
- **Config**: n_steps=256, 4096 envs, 4 epochs, cosine LR, ±90° yaw, fixed 1.5m gates
- **Rewards**: lambda_prog=10, lambda_gate=30, lambda_gate_inc=20, lambda_rate=0.001
- All three ran in parallel (~26 min total)

| Exp | Frames | Input Dim | Params | FPS | Train >=1 | Train Avg | **Eval Kidney Avg** | **Eval Kidney Max** | **Eval Fig8** |
|-----|--------|-----------|--------|-----|-----------|-----------|---------------------|---------------------|---------------|
| 24 | 4 | 96 | 289K | 66K | 91.1% | 3.77 | 8.0 | 8 | 0.0 |
| 25 | 8 | 192 | 314K | 64K | 91.3% | 3.65 | 14.3 | 19 | 0.0 |
| 26 | 16 | 384 | 363K | 65K | 90.5% | 4.21 | **19.0** | **19** | 0.5 |

**Key findings:**
- **FSPPO matches or beats RPPO** on training metrics at 100M, at 2.4x the speed
- **16 frames dominant on eval** — 19/20 gates consistently (10/10 episodes), nearly 2 full laps
- 4-frame models all crash at gate 8-9 boundary (not enough temporal context for that turn)
- 8-frame bimodal: 60% reach 18-19 gates, 40% crash at gate 8
- **Zero generalization to figure-8** — expected since trained only on kidney
- Training metrics similar across all three, but eval reveals huge differences in robustness

---

## Evaluation System
- **Unified evaluator**: `python -m evaluation.evaluate --checkpoint <path> --model-type <rppo|mamba|fsppo> --tracks kidney figure8 --episodes 10`
- Eval results saved to `evaluation/results/<name>/` with trajectory plots and `results.json`
- All models evaluated deterministically (mean action, no sampling) on 10 episodes from starting line

### Experiments 27-28: Multi-Track Training (200M)
- **Architecture**: Same FSPPO as experiments 24-26, but trained on kidney AND figure-8 simultaneously
- **Multi-track method**: Each env randomly assigned kidney or figure-8 on reset. Per-env gate tensors `[N, G, 3]`.
- **Config**: n_steps=256, 4096 envs, 4 epochs, cosine LR, ±90° yaw, fixed 1.5m gates, 200M steps

| Exp | Frames | Params | **Kidney Completions** | **Kidney Best** | **Fig8 Completions** | **Fig8 Best** |
|-----|--------|--------|------------------------|-----------------|----------------------|---------------|
| 27 | 16 | 363K | 10/10 | 8.61s | 10/10 | 11.64s |
| 28 | 64 | 658K | 10/10 | 9.36s | 10/10 | 11.83s |

**Key findings:**
- **Multi-track training solves generalization**: 0% → 100% completion on figure-8 (was 0% when trained on kidney only)
- **fs16 beats fs64 on speed**: 8.61s vs 9.36s on kidney, 11.64s vs 11.83s on figure-8, despite fewer params
- **Multi-track fs16 is faster on kidney than single-track fs16**: 8.61s vs 9.44s — learning diverse tracks improves overall racing
- **Figure-8 takes ~35% longer than kidney**: 11.7s vs 8.6s — larger track with tighter crossing gates
- **Single-track models have zero generalization** to unseen tracks

---

## Next Steps
- **Image-based training**: Explore end-to-end image RL with CNN encoder (Isaac Sim, Flightmare, or lightweight renderer)
- Run FSPPO fs16 multi-track for 500M+ steps to push lap times lower
- Try more tracks (easy, medium, hard, expert) for broader generalization
- Gate size curriculum: shrink gates after mastering 1.5m
- Reward function sweep using FSPPO (fast iteration): angular rate penalty, alignment bonus, etc.
