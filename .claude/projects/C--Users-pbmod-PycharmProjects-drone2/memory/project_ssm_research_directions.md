---
name: SSM and model architecture research directions
description: Prioritized list of SSM/recurrent architectures to test for drone racing RL policy, with implementation status and rationale
type: project
---

## SSM Architecture Research Directions (as of 2026-03-14)

### Tier 1: Ready to Test Now

**Mamba-1 with CUDA kernels**
- `mamba_fast.py` wrapper built, drops into existing `MambaActorCritic`
- Expected speedup: 4K → ~25K FPS (6x) via `mamba-ssm` package
- **Why:** Validate speedup, then run 500M step experiment to compare with RPPO's 5.82 avg gates
- **How to apply:** Queue experiment in `autoresearch/queue/tier1_e2e_rl/tier2_mamba/`

**Mamba-2 (SSD — Structured State Space Duality)**
- Same `mamba-ssm` package: `mamba_ssm.modules.mamba2.Mamba2`
- Faster training than Mamba-1 due to more parallelizable scan
- Larger d_state (64-128) efficient
- **Why:** Strictly better throughput, same API shape, minimal wrapper changes

**RPPO (LSTM) — keep as strong baseline**
- Best result: 5.82 avg gates, 29% full laps (exp #22, 500M steps)
- cuDNN LSTM is extremely mature and fast (~27K FPS)
- **Why:** Don't abandon what works. All new architectures must beat this.

### Tier 2: Investigate When Available

**Mamba-3 (ICLR 2026)**
- Paper: https://openreview.net/forum?id=HwCvaJOiCj
- Key innovations: complex-valued state dynamics, MIMO recurrence, improved discretization
- Matches Mamba-2 perplexity with HALF the state size → less memory for RL rollouts
- Complex dynamics could encode oscillatory/rotational patterns (drone curves) more naturally
- **Status: NO CODE YET** — state-spaces/mamba repo is on v2.3.1, no mamba3.py
- **How to apply:** When code drops, adapt FastMambaBlock wrapper. Placeholder experiment in queue.

**Drama (Mamba world model for model-based RL)**
- Paper: https://arxiv.org/abs/2410.08893
- 7M param Mamba world model competitive with DreamerV3 on Atari
- Relevant for `tier1_world_model/` experiments
- **Why:** If exploring world model approach, this is the Mamba-native way

**LocoMamba (vision-driven locomotion with Mamba)**
- Paper: https://arxiv.org/html/2508.11849v1
- E2E vision + RL + Mamba for quadruped robotics
- Architecture ideas transfer to drone racing with camera input

### Tier 3: Lower Priority

**VMamba / Vision Mamba**
- Bidirectional scan for image classification
- NOT useful for causal RL policy, BUT relevant as image encoder when camera pipeline is added
- Architecture: VMamba image encoder → causal Mamba policy

**RWKV-6** — competitive with Mamba on NLP, no RL work, no optimized step() kernel
**Hyena** — long-convolution, advantage at 100K+ seq length. Episodes are 4000 steps max. Overkill.
**Samba** — Mamba + sliding window attention hybrid. Designed for language, overkill for 24D obs RL.
**Griffin** — gated linear recurrence (Google). No open recurrence kernel code.
**Jamba** — Mamba + Transformer + MoE. Way too complex for RL policy.

### Key Insight from RL-Mamba Literature
Mamba's selective scan is well-suited for RL because input-dependent gating naturally learns which timesteps matter — similar to attention but O(1) per-step during rollout. Papers: Decision Mamba, Drama, LocoMamba all confirm this.

### Why FSPPO Won't Work for Vision
Frame stacking 16 camera frames = massive memory. Recurrent models (LSTM/Mamba) carry temporal context in fixed-size hidden state regardless of observation dimensionality. Mamba speed optimization is critical for the vision pipeline.
