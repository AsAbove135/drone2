---
name: Autoresearch system setup
description: Automated experiment infrastructure built 2026-03-14 — vast.ai orchestration, cost tracking, 3-tier queue, feature branch workflow
type: project
---

## Autoresearch System (built 2026-03-14)

### What exists
- `autoresearch/` — experiment runner with JSON configs, auto-analysis, reporting
- `autoresearch/cloud/` — vast.ai fleet management with cost tracking and budget caps
- `autoresearch/queue/` — 3-tier experiment hierarchy (approach → model → hyperparameters)
- `autoresearch/cloud/Dockerfile` — worker image with mamba-ssm CUDA kernels
- Budget caps in `autoresearch/budget.json` (daily $15, weekly $75, total $300)
- Workers create feature branches + PRs for code changes (never write to main)
- Each experiment tracks estimated vs actual GPU cost and cost efficiency (avg_gates/$)
- Master journal at `D:/drone2_training/autoresearch/JOURNAL.md`

### Commands
```bash
python -m autoresearch queue --dry-run          # Preview what would run
python -m autoresearch run <config.json>         # Single experiment
python -m autoresearch.cloud.coordinator launch  # Start vast.ai workers
python -m autoresearch.cloud.coordinator estimate # Cost estimates
python -m autoresearch.cloud.coordinator status   # Fleet status
python -m autoresearch.cloud.coordinator stop     # Kill all workers
python -m autoresearch compare <id1> <id2>       # Compare experiments
python -m autoresearch history                   # List all experiments
```

### 3 experiments queued
1. FSPPO higher entropy (0.01 vs 0.005)
2. FSPPO longer LR cycle (100M vs 50M)
3. RPPO 256 hidden (vs 128)

### Key: mamba-ssm integration
- `control_mamba/mamba_fast.py` wraps official CUDA kernels with fallback to pure PyTorch
- `MambaActorCritic` auto-detects and uses fast path when available
- Expected speedup: 4K → ~25K FPS
- Dockerfile installs `mamba-ssm` + `causal-conv1d` on workers
