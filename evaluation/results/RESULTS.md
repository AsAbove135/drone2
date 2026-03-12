# Evaluation Results

All evaluations: 10 episodes, deterministic policy (mean action), 1.5m gates, fixed start behind G1.

## Kidney Track (2 laps = 20 gates)

| Experiment | Model | Training | Completions | Avg Gates | Avg Time | Best Time |
|------------|-------|----------|-------------|-----------|----------|-----------|
| fsppo_fs4_100M | FSPPO fs=4 | kidney 100M | 0/10 | 8.0 | - | - |
| fsppo_fs8_100M | FSPPO fs=8 | kidney 100M | 2/10 | 12.4 | 9.61s | 9.61s |
| fsppo_fs16_100M | FSPPO fs=16 | kidney 100M | 10/10 | 20.0 | 9.46s | 9.44s |
| fsppo_fs16_multitrack_200M | FSPPO fs=16 | **kidney+fig8 200M** | **10/10** | **20.0** | **8.63s** | **8.61s** |
| fsppo_fs64_multitrack_200M | FSPPO fs=64 | kidney+fig8 200M | 10/10 | 20.0 | 9.40s | 9.36s |

## Figure-8 Track (2 laps = 20 gates)

| Experiment | Model | Training | Completions | Avg Gates | Avg Time | Best Time |
|------------|-------|----------|-------------|-----------|----------|-----------|
| fsppo_fs4_100M | FSPPO fs=4 | kidney 100M | 0/10 | 0.0 | - | - |
| fsppo_fs8_100M | FSPPO fs=8 | kidney 100M | 0/10 | 0.0 | - | - |
| fsppo_fs16_100M | FSPPO fs=16 | kidney 100M | 0/10 | 0.2 | - | - |
| fsppo_fs16_multitrack_200M | FSPPO fs=16 | **kidney+fig8 200M** | **10/10** | **20.0** | **11.67s** | **11.64s** |
| fsppo_fs64_multitrack_200M | FSPPO fs=64 | kidney+fig8 200M | 10/10 | 20.0 | 11.85s | 11.83s |

## Key Findings
- **Multi-track training solves generalization**: 0% → 100% completion on figure-8
- **fs16 is faster than fs64** on both tracks despite fewer parameters (363K vs 658K)
- Multi-track fs16 is **faster on kidney** than single-track fs16 (8.6s vs 9.4s) — learning diverse tracks improves overall racing
- Figure-8 takes ~35% longer than kidney (11.7s vs 8.6s) — larger track with tighter crossing gates
- Single-track models have zero generalization to unseen tracks
