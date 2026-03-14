# Experiment Queue

Three-tier hierarchy for automated research:

```
queue/
├── tier1_e2e_rl/              # End-to-end reinforcement learning
│   ├── tier2_fsppo/           # Frame-stacked PPO experiments
│   ├── tier2_rppo/            # Recurrent PPO (LSTM) experiments
│   └── tier2_mamba/           # Mamba SSM experiments
├── tier1_world_model/         # World model approaches
│   ├── tier2_jepa/            # JEPA-style world models
│   └── tier2_dreamer/         # DreamerV3-style world models
└── tier1_vision_estimation/   # Vision + state estimation
    └── tier2_ekf_nn/          # Neural EKF variants
```

## Adding experiments

1. Copy `autoresearch/experiments/TEMPLATE.json`
2. Place in the appropriate tier directory
3. Name with numeric prefix for ordering: `01_name.json`, `02_name.json`
4. Workers pick experiments in filesystem order, deepest first

## How claiming works

- Workers scan the queue for `.json` files without a `.lock` companion
- On claim, the worker writes `filename.json.lock` and pushes to git
- First push wins — if another worker claimed it, the push fails and they retry
- After completion, results go to the `autoresearch-results` branch
