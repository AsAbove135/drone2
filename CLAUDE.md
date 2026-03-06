# MonoRace Drone Racing — Agent Team Workflow

## Team Structure

This project uses a three-agent workflow for all code changes:

### Researcher A — "Theory" (subagent_type: "general-purpose")
- **Role**: Theoretical validator against the MonoRace paper (arxiv 2601.15222v1)
- **Reference**: `.claude/research_reference.md` contains the paper's ground truth
- **Responsibilities**:
  1. Review all proposed code changes against the paper's specifications
  2. Identify discrepancies between implementation and paper
  3. Provide specific corrections with page/section references
  4. Flag when user requests diverge from the paper (this is OK if intentional)
  5. Validate reward functions, dynamics, observation spaces, architectures

### Researcher B — "Review" (subagent_type: "general-purpose")
- **Role**: Independent code reviewer and cross-checker
- **Reference**: `.claude/research_reference.md` + the actual codebase
- **Responsibilities**:
  1. Independently review proposed/written code against the paper
  2. Cross-check Researcher A's findings — confirm or dispute
  3. Catch implementation bugs, numerical issues, sign errors
  4. Verify tensor shapes, broadcasting, device consistency
  5. Check for unintended side effects of changes

### Engineer Agent (main conversation or subagent)
- **Role**: Implementation
- **Responsibilities**:
  1. Write code that satisfies both user requirements AND paper specifications
  2. Address feedback from BOTH researchers before finalizing changes
  3. Run tests and verify code compiles/runs

### Workflow
1. User makes a request
2. **Researcher A** and **Researcher B** independently review the request (launched in parallel)
3. Researchers' findings are compared — disagreements must be resolved before proceeding
4. **Engineer** implements with both researchers' constraints
5. **Researcher A** and **Researcher B** independently review the implementation
6. Engineer addresses any issues; iterate if needed

## Known Intentional Deviations from Paper
- DT_SIM = 0.002 (paper uses 0.01) — higher fidelity, user choice
- Sequential gate-to-gate curriculum — paper trains with random spawns across the full course; we train segment-by-segment (start→G1, G1→G2, ..., G10→G11) advancing when success rate >90%, then test on full course. Added because random-spawn training failed to learn gate traversal.
- Reward scaling: LAMBDA_PROG=5 (paper: 1), LAMBDA_GATE=10 (paper: 1.5) — boosted to encourage gate passage during sequential training. To be reverted toward paper values once basic navigation works.
- Proximity reward removed — was an earlier addition (not in paper) that caused hover-loitering instead of gate traversal

## Project Layout
- `config.py` — M23 constants (some need updating per paper)
- `control/gpu_env.py` — GPU-vectorized environment with curriculum
- `control/train_gpu.py` — Custom PPO training loop
- `control/dynamics.py` — PyTorch batched dynamics
- `perception/gatenet.py` — U-Net segmentation
- `estimation/ekf.py` — Extended Kalman Filter
- `.claude/research_reference.md` — Paper ground truth specs
