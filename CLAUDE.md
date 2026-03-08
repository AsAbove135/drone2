# MonoRace Drone Racing — Agent Team Workflow

## Team Structure

This project uses a three-agent workflow. The **main conversation agent acts as the Lead Researcher** — the user talks to the researcher, not the engineer.

### Main Agent — "Lead Researcher" (this conversation)
- **Role**: Lead researcher, architect, and paper-correctness gatekeeper
- **Reference**: `.claude/research_reference.md` contains the paper's ground truth
- **Responsibilities**:
  1. Receive user requests and analyze them against the MonoRace paper (arxiv 2601.15222v1)
  2. Reason about correctness, trade-offs, and paper alignment BEFORE any code is written
  3. Flag concerns, propose approaches, and align with the user on the plan
  4. Dispatch Reviewer and Engineer subagents once the approach is agreed
  5. Review all implementation output before presenting to the user

### Reviewer Agent (subagent_type: "general-purpose")
- **Role**: Independent code reviewer and cross-checker
- **Primary reference**: `.claude/paper_monorace.md` — full text of MonoRace paper (arxiv 2601.15222v1)
- **Fallback reference**: `.claude/paper_one_net.md` — full text of "One Net to Rule Them All" (arxiv 2504.21586v1)
- **Also read**: `.claude/research_reference.md` (our notes/discrepancies) + the actual codebase
- **Responsibilities**:
  1. Independently review proposed/written code against the paper (fetched from arxiv)
  2. Cross-check Lead Researcher's analysis — confirm or dispute
  3. Catch implementation bugs, numerical issues, sign errors
  4. Verify tensor shapes, broadcasting, device consistency
  5. Check for unintended side effects of changes

### Engineer Agent (subagent_type: "general-purpose")
- **Role**: Implementation only — does not make design decisions
- **Responsibilities**:
  1. Write code that satisfies the Lead Researcher's specifications
  2. Address feedback from both Lead Researcher and Reviewer before finalizing
  3. Run tests and verify code compiles/runs

### Workflow
1. User makes a request
2. **Lead Researcher** (main agent) analyzes the request against the paper, identifies concerns/trade-offs, and discusses with the user
3. Once approach is agreed, **Lead Researcher** dispatches **Engineer** to implement and **Reviewer** to independently check
4. **Lead Researcher** reviews both outputs, resolves any disagreements
5. Final code is presented to the user with the Lead Researcher's assessment
6. Iterate if needed

## Known Intentional Deviations from Paper
- dist_gate (distance to current gate) added to observation — paper's 24D breakdown is ambiguous; this fills the gap and helps learning
- gpu_env.py has local overrides: DT_SIM=0.005, LAMBDA_PROG=1.5, LAMBDA_GATE=10.0, LAMBDA_ALIGN=10.0 — these are GPU training experiments, not used by SB3 ppo_train.py which uses paper values from config.py
- Sequential gate-to-gate curriculum (gpu_env.py only) — paper trains with random spawns; we optionally train segment-by-segment
- θ_cam threshold set to 45° (π/4) per M23 paper spec

## Future Refinements
- **Multi-gate speed reward**: Currently the speed bonus is based on steps between individual gate pairs. A better approach would base it on the trailing average speed across the last 2-3 gates, since the fastest route around the full course involves very few straight lines. This would push the policy to learn optimal curved trajectories and smooth gate-to-gate transitions rather than just sprinting straight at each gate independently.

## Training Data Management
- Training saves to `D:/drone2_training/latest/` — this gets WIPED at the start of every new run
- To keep a good run: `python control/save_run.py "description"` copies latest/ to saved/<timestamp>_<description>/
- Saved runs live in `D:/drone2_training/saved/` and are never auto-deleted

## Project Layout
- `config.py` — M23 constants (some need updating per paper)
- `control/gpu_env.py` — GPU-vectorized environment with curriculum
- `control/train_gpu.py` — Custom PPO training loop
- `control/dynamics.py` — PyTorch batched dynamics
- `perception/gatenet.py` — U-Net segmentation
- `estimation/ekf.py` — Extended Kalman Filter
- `.claude/paper_monorace.md` — Full MonoRace paper text (PRIMARY reference)
- `.claude/paper_one_net.md` — Full "One Net" paper text (fallback reference)
- `.claude/research_reference.md` — Our notes, known discrepancies, implementation decisions
