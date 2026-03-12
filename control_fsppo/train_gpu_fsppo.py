"""
GPU Frame-Stacked PPO training for MonoRace M23 policy.
Feedforward MLP actor-critic with frame-stacked observations on CUDA.
Based on RPPO trainer but replaces LSTM with frame stacking for temporal context.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.gpu_env import BatchedDroneEnv
from config import OBS_DIM, TOTAL_TIMESTEPS, NUM_GATES, NUM_LAPS


# ── Frame-Stacked Actor-Critic Network ──────────────────────────

class FrameStackedActorCritic(nn.Module):
    """Actor-Critic with frame-stacked feedforward MLP.
    Separate actor and critic MLPs sharing the first encoder layer."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=4, frame_stack=16):
        super().__init__()
        input_dim = obs_dim * frame_stack

        # Shared input encoding (first layer)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
        )

        # Actor MLP
        self.actor_mlp = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(256, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std_min = -2.0
        self.log_std_max = 0.5

        # Critic MLP
        self.critic_mlp = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.critic_head = nn.Linear(256, 1)

        # Orthogonal init
        for module in [self.encoder, self.actor_mlp, self.critic_mlp]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)
                    nn.init.constant_(layer.bias, 0.0)
        # Last layers with small gain
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def forward(self, stacked_obs):
        """
        Forward pass.
        stacked_obs: [N, obs_dim * frame_stack]
        Returns: mean [N, act_dim], std [N, act_dim], value [N]
        """
        enc = self.encoder(stacked_obs)  # [N, 256]

        actor_feat = self.actor_mlp(enc)  # [N, 256]
        mean = self.actor_mean(actor_feat)
        std = self.actor_log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)

        critic_feat = self.critic_mlp(enc)  # [N, 256]
        value = self.critic_head(critic_feat).squeeze(-1)

        return mean, std, value

    def get_action(self, stacked_obs):
        """Sample action during rollout. Returns action, log_prob, value, action_raw."""
        mean, std, value = self.forward(stacked_obs)
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        action = torch.sigmoid(action_raw)
        return action, log_prob, value, action_raw

    def evaluate(self, stacked_obs, actions_raw):
        """
        Evaluate actions for PPO update.
        stacked_obs: [N, obs_dim * frame_stack]
        actions_raw: [N, act_dim]
        Returns: log_probs [N], values [N], entropy [N]
        """
        mean, std, value = self.forward(stacked_obs)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, value, entropy


# ── Frame Stack Buffer ───────────────────────────────────────────

class FrameStackBuffer:
    """Maintains a rolling buffer of recent observations for each env."""
    def __init__(self, num_envs, frame_stack, obs_dim, device):
        self.num_envs = num_envs
        self.frame_stack = frame_stack
        self.obs_dim = obs_dim
        self.device = device
        # [num_envs, frame_stack, obs_dim] — index 0 is oldest, -1 is newest
        self.buffer = torch.zeros(num_envs, frame_stack, obs_dim, device=device)

    def push(self, obs):
        """Shift buffer and insert new observation. obs: [num_envs, obs_dim]"""
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1] = obs

    def reset(self, env_mask):
        """Zero out frame stack for envs where env_mask is True. env_mask: [num_envs] bool or float."""
        if env_mask.any():
            self.buffer[env_mask] = 0.0

    def get_stacked(self):
        """Return flattened stacked obs: [num_envs, frame_stack * obs_dim]"""
        return self.buffer.reshape(self.num_envs, -1)


# ── Rollout Buffer ───────────────────────────────────────────────

class RolloutBuffer:
    """Standard (non-recurrent) rollout buffer for frame-stacked PPO."""
    def __init__(self, num_envs, n_steps, stacked_obs_dim, act_dim, device):
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.device = device

        self.obs = torch.zeros(n_steps, num_envs, stacked_obs_dim, device=device)
        self.actions_raw = torch.zeros(n_steps, num_envs, act_dim, device=device)
        self.rewards = torch.zeros(n_steps, num_envs, device=device)
        self.dones = torch.zeros(n_steps, num_envs, device=device)
        self.log_probs = torch.zeros(n_steps, num_envs, device=device)
        self.values = torch.zeros(n_steps, num_envs, device=device)
        self.advantages = torch.zeros(n_steps, num_envs, device=device)
        self.returns = torch.zeros(n_steps, num_envs, device=device)

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95, active_steps=None):
        """Compute GAE."""
        T = active_steps if active_steps is not None else self.n_steps
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_done - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_done * gae
            self.advantages[t] = gae
        self.returns[:T] = self.advantages[:T] + self.values[:T]


# ── Observation Normalizer ────────────────────────────────────────

class RunningNorm:
    """Welford's online running mean/variance for observation normalization."""
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = M2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        return ((x - self.mean) / (self.var.sqrt() + 1e-8)).clamp(-clip, clip)


# ── Training Loop ─────────────────────────────────────────────────

def train(
    num_envs=2048,
    total_timesteps=TOTAL_TIMESTEPS,
    n_steps=512,
    n_epochs=5,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    frame_stack=16,
    minibatch_size=4096,
    fixed_start=False,
    single_gate=False,
    random_segments=True,
    gate_size=None,
    lambda_prog=None,
    lambda_gate=None,
    lambda_gate_inc=None,
    lambda_rate=None,
    curriculum_advance_pct=85.0,
    lr_cycle=50_000_000,
    lr_min=1e-5,
    cosine_lr=True,
    resume=None,
    track_names=None,
    save_dir="D:/drone2_training/fsppo_latest",
    device="cuda",
):
    import shutil
    from datetime import datetime
    device = torch.device(device)

    # Wipe previous run data (skip when resuming)
    if resume is None:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print(f"Cleaned up previous run in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    stacked_obs_dim = OBS_DIM * frame_stack

    mode_parts = ["Frame-Stacked PPO"]
    if fixed_start:
        mode_parts.append("FIXED START")
    if random_segments:
        mode_parts.append("RANDOM SEGMENTS")
    elif single_gate:
        mode_parts.append("SINGLE GATE")
    if gate_size:
        mode_parts.append(f"GATE={gate_size}m")
    if track_names:
        mode_parts.append(f"TRACKS={'+'.join(track_names)}")
    mode = ", ".join(mode_parts)
    print(f"GPU Frame-Stacked PPO Training: {num_envs} envs, {total_timesteps:,} timesteps [{mode}]")
    print(f"Frame stack: {frame_stack}, input dim: {stacked_obs_dim}")
    print(f"Minibatch size: {minibatch_size}")
    print(f"Device: {device}")

    # Environment
    env = BatchedDroneEnv(num_envs=num_envs, device=device,
                          fixed_start=fixed_start,
                          domain_randomize=not fixed_start,
                          single_gate=single_gate,
                          gate_size_override=gate_size,
                          random_segments=random_segments,
                          lambda_prog=lambda_prog,
                          lambda_gate=lambda_gate,
                          lambda_gate_inc=lambda_gate_inc,
                          lambda_rate=lambda_rate,
                          track_names=track_names)

    # Policy
    policy = FrameStackedActorCritic(
        obs_dim=OBS_DIM, act_dim=4, frame_stack=frame_stack,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"FrameStackedActorCritic parameters: {param_count:,}")

    # Buffer
    buffer = RolloutBuffer(
        num_envs, n_steps, stacked_obs_dim, 4, device
    )

    # Frame stack buffer
    fs_buffer = FrameStackBuffer(num_envs, frame_stack, OBS_DIM, device)

    # Observation normalizer (operates on raw obs_dim, not stacked)
    obs_norm = RunningNorm(OBS_DIM, device)

    # Resume from checkpoint
    resume_step = 0
    if resume is not None:
        print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt['policy_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  Loaded optimizer state")
        obs_norm.mean = ckpt['obs_norm_mean']
        obs_norm.var = ckpt['obs_norm_var']
        resume_step = ckpt.get('global_step', 0)
        print(f"  Loaded policy + obs_norm. Resuming from step {resume_step:,}")

    # Initial obs
    raw_obs = env.reset_all()
    if resume is None:
        obs_norm.update(raw_obs)
    norm_obs = obs_norm.normalize(raw_obs)
    fs_buffer.push(norm_obs)

    global_step = resume_step
    start_time = time.time()

    num_iterations = total_timesteps // (num_envs * n_steps)

    # CSV logging
    csv_path = os.path.join(save_dir, "training_stats.csv")
    num_total_gates = NUM_GATES * NUM_LAPS
    with open(csv_path, 'w') as f:
        cols = ["steps", "reward_mean", "pg_loss", "v_loss", "entropy", "clip_frac",
                "lr", "ent_coef", "difficulty", "gate_pass_rate", "avg_gates",
                "gates_passed", "episodes", "fps",
                "r_prog", "r_gate", "r_offset", "r_rate", "r_perc", "r_align",
                "curriculum_stage"]
        cols += [f"pct_ge_{i}" for i in range(1, num_total_gates + 1)]
        cols += [f"seg_{i}_pass_rate" for i in range(NUM_GATES)]
        f.write(",".join(cols) + "\n")

    # Gate size curriculum: 1.5m -> 0.5m over first 20% of training
    GATE_SIZE_START, GATE_SIZE_END = 1.5, 0.5
    GATE_CURRICULUM_FRAC = 0.20

    # Sequential gate curriculum
    CURRICULUM_ADVANCE_PCT = curriculum_advance_pct
    curriculum_stage = 1
    curriculum_stage_steps = 0

    # Auto-plotting setup
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_tracking')
    save_dir_label = os.path.basename(save_dir.rstrip('/\\'))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M") + "_" + save_dir_label
    last_plot_time = 0

    print(f"Plots updating every 5 min to: results_tracking/{run_name}/")
    print("-" * 70)

    iteration = 0
    while global_step < total_timesteps:
        iteration += 1

        # ── Gate size curriculum (skip if fixed gate size provided) ──
        if gate_size is None:
            gate_frac = min(1.0, global_step / (total_timesteps * GATE_CURRICULUM_FRAC))
            env.gate_size = GATE_SIZE_START + (GATE_SIZE_END - GATE_SIZE_START) * gate_frac
            difficulty = gate_frac
        else:
            difficulty = 1.0

        # ── LR schedule ──
        if cosine_lr:
            cycle_progress = (global_step % lr_cycle) / lr_cycle
            lr_now = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(2.0 * math.pi * cycle_progress))
        else:
            lr_now = lr * (1.0 - global_step / total_timesteps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # ── Collect rollouts ──
        policy.eval()
        with torch.no_grad():
            for t in range(n_steps):
                stacked_obs = fs_buffer.get_stacked()  # [N, frame_stack * obs_dim]

                action, log_prob, value, action_raw = policy.get_action(stacked_obs)

                next_raw_obs, rewards, dones = env.step(action)
                obs_norm.update(next_raw_obs)
                next_norm_obs = obs_norm.normalize(next_raw_obs)

                buffer.obs[t] = stacked_obs
                buffer.actions_raw[t] = action_raw
                buffer.rewards[t] = rewards
                buffer.dones[t] = dones.float()
                buffer.log_probs[t] = log_prob
                buffer.values[t] = value

                # Reset frame stack for finished episodes, then push new obs
                fs_buffer.reset(dones.bool())
                fs_buffer.push(next_norm_obs)

                global_step += num_envs

            # Bootstrap value
            stacked_obs = fs_buffer.get_stacked()
            _, _, last_value = policy.forward(stacked_obs)

        # ── Compute advantages ──
        buffer.compute_gae(last_value, gamma, gae_lambda)

        # ── PPO update ──
        policy.train()
        total_samples = n_steps * num_envs

        # Flatten rollout data
        flat_obs = buffer.obs.reshape(total_samples, stacked_obs_dim)
        flat_actions = buffer.actions_raw.reshape(total_samples, 4)
        flat_log_probs = buffer.log_probs.reshape(total_samples)
        flat_values = buffer.values.reshape(total_samples)
        flat_returns = buffer.returns.reshape(total_samples)
        flat_advantages = buffer.advantages.reshape(total_samples)

        clip_fracs = []
        pg_losses = []
        v_losses = []
        ent_losses = []

        for epoch in range(n_epochs):
            # Shuffle sample indices
            indices = torch.randperm(total_samples, device=device)

            for start in range(0, total_samples, minibatch_size):
                end = min(start + minibatch_size, total_samples)
                mb_idx = indices[start:end]

                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_lp = flat_log_probs[mb_idx]
                mb_old_values = flat_values[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_advantages = flat_advantages[mb_idx]

                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Forward pass
                new_log_prob, new_value, entropy = policy.evaluate(mb_obs, mb_actions)

                # Policy loss
                ratio = (new_log_prob - mb_old_lp).exp()
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio.clamp(1 - clip_range, 1 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * (new_value - mb_returns).pow(2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                clip_fracs.append(((ratio - 1).abs() > clip_range).float().mean().item())
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())

        # ── Logging ──
        elapsed = time.time() - start_time
        fps = global_step / elapsed

        if iteration % 5 == 0 or iteration == 1:
            avg_reward = buffer.rewards.sum(dim=0).mean().item()
            gates_passed = env.gates_passed_count
            eps_ended = env.episodes_ended_count
            pass_rate = gates_passed / max(eps_ended, 1)
            env.gates_passed_count = 0
            env.episodes_ended_count = 0

            # Reward components: convert accumulated sums to per-env-per-step averages
            total_steps_in_period = n_steps * num_envs * 5  # 5 iterations between logs
            rc = {k: v / max(total_steps_in_period, 1) for k, v in env.reward_components.items()}
            env.reward_components = {k: 0.0 for k in env.reward_components}

            # Gate distribution from per-episode gate counts
            ep_counts = env.ep_gates_list if env.ep_gates_list else [0]
            avg_gates = sum(ep_counts) / len(ep_counts)
            n_eps = len(ep_counts)
            pct_ge = []
            for threshold in range(1, num_total_gates + 1):
                pct = sum(1 for c in ep_counts if c >= threshold) / max(n_eps, 1) * 100
                pct_ge.append(pct)
            env.ep_gates_list = []

            # Per-segment pass rates
            seg_pass_rates = []
            for si in range(NUM_GATES):
                ep_count = env.segment_episodes[si]
                gate_count = env.segment_gates[si]
                seg_pass_rates.append(gate_count / max(ep_count, 1))
            env.segment_gates = [0] * NUM_GATES
            env.segment_episodes = [0] * NUM_GATES

            # Curriculum advancement
            if curriculum_stage <= num_total_gates and len(pct_ge) >= curriculum_stage:
                if pct_ge[curriculum_stage - 1] >= CURRICULUM_ADVANCE_PCT:
                    curriculum_stage += 1
                    curriculum_stage_steps = global_step
                    print(f"  [Curriculum] Advanced to stage {curriculum_stage}: >= {curriculum_stage} gates at {global_step:,} steps")

            avg_pg = sum(pg_losses)/len(pg_losses)
            avg_vl = sum(v_losses)/len(v_losses)
            avg_ent = sum(ent_losses)/len(ent_losses)
            avg_clip = sum(clip_fracs)/len(clip_fracs)
            cur_lr = optimizer.param_groups[0]['lr']

            print(
                f"Iter {iteration:5d}/{num_iterations} | "
                f"Steps: {global_step:>12,} | "
                f"FPS: {fps:>8,.0f} | "
                f"Reward: {avg_reward:>8.3f} | "
                f"PG: {avg_pg:.4f} | "
                f"VL: {avg_vl:.4f} | "
                f"Ent: {avg_ent:.3f} | "
                f"Clip: {avg_clip:.3f} | "
                f"LR: {cur_lr:.2e} | "
                f"Diff: {difficulty:.2f} | "
                f"Stage: {curriculum_stage} | "
                f"Gates: {gates_passed}/{eps_ended} ({pass_rate:.1%}) | "
                f"Avg: {avg_gates:.2f} | "
                f">=1: {pct_ge[0]:.0f}%",
                flush=True,
            )

            # Write CSV row
            with open(csv_path, 'a') as f:
                row = [global_step, avg_reward, avg_pg, avg_vl,
                       avg_ent, avg_clip, cur_lr, ent_coef,
                       difficulty, pass_rate, avg_gates,
                       gates_passed, eps_ended, fps,
                       rc['prog'], rc['gate'], rc['offset'],
                       rc['rate'], rc['perc'], rc['align'],
                       curriculum_stage]
                row += pct_ge
                row += seg_pass_rates
                f.write(",".join(f"{v:.4f}" for v in row) + "\n")

        # ── Periodic plots ──
        now = time.time()
        if now - last_plot_time >= 300:
            last_plot_time = now
            try:
                from control.plot_training import generate_charts
                generate_charts(csv_path, results_dir, run_name)
                print(f"  [Plots] Updated charts in results_tracking/{run_name}/", flush=True)
            except Exception as e:
                print(f"  [Plots] Failed: {e}", flush=True)

        # ── Checkpoints ──
        if iteration % 50 == 0:
            ckpt_path = os.path.join(save_dir, f"fsppo_gpu_{global_step}.pt")
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'obs_norm_mean': obs_norm.mean,
                'obs_norm_var': obs_norm.var,
                'global_step': global_step,
            }, ckpt_path)

    # Save final
    tag = "randseg" if random_segments else ("single" if single_gate else ("fixed" if fixed_start else "gpu"))
    final_path = os.path.join(save_dir, f"fsppo_m23_{tag}_final.pt")
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'obs_norm_mean': obs_norm.mean,
        'obs_norm_var': obs_norm.var,
        'global_step': global_step,
    }, final_path)

    # Final plots
    try:
        from control.plot_training import generate_charts
        generate_charts(csv_path, results_dir, run_name)
    except Exception:
        pass

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Training complete! {global_step:,} steps in {elapsed:.1f}s ({global_step/elapsed:,.0f} FPS)")
    print(f"Model saved to {final_path}")
    print('To keep this run: python control/save_run.py "description"')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Frame-Stacked PPO Training for MonoRace M23")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="D:/drone2_training/fsppo_latest")
    parser.add_argument("--frame-stack", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--single-gate", action="store_true")
    parser.add_argument("--no-random-segments", action="store_true")
    parser.add_argument("--gate-size", type=float, default=None)
    parser.add_argument("--lambda-prog", type=float, default=None)
    parser.add_argument("--lambda-gate", type=float, default=None)
    parser.add_argument("--lambda-gate-inc", type=float, default=None)
    parser.add_argument("--lambda-rate", type=float, default=None)
    parser.add_argument("--curriculum-advance-pct", type=float, default=85.0,
                        help="Advance curriculum stage when this %% of episodes pass >= stage gates")
    parser.add_argument("--lr-cycle", type=int, default=50_000_000,
                        help="Cosine LR warm restart cycle length in steps")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="Minimum LR for cosine schedule")
    parser.add_argument("--cosine-lr", action=argparse.BooleanOptionalAction, default=True,
                        help="Use cosine LR with warm restarts (--no-cosine-lr to disable)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from (loads policy, optimizer, obs_norm)")
    parser.add_argument("--track-names", type=str, nargs='+', default=None,
                        help="Track names for multi-track training (e.g., --track-names kidney figure8)")
    args = parser.parse_args()

    train(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        device=args.device,
        save_dir=args.save_dir,
        frame_stack=args.frame_stack,
        minibatch_size=args.minibatch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        fixed_start=args.fixed_start,
        single_gate=args.single_gate,
        random_segments=not args.no_random_segments,
        gate_size=args.gate_size,
        lambda_prog=args.lambda_prog,
        lambda_gate=args.lambda_gate,
        lambda_gate_inc=args.lambda_gate_inc,
        lambda_rate=args.lambda_rate,
        curriculum_advance_pct=args.curriculum_advance_pct,
        lr_cycle=args.lr_cycle,
        lr_min=args.lr_min,
        cosine_lr=args.cosine_lr,
        resume=args.resume,
        track_names=args.track_names,
    )
