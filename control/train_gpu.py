"""
Custom PPO training loop for GPU-vectorized drone environment.
Everything runs on CUDA — no numpy in the hot loop.
Matches SB3's PPO implementation details for proper convergence.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.gpu_env import BatchedDroneEnv
from config import OBS_DIM, TOTAL_TIMESTEPS


# ── Actor-Critic Network ─────────────────────────────────────────

class ActorCritic(nn.Module):
    """Actor-Critic with separate networks. M23 architecture: 3×64 FC, ReLU."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=4):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Orthogonal initialization (matches SB3)
        for module in [self.actor_mean, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)
                    nn.init.constant_(layer.bias, 0.0)
        # Last layers with small gain
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs):
        mean = self.actor_mean(obs)
        std = self.actor_log_std.clamp(-2.0, 0.5).exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def get_action(self, obs):
        """Sample action, return (action, log_prob, value)."""
        mean, std, value = self(obs)
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        # Squash to [0, 1] via sigmoid
        action = torch.sigmoid(action_raw)
        return action, log_prob, value, action_raw

    def evaluate(self, obs, action_raw):
        """Evaluate log_prob and value for given obs and raw (pre-sigmoid) actions."""
        mean, std, value = self(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy


# ── Rollout Buffer ────────────────────────────────────────────────

class RolloutBuffer:
    """Stores rollout data as GPU tensors."""
    def __init__(self, num_envs, n_steps, obs_dim, act_dim, device):
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.device = device

        self.obs = torch.zeros(n_steps, num_envs, obs_dim, device=device)
        self.actions_raw = torch.zeros(n_steps, num_envs, act_dim, device=device)
        self.rewards = torch.zeros(n_steps, num_envs, device=device)
        self.dones = torch.zeros(n_steps, num_envs, device=device)
        self.log_probs = torch.zeros(n_steps, num_envs, device=device)
        self.values = torch.zeros(n_steps, num_envs, device=device)
        self.advantages = torch.zeros(n_steps, num_envs, device=device)
        self.returns = torch.zeros(n_steps, num_envs, device=device)

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation."""
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_done - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_done * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def flatten(self):
        """Flatten [T, N, ...] -> [T*N, ...]."""
        T, N = self.n_steps, self.num_envs
        return (
            self.obs.reshape(T * N, -1),
            self.actions_raw.reshape(T * N, -1),
            self.log_probs.reshape(T * N),
            self.values.reshape(T * N),
            self.returns.reshape(T * N),
            self.advantages.reshape(T * N),
        )


# ── Observation Normalizer ────────────────────────────────────────

class RunningNorm:
    """Welford's online running mean/variance for observation normalization."""
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x):
        """Update with a batch [N, D]."""
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
    n_steps=64,             # Scaled from SB3 default (2048 / 1 env = 64 / 2048 envs)
    batch_size=4096,        # Scaled: keeps 32 minibatches/epoch like SB3's 2048/64
    n_epochs=10,            # SB3 default
    lr=3e-4,                # SB3 default
    gamma=0.99,             # SB3 default
    gae_lambda=0.95,        # SB3 default
    clip_range=0.2,         # SB3 default
    ent_coef=0.0,           # M23 paper: 0 entropy
    vf_coef=0.5,            # SB3 default
    max_grad_norm=0.5,      # SB3 default
    vf_clip_range=None,     # SB3 default (no VF clipping)
    fixed_start=False,      # Spawn all envs in front of gate 1, no DR
    single_gate=False,      # Terminate on first gate passage (success)
    gate_size=None,         # Override gate size (None = use config default)
    save_dir="D:/drone2_training",
    device="cuda",
):
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    mode_parts = []
    if fixed_start:
        mode_parts.append("FIXED START")
    if single_gate:
        mode_parts.append("SINGLE GATE")
    if gate_size:
        mode_parts.append(f"GATE={gate_size}m")
    mode = ", ".join(mode_parts) if mode_parts else "CURRICULUM"
    print(f"GPU PPO Training: {num_envs} envs, {total_timesteps:,} timesteps [{mode}]")
    print(f"Device: {device}")

    # Environment
    env = BatchedDroneEnv(num_envs=num_envs, device=device,
                          fixed_start=fixed_start,
                          domain_randomize=not fixed_start,
                          single_gate=single_gate,
                          gate_size_override=gate_size)

    # Policy
    policy = ActorCritic(obs_dim=OBS_DIM, act_dim=4).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"ActorCritic parameters: {param_count:,}")

    # Buffer
    buffer = RolloutBuffer(num_envs, n_steps, OBS_DIM, 4, device)

    # Observation normalizer
    obs_norm = RunningNorm(OBS_DIM, device)

    # Initial obs
    obs = env.reset_all()
    obs_norm.update(obs)
    obs = obs_norm.normalize(obs)

    num_iterations = total_timesteps // (num_envs * n_steps)
    global_step = 0
    start_time = time.time()

    print(f"Iterations: {num_iterations}, steps/iter: {num_envs * n_steps:,}")
    print("-" * 70)

    for iteration in range(1, num_iterations + 1):
        # ── Curriculum schedule (skipped in fixed_start mode) ──
        if not fixed_start:
            difficulty = min(1.0, global_step / total_timesteps * 1.25)
            env.set_difficulty(difficulty)
        else:
            difficulty = 0.0

        # ── Linear LR annealing (SB3 default) ──
        frac = 1.0 - (iteration - 1) / num_iterations
        for pg in optimizer.param_groups:
            pg['lr'] = lr * frac

        # ── Collect rollouts ──
        policy.eval()
        with torch.no_grad():
            for t in range(n_steps):
                action, log_prob, value, action_raw = policy.get_action(obs)

                next_obs, rewards, dones = env.step(action)

                obs_norm.update(next_obs)
                next_obs_norm = obs_norm.normalize(next_obs)

                buffer.obs[t] = obs
                buffer.actions_raw[t] = action_raw
                buffer.rewards[t] = rewards
                buffer.dones[t] = dones.float()
                buffer.log_probs[t] = log_prob
                buffer.values[t] = value

                obs = next_obs_norm
                global_step += num_envs

            # Bootstrap value
            _, _, last_value = policy(obs)

        # ── Compute advantages ──
        buffer.compute_gae(last_value, gamma, gae_lambda)

        # ── PPO update ──
        policy.train()
        b_obs, b_actions, b_old_log_probs, b_old_values, b_returns, b_advantages = buffer.flatten()

        total_samples = num_envs * n_steps
        clip_fracs = []
        pg_losses = []
        v_losses = []
        ent_losses = []

        for epoch in range(n_epochs):
            # Random permutation for minibatches
            indices = torch.randperm(total_samples, device=device)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_lp = b_old_log_probs[mb_idx]
                mb_old_values = b_old_values[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_advantages = b_advantages[mb_idx]

                # Per-minibatch advantage normalization (SB3 default)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_log_prob, new_value, entropy = policy.evaluate(mb_obs, mb_actions)

                # Policy loss
                ratio = (new_log_prob - mb_old_lp).exp()
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio.clamp(1 - clip_range, 1 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping (SB3 default)
                if vf_clip_range is not None:
                    v_clipped = mb_old_values + (new_value - mb_old_values).clamp(
                        -vf_clip_range, vf_clip_range)
                    v_loss_unclipped = (new_value - mb_returns).pow(2)
                    v_loss_clipped = (v_clipped - mb_returns).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
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

        if iteration % 10 == 0 or iteration == 1:
            avg_reward = buffer.rewards.sum(dim=0).mean().item()
            print(
                f"Iter {iteration:5d}/{num_iterations} | "
                f"Steps: {global_step:>12,} | "
                f"FPS: {fps:>8,.0f} | "
                f"Reward: {avg_reward:>8.3f} | "
                f"PG Loss: {sum(pg_losses)/len(pg_losses):.4f} | "
                f"V Loss: {sum(v_losses)/len(v_losses):.4f} | "
                f"Entropy: {sum(ent_losses)/len(ent_losses):.3f} | "
                f"Clip: {sum(clip_fracs)/len(clip_fracs):.3f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Diff: {difficulty:.2f}",
                flush=True,
            )

        # ── Checkpoints ──
        if iteration % 100 == 0:
            ckpt_path = os.path.join(save_dir, f"gcnet_gpu_{global_step}.pt")
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'obs_norm_mean': obs_norm.mean,
                'obs_norm_var': obs_norm.var,
                'global_step': global_step,
            }, ckpt_path)

    # Save final
    tag = "single" if single_gate else ("fixed" if fixed_start else "gpu")
    final_path = os.path.join(save_dir, f"gcnet_m23_{tag}_final.pt")
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'obs_norm_mean': obs_norm.mean,
        'obs_norm_var': obs_norm.var,
        'global_step': global_step,
    }, final_path)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Training complete! {global_step:,} steps in {elapsed:.1f}s ({global_step/elapsed:,.0f} FPS)")
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU PPO Training for MonoRace M23")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="D:/drone2_training")
    parser.add_argument("--fixed-start", action="store_true",
                        help="Spawn all envs in front of gate 1, no DR or curriculum")
    parser.add_argument("--single-gate", action="store_true",
                        help="Terminate episode on first gate passage (success)")
    parser.add_argument("--gate-size", type=float, default=None,
                        help="Override gate size in meters (default: config value)")
    args = parser.parse_args()

    train(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        device=args.device,
        save_dir=args.save_dir,
        fixed_start=args.fixed_start,
        single_gate=args.single_gate,
        gate_size=args.gate_size,
    )
