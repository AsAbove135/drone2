"""
GPU Recurrent PPO training for MonoRace M23 policy.
Custom LSTM actor-critic on CUDA with sequence-aware minibatching.
Based on train_gpu.py but with recurrent policy.
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


# ── Recurrent Actor-Critic Network ───────────────────────────────

class RecurrentActorCritic(nn.Module):
    """Actor-Critic with LSTM. Separate LSTM + MLP heads for actor and critic."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=4, lstm_hidden=128, n_lstm_layers=1):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.n_lstm_layers = n_lstm_layers

        # Shared input encoding
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
        )

        # Separate LSTMs for actor and critic
        self.actor_lstm = nn.LSTM(64, lstm_hidden, n_lstm_layers, batch_first=False)
        self.critic_lstm = nn.LSTM(64, lstm_hidden, n_lstm_layers, batch_first=False)

        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std_min = -2.0
        self.log_std_max = 0.5

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Orthogonal init
        for name, param in self.actor_lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        for name, param in self.critic_lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        for module in [self.encoder, self.actor_mean, self.critic_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)
                    nn.init.constant_(layer.bias, 0.0)
        # Last layers with small gain
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def initial_state(self, num_envs, device):
        """Return zero hidden states: ((h_actor, c_actor), (h_critic, c_critic))."""
        z = lambda: torch.zeros(self.n_lstm_layers, num_envs, self.lstm_hidden, device=device)
        return (z(), z()), (z(), z())

    def forward_single(self, obs, actor_hc, critic_hc):
        """
        Forward pass for a single timestep during rollout collection.
        obs: [N, obs_dim]
        actor_hc, critic_hc: tuple of (h, c) each [n_layers, N, hidden]
        Returns: mean, std, value, new_actor_hc, new_critic_hc
        """
        enc = self.encoder(obs)  # [N, 64]
        enc = enc.unsqueeze(0)   # [1, N, 64] for LSTM (seq_len=1)

        actor_out, new_actor_hc = self.actor_lstm(enc, actor_hc)
        critic_out, new_critic_hc = self.critic_lstm(enc, critic_hc)

        actor_out = actor_out.squeeze(0)   # [N, hidden]
        critic_out = critic_out.squeeze(0)  # [N, hidden]

        mean = self.actor_mean(actor_out)
        std = self.actor_log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)
        value = self.critic_head(critic_out).squeeze(-1)

        return mean, std, value, new_actor_hc, new_critic_hc

    def get_action(self, obs, actor_hc, critic_hc):
        """Sample action during rollout. Returns action, log_prob, value, action_raw, new hidden states."""
        mean, std, value, new_ahc, new_chc = self.forward_single(obs, actor_hc, critic_hc)
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        action = torch.sigmoid(action_raw)
        return action, log_prob, value, action_raw, new_ahc, new_chc

    def evaluate_sequences(self, obs_seq, actions_seq, dones_seq, actor_hc_init, critic_hc_init):
        """
        Evaluate sequences for PPO update.
        obs_seq: [seq_len, batch, obs_dim]
        actions_seq: [seq_len, batch, act_dim]
        dones_seq: [seq_len, batch]
        actor_hc_init, critic_hc_init: initial hidden states [n_layers, batch, hidden]
        Returns: log_probs [seq_len, batch], values [seq_len, batch], entropy [seq_len, batch]
        """
        seq_len, batch_size = obs_seq.shape[:2]

        # Encode all timesteps at once
        enc = self.encoder(obs_seq.reshape(seq_len * batch_size, -1))
        enc = enc.reshape(seq_len, batch_size, -1)  # [S, B, 64]

        # Process LSTM step-by-step to handle done resets
        actor_h, actor_c = actor_hc_init
        critic_h, critic_c = critic_hc_init

        actor_outputs = []
        critic_outputs = []

        for t in range(seq_len):
            # Reset hidden state where episodes ended at previous step
            if t > 0:
                done_mask = dones_seq[t - 1].unsqueeze(0).unsqueeze(-1)  # [1, B, 1]
                actor_h = actor_h * (1.0 - done_mask)
                actor_c = actor_c * (1.0 - done_mask)
                critic_h = critic_h * (1.0 - done_mask)
                critic_c = critic_c * (1.0 - done_mask)

            inp = enc[t:t+1]  # [1, B, 64]
            actor_out, (actor_h, actor_c) = self.actor_lstm(inp, (actor_h, actor_c))
            critic_out, (critic_h, critic_c) = self.critic_lstm(inp, (critic_h, critic_c))
            actor_outputs.append(actor_out.squeeze(0))
            critic_outputs.append(critic_out.squeeze(0))

        actor_outputs = torch.stack(actor_outputs)   # [S, B, hidden]
        critic_outputs = torch.stack(critic_outputs)  # [S, B, hidden]

        # Compute action distribution and values
        mean = self.actor_mean(actor_outputs.reshape(seq_len * batch_size, -1))
        mean = mean.reshape(seq_len, batch_size, -1)
        std = self.actor_log_std.clamp(self.log_std_min, self.log_std_max).exp()
        std = std.unsqueeze(0).unsqueeze(0).expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions_seq).sum(dim=-1)  # [S, B]
        entropy = dist.entropy().sum(dim=-1)  # [S, B]

        values = self.critic_head(critic_outputs.reshape(seq_len * batch_size, -1))
        values = values.reshape(seq_len, batch_size)

        return log_probs, values, entropy


# ── Recurrent Rollout Buffer ─────────────────────────────────────

class RecurrentRolloutBuffer:
    """Stores rollout data + LSTM hidden states for sequence-based training."""
    def __init__(self, num_envs, n_steps, obs_dim, act_dim, lstm_hidden, n_lstm_layers, device):
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

        # Store LSTM hidden states at the START of each sequence chunk
        # We'll compute these during sequence generation
        self.n_lstm_layers = n_lstm_layers
        self.lstm_hidden = lstm_hidden

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute GAE (same as feedforward version)."""
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

    def generate_sequences(self, seq_len, actor_hc_rollout_start, critic_hc_rollout_start):
        """
        Chunk the [n_steps, num_envs] rollout into sequences of length seq_len.
        Each env's rollout is split into n_steps/seq_len non-overlapping sequences.

        Returns list of (obs, actions, dones, log_probs, values, returns, advantages,
                         actor_hc_init, critic_hc_init) tuples.

        actor/critic_hc_rollout_start: hidden states at the start of the rollout [n_layers, N, H].
        We propagate these through dones to get the correct initial state for each chunk.
        """
        n_chunks = self.n_steps // seq_len
        device = self.device

        # Pre-compute initial hidden states for each chunk by tracking resets
        # For chunk 0: use rollout_start hidden states
        # For chunk k: propagate through dones from previous chunks
        # We store per-env hidden states and zero them on dones

        ah, ac = actor_hc_rollout_start  # [L, N, H]
        ch, cc = critic_hc_rollout_start

        chunk_actor_hc = []   # list of (h, c) for each chunk start
        chunk_critic_hc = []

        for chunk_idx in range(n_chunks):
            t_start = chunk_idx * seq_len

            # Save the hidden state at the start of this chunk
            chunk_actor_hc.append((ah.clone(), ac.clone()))
            chunk_critic_hc.append((ch.clone(), cc.clone()))

            # Propagate through this chunk's dones to get state for next chunk
            # We just need to zero out hidden states where episodes ended
            for t in range(t_start, t_start + seq_len):
                done_mask = self.dones[t].unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
                ah = ah * (1.0 - done_mask)
                ac = ac * (1.0 - done_mask)
                ch = ch * (1.0 - done_mask)
                cc = cc * (1.0 - done_mask)

        # Now create sequence tuples: each env x each chunk = one sequence
        # We'll flatten env dimension into batch dimension
        all_obs = []
        all_actions = []
        all_dones = []
        all_log_probs = []
        all_values = []
        all_returns = []
        all_advantages = []
        all_actor_h = []
        all_actor_c = []
        all_critic_h = []
        all_critic_c = []

        for chunk_idx in range(n_chunks):
            t_start = chunk_idx * seq_len
            t_end = t_start + seq_len

            # [seq_len, num_envs, ...]
            all_obs.append(self.obs[t_start:t_end])
            all_actions.append(self.actions_raw[t_start:t_end])
            all_dones.append(self.dones[t_start:t_end])
            all_log_probs.append(self.log_probs[t_start:t_end])
            all_values.append(self.values[t_start:t_end])
            all_returns.append(self.returns[t_start:t_end])
            all_advantages.append(self.advantages[t_start:t_end])

            ah_init, ac_init = chunk_actor_hc[chunk_idx]
            ch_init, cc_init = chunk_critic_hc[chunk_idx]
            all_actor_h.append(ah_init)
            all_actor_c.append(ac_init)
            all_critic_h.append(ch_init)
            all_critic_c.append(cc_init)

        # Stack chunks along the env/batch dimension:
        # obs: n_chunks x [S, N, D] -> [S, n_chunks*N, D]
        obs = torch.cat(all_obs, dim=1)
        actions = torch.cat(all_actions, dim=1)
        dones = torch.cat(all_dones, dim=1)
        log_probs = torch.cat(all_log_probs, dim=1)
        values = torch.cat(all_values, dim=1)
        returns = torch.cat(all_returns, dim=1)
        advantages = torch.cat(all_advantages, dim=1)

        # Hidden states: [L, N] per chunk -> [L, n_chunks*N, H]
        actor_h = torch.cat(all_actor_h, dim=1)
        actor_c = torch.cat(all_actor_c, dim=1)
        critic_h = torch.cat(all_critic_h, dim=1)
        critic_c = torch.cat(all_critic_c, dim=1)

        return (obs, actions, dones, log_probs, values, returns, advantages,
                (actor_h, actor_c), (critic_h, critic_c))


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
    n_steps=256,            # Rollout length (must be divisible by seq_len)
    seq_len=16,             # LSTM sequence length for training
    n_epochs=5,             # Fewer epochs for recurrent (sequences are correlated)
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    lstm_hidden=128,
    n_lstm_layers=1,
    num_seq_per_batch=64,   # Number of sequences per minibatch
    fixed_start=False,
    single_gate=False,
    random_segments=True,
    gate_size=None,
    save_dir="D:/drone2_training/rppo_latest",
    device="cuda",
):
    import shutil
    from datetime import datetime
    device = torch.device(device)

    assert n_steps % seq_len == 0, f"n_steps ({n_steps}) must be divisible by seq_len ({seq_len})"

    # Wipe previous run data
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Cleaned up previous run in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    mode_parts = ["RPPO"]
    if fixed_start:
        mode_parts.append("FIXED START")
    if random_segments:
        mode_parts.append("RANDOM SEGMENTS")
    elif single_gate:
        mode_parts.append("SINGLE GATE")
    if gate_size:
        mode_parts.append(f"GATE={gate_size}m")
    mode = ", ".join(mode_parts)
    print(f"GPU Recurrent PPO Training: {num_envs} envs, {total_timesteps:,} timesteps [{mode}]")
    print(f"LSTM: {n_lstm_layers} layer(s), {lstm_hidden} hidden, seq_len={seq_len}")
    print(f"Device: {device}")

    # Environment
    env = BatchedDroneEnv(num_envs=num_envs, device=device,
                          fixed_start=fixed_start,
                          domain_randomize=not fixed_start,
                          single_gate=single_gate,
                          gate_size_override=gate_size,
                          random_segments=random_segments)

    # Policy
    policy = RecurrentActorCritic(
        obs_dim=OBS_DIM, act_dim=4,
        lstm_hidden=lstm_hidden, n_lstm_layers=n_lstm_layers,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"RecurrentActorCritic parameters: {param_count:,}")

    # Buffer
    buffer = RecurrentRolloutBuffer(
        num_envs, n_steps, OBS_DIM, 4, lstm_hidden, n_lstm_layers, device
    )

    # Observation normalizer
    obs_norm = RunningNorm(OBS_DIM, device)

    # Initial obs and hidden states
    obs = env.reset_all()
    obs_norm.update(obs)
    obs = obs_norm.normalize(obs)
    actor_hc, critic_hc = policy.initial_state(num_envs, device)

    num_iterations = total_timesteps // (num_envs * n_steps)
    global_step = 0
    start_time = time.time()

    # CSV logging
    csv_path = os.path.join(save_dir, "training_stats.csv")
    with open(csv_path, 'w') as f:
        f.write("steps,reward,pg_loss,v_loss,entropy,clip_frac,lr,ent_coef,difficulty,gates_passed,episodes,pass_rate,fps\n")

    # Auto-plotting setup
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_tracking')
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M") + "_rppo_gpu"
    last_plot_time = 0

    print(f"Iterations: {num_iterations}, steps/iter: {num_envs * n_steps:,}")
    print(f"Plots updating every 5 min to: results_tracking/{run_name}/")
    print("-" * 70)

    for iteration in range(1, num_iterations + 1):
        # ── Curriculum ──
        if not fixed_start:
            difficulty = min(1.0, global_step / total_timesteps * 1.25)
            env.set_difficulty(difficulty)
        else:
            difficulty = 0.0

        # ── LR annealing ──
        frac = 1.0 - (iteration - 1) / num_iterations
        for pg in optimizer.param_groups:
            pg['lr'] = lr * frac

        # ── Save hidden states at rollout start (for sequence generation) ──
        rollout_actor_hc = (actor_hc[0].clone(), actor_hc[1].clone())
        rollout_critic_hc = (critic_hc[0].clone(), critic_hc[1].clone())

        # ── Collect rollouts ──
        policy.eval()
        with torch.no_grad():
            for t in range(n_steps):
                action, log_prob, value, action_raw, actor_hc, critic_hc = \
                    policy.get_action(obs, actor_hc, critic_hc)

                next_obs, rewards, dones = env.step(action)
                obs_norm.update(next_obs)
                next_obs_norm = obs_norm.normalize(next_obs)

                buffer.obs[t] = obs
                buffer.actions_raw[t] = action_raw
                buffer.rewards[t] = rewards
                buffer.dones[t] = dones.float()
                buffer.log_probs[t] = log_prob
                buffer.values[t] = value

                # Reset LSTM hidden state for finished episodes
                done_mask = dones.float().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
                actor_hc = (actor_hc[0] * (1.0 - done_mask),
                           actor_hc[1] * (1.0 - done_mask))
                critic_hc = (critic_hc[0] * (1.0 - done_mask),
                            critic_hc[1] * (1.0 - done_mask))

                obs = next_obs_norm
                global_step += num_envs

            # Bootstrap value
            _, _, last_value, _, _ = policy.forward_single(obs, actor_hc, critic_hc)

        # ── Compute advantages ──
        buffer.compute_gae(last_value, gamma, gae_lambda)

        # ── Generate sequences for training ──
        (seq_obs, seq_actions, seq_dones, seq_old_lp, seq_old_values,
         seq_returns, seq_advantages,
         seq_actor_hc, seq_critic_hc) = buffer.generate_sequences(
            seq_len, rollout_actor_hc, rollout_critic_hc
        )

        # Total sequences: n_chunks * num_envs
        total_sequences = seq_obs.shape[1]  # [seq_len, total_sequences, ...]

        # ── PPO update ──
        policy.train()
        clip_fracs = []
        pg_losses = []
        v_losses = []
        ent_losses = []

        for epoch in range(n_epochs):
            # Shuffle sequence indices
            seq_indices = torch.randperm(total_sequences, device=device)

            for start in range(0, total_sequences, num_seq_per_batch):
                end = min(start + num_seq_per_batch, total_sequences)
                mb_idx = seq_indices[start:end]

                mb_obs = seq_obs[:, mb_idx]          # [S, B, D]
                mb_actions = seq_actions[:, mb_idx]   # [S, B, A]
                mb_dones = seq_dones[:, mb_idx]       # [S, B]
                mb_old_lp = seq_old_lp[:, mb_idx]     # [S, B]
                mb_old_values = seq_old_values[:, mb_idx]
                mb_returns = seq_returns[:, mb_idx]
                mb_advantages = seq_advantages[:, mb_idx]

                # Initial hidden states for this minibatch
                mb_ahc = (seq_actor_hc[0][:, mb_idx], seq_actor_hc[1][:, mb_idx])
                mb_chc = (seq_critic_hc[0][:, mb_idx], seq_critic_hc[1][:, mb_idx])

                # Flatten advantages for normalization
                mb_adv_flat = mb_advantages.reshape(-1)
                mb_adv_flat = (mb_adv_flat - mb_adv_flat.mean()) / (mb_adv_flat.std() + 1e-8)
                mb_advantages = mb_adv_flat.reshape(mb_advantages.shape)

                # Forward through LSTM sequences
                new_log_prob, new_value, entropy = policy.evaluate_sequences(
                    mb_obs, mb_actions, mb_dones, mb_ahc, mb_chc
                )

                # Flatten for loss computation
                new_lp = new_log_prob.reshape(-1)
                new_val = new_value.reshape(-1)
                ent = entropy.reshape(-1)
                old_lp = mb_old_lp.reshape(-1)
                old_val = mb_old_values.reshape(-1)
                ret = mb_returns.reshape(-1)
                adv = mb_advantages.reshape(-1)

                # Policy loss
                ratio = (new_lp - old_lp).exp()
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * ratio.clamp(1 - clip_range, 1 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * (new_val - ret).pow(2).mean()

                # Entropy bonus
                ent_loss = ent.mean()

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
            extra = f"Gates: {gates_passed}/{eps_ended} ({pass_rate:.1%})"
            env.gates_passed_count = 0
            env.episodes_ended_count = 0
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
                f"{extra}",
                flush=True,
            )

            # Write CSV row
            with open(csv_path, 'a') as f:
                f.write(f"{global_step},{avg_reward:.4f},{avg_pg:.6f},{avg_vl:.6f},"
                        f"{avg_ent:.4f},{avg_clip:.4f},{cur_lr:.6f},{ent_coef:.6f},"
                        f"{difficulty:.4f},"
                        f"{gates_passed},{eps_ended},{pass_rate:.6f},{fps:.0f}\n")

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
            ckpt_path = os.path.join(save_dir, f"rppo_gpu_{global_step}.pt")
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'obs_norm_mean': obs_norm.mean,
                'obs_norm_var': obs_norm.var,
                'global_step': global_step,
            }, ckpt_path)

    # Save final
    tag = "randseg" if random_segments else ("single" if single_gate else ("fixed" if fixed_start else "gpu"))
    final_path = os.path.join(save_dir, f"rppo_m23_{tag}_final.pt")
    torch.save({
        'policy_state_dict': policy.state_dict(),
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
    parser = argparse.ArgumentParser(description="GPU Recurrent PPO Training for MonoRace M23")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="D:/drone2_training/rppo_latest")
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--single-gate", action="store_true")
    parser.add_argument("--no-random-segments", action="store_true")
    parser.add_argument("--gate-size", type=float, default=None)
    args = parser.parse_args()

    train(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        device=args.device,
        save_dir=args.save_dir,
        lstm_hidden=args.lstm_hidden,
        n_lstm_layers=args.lstm_layers,
        seq_len=args.seq_len,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        fixed_start=args.fixed_start,
        single_gate=args.single_gate,
        random_segments=not args.no_random_segments,
        gate_size=args.gate_size,
    )
