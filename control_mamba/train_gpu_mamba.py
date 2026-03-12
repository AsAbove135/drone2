"""
GPU Mamba-PPO training for MonoRace M23 policy.
Selective State Space Model (S6) actor-critic — pure PyTorch, no mamba-ssm dependency.
Based on train_gpu_rppo.py but replaces LSTM with Mamba blocks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.gpu_env import BatchedDroneEnv
from config import OBS_DIM, TOTAL_TIMESTEPS, NUM_GATES, NUM_LAPS


# ── Selective State Space Model (Mamba Block) ────────────────────

class MambaBlock(nn.Module):
    """
    Single Mamba block: selective SSM with input-dependent discretization.
    Pure PyTorch implementation (no custom CUDA kernels).

    Architecture:
        input → [in_proj → (SSM path + gate path)]
        SSM path: Conv1d → SiLU → selective SSM → * gate → out_proj
        Gate path: SiLU

    Hidden state per block:
        ssm_state: [batch, d_inner, d_state]
        conv_state: [batch, d_inner, d_conv-1]
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = max(1, d_model // 16)

        # Input projection: split into SSM path (x) and gate path (z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv on SSM path
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=0,  # we handle padding manually via conv_state
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameter projections (from activated SSM path)
        # Projects to: B (d_state) + C (d_state) + dt (dt_rank)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # dt projection: dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # Initialize dt bias for reasonable discretization steps
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.bias, -4.0, -2.0)  # softplus → ~0.02-0.13

        # A: structured state matrix (log-scale, negative after exp)
        # Initialized as -[1, 2, ..., d_state] per channel (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection (residual)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def initial_state(self, batch_size, device):
        """Return zero (ssm_state, conv_state)."""
        ssm = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        conv = torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device)
        return ssm, conv

    def forward_single(self, x, state):
        """
        Single-step forward for rollout collection.
        x: [batch, d_model]
        state: (ssm_state [B, d_inner, d_state], conv_state [B, d_inner, d_conv-1])
        Returns: output [batch, d_model], new_state
        """
        ssm_state, conv_state = state

        # Input projection
        xz = self.in_proj(x)  # [B, 2*d_inner]
        x_path, z = xz.chunk(2, dim=-1)  # each [B, d_inner]

        # Causal conv: append new input to conv buffer, run conv
        # conv_state: [B, d_inner, d_conv-1], x_path: [B, d_inner]
        conv_input = torch.cat([conv_state, x_path.unsqueeze(-1)], dim=-1)  # [B, d_inner, d_conv]
        new_conv_state = conv_input[:, :, 1:]  # shift buffer

        # Apply conv weights manually (depthwise)
        x_conv = (conv_input * self.conv1d.weight.squeeze(1)).sum(dim=-1) + self.conv1d.bias  # [B, d_inner]
        x_conv = F.silu(x_conv)

        # SSM parameter projection
        x_proj = self.x_proj(x_conv)  # [B, dt_rank + 2*d_state]
        dt_input, B, C = x_proj.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Discretization step
        dt = F.softplus(self.dt_proj(dt_input))  # [B, d_inner]

        # Discretize A and B
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, d_state]
        dB = dt.unsqueeze(-1) * B.unsqueeze(1)  # [B, d_inner, d_state]

        # SSM recurrence
        new_ssm_state = dA * ssm_state + dB * x_conv.unsqueeze(-1)  # [B, d_inner, d_state]

        # Output
        y = (new_ssm_state * C.unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
        y = y + self.D * x_conv  # skip connection

        # Gate and project
        output = self.out_proj(y * F.silu(z))  # [B, d_model]

        return output, (new_ssm_state, new_conv_state)

    def forward_sequence(self, x_seq, state, dones_seq=None):
        """
        Process a sequence step-by-step (handles done resets).
        x_seq: [seq_len, batch, d_model]
        state: (ssm_state, conv_state)
        dones_seq: [seq_len, batch] or None
        Returns: outputs [seq_len, batch, d_model], final_state
        """
        seq_len = x_seq.shape[0]
        ssm_state, conv_state = state
        outputs = []

        for t in range(seq_len):
            # Reset state where episodes ended at previous step
            if dones_seq is not None and t > 0:
                done_mask = dones_seq[t - 1]  # [B]
                ssm_state = ssm_state * (1.0 - done_mask.unsqueeze(-1).unsqueeze(-1))
                conv_state = conv_state * (1.0 - done_mask.unsqueeze(-1).unsqueeze(-1))

            out, (ssm_state, conv_state) = self.forward_single(x_seq[t], (ssm_state, conv_state))
            outputs.append(out)

        return torch.stack(outputs), (ssm_state, conv_state)


class MambaActorCritic(nn.Module):
    """
    Actor-Critic with Mamba (selective SSM) instead of LSTM.
    Separate Mamba blocks + MLP heads for actor and critic.
    """
    def __init__(self, obs_dim=OBS_DIM, act_dim=4, d_model=64, d_state=16,
                 d_conv=4, expand=2, n_mamba_layers=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.n_mamba_layers = n_mamba_layers

        # Shared input encoding
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model), nn.ReLU(),
        )

        # Separate Mamba blocks for actor and critic (can stack multiple)
        self.actor_mamba = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_mamba_layers)
        ])
        self.critic_mamba = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_mamba_layers)
        ])

        # Layer norms between stacked blocks
        self.actor_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mamba_layers)])
        self.critic_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mamba_layers)])

        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std_min = -2.0
        self.log_std_max = 0.5

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Init
        for module in [self.encoder, self.actor_mean, self.critic_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)
                    nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def initial_state(self, num_envs, device):
        """Return zero states: (actor_states, critic_states) each a list of (ssm, conv) per layer."""
        actor_states = [block.initial_state(num_envs, device) for block in self.actor_mamba]
        critic_states = [block.initial_state(num_envs, device) for block in self.critic_mamba]
        return actor_states, critic_states

    def forward_single(self, obs, actor_states, critic_states):
        """
        Single-step forward for rollout.
        obs: [N, obs_dim]
        actor_states, critic_states: list of (ssm_state, conv_state) per layer
        Returns: mean, std, value, new_actor_states, new_critic_states
        """
        enc = self.encoder(obs)  # [N, d_model]

        # Actor Mamba
        a_out = enc
        new_actor_states = []
        for i, (block, norm) in enumerate(zip(self.actor_mamba, self.actor_norms)):
            residual = a_out
            a_out, new_state = block.forward_single(norm(a_out), actor_states[i])
            a_out = residual + a_out  # residual connection
            new_actor_states.append(new_state)

        # Critic Mamba
        c_out = enc
        new_critic_states = []
        for i, (block, norm) in enumerate(zip(self.critic_mamba, self.critic_norms)):
            residual = c_out
            c_out, new_state = block.forward_single(norm(c_out), critic_states[i])
            c_out = residual + c_out
            new_critic_states.append(new_state)

        mean = self.actor_mean(a_out)
        std = self.actor_log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)
        value = self.critic_head(c_out).squeeze(-1)

        return mean, std, value, new_actor_states, new_critic_states

    def get_action(self, obs, actor_states, critic_states):
        """Sample action during rollout."""
        mean, std, value, new_as, new_cs = self.forward_single(obs, actor_states, critic_states)
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        action = torch.sigmoid(action_raw)
        return action, log_prob, value, action_raw, new_as, new_cs

    def evaluate_sequences(self, obs_seq, actions_seq, dones_seq, actor_states_init, critic_states_init):
        """
        Evaluate sequences for PPO update.
        obs_seq: [seq_len, batch, obs_dim]
        actions_seq: [seq_len, batch, act_dim]
        dones_seq: [seq_len, batch]
        actor/critic_states_init: list of (ssm, conv) per layer
        Returns: log_probs [S, B], values [S, B], entropy [S, B]
        """
        seq_len, batch_size = obs_seq.shape[:2]

        # Encode all at once
        enc = self.encoder(obs_seq.reshape(seq_len * batch_size, -1))
        enc = enc.reshape(seq_len, batch_size, -1)  # [S, B, d_model]

        # Actor Mamba sequence
        a_out = enc
        for i, (block, norm) in enumerate(zip(self.actor_mamba, self.actor_norms)):
            # Apply norm per-step
            normed = norm(a_out.reshape(seq_len * batch_size, -1)).reshape(seq_len, batch_size, -1)
            block_out, _ = block.forward_sequence(normed, actor_states_init[i], dones_seq)
            a_out = a_out + block_out  # residual

        # Critic Mamba sequence
        c_out = enc
        for i, (block, norm) in enumerate(zip(self.critic_mamba, self.critic_norms)):
            normed = norm(c_out.reshape(seq_len * batch_size, -1)).reshape(seq_len, batch_size, -1)
            block_out, _ = block.forward_sequence(normed, critic_states_init[i], dones_seq)
            c_out = c_out + block_out

        # Action distribution
        mean = self.actor_mean(a_out.reshape(seq_len * batch_size, -1))
        mean = mean.reshape(seq_len, batch_size, -1)
        std = self.actor_log_std.clamp(self.log_std_min, self.log_std_max).exp()
        std = std.unsqueeze(0).unsqueeze(0).expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions_seq).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        values = self.critic_head(c_out.reshape(seq_len * batch_size, -1))
        values = values.reshape(seq_len, batch_size)

        return log_probs, values, entropy


# ── Rollout Buffer (adapted for Mamba states) ───────────────────

class MambaRolloutBuffer:
    """Stores rollout data + Mamba hidden states for sequence-based training."""
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

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95, active_steps=None):
        T = active_steps or self.n_steps
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

    def generate_sequences(self, seq_len, actor_states_start, critic_states_start, policy, active_steps=None):
        """
        Chunk rollout into sequences, propagating Mamba states through done resets.

        actor/critic_states_start: list of (ssm_state, conv_state) per layer at rollout start.
        policy: needed to get layer count and dims.

        Returns: (obs, actions, dones, log_probs, values, returns, advantages,
                  chunk_actor_states, chunk_critic_states)
        where chunk_*_states are lists (per layer) of (ssm, conv) with batch dim = n_chunks * num_envs.
        """
        T = active_steps or self.n_steps
        n_chunks = T // seq_len

        # Track per-layer states
        n_layers = len(actor_states_start)
        # Current states: list of (ssm [N, d_inner, d_state], conv [N, d_inner, d_conv-1])
        actor_s = [(s.clone(), c.clone()) for s, c in actor_states_start]
        critic_s = [(s.clone(), c.clone()) for s, c in critic_states_start]

        # Collect chunk initial states
        all_chunk_actor = [[] for _ in range(n_layers)]  # [layer][chunk] = (ssm, conv)
        all_chunk_critic = [[] for _ in range(n_layers)]

        all_obs = []
        all_actions = []
        all_dones = []
        all_log_probs = []
        all_values = []
        all_returns = []
        all_advantages = []

        for chunk_idx in range(n_chunks):
            t_start = chunk_idx * seq_len
            t_end = t_start + seq_len

            # Save initial states for this chunk
            for layer in range(n_layers):
                all_chunk_actor[layer].append((actor_s[layer][0].clone(), actor_s[layer][1].clone()))
                all_chunk_critic[layer].append((critic_s[layer][0].clone(), critic_s[layer][1].clone()))

            # Propagate through dones to get state for next chunk
            for t in range(t_start, t_end):
                done_mask = self.dones[t]  # [N]
                dm_3d = done_mask.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
                for layer in range(n_layers):
                    ssm, conv = actor_s[layer]
                    actor_s[layer] = (ssm * (1.0 - dm_3d), conv * (1.0 - dm_3d))
                    ssm, conv = critic_s[layer]
                    critic_s[layer] = (ssm * (1.0 - dm_3d), conv * (1.0 - dm_3d))

            all_obs.append(self.obs[t_start:t_end])
            all_actions.append(self.actions_raw[t_start:t_end])
            all_dones.append(self.dones[t_start:t_end])
            all_log_probs.append(self.log_probs[t_start:t_end])
            all_values.append(self.values[t_start:t_end])
            all_returns.append(self.returns[t_start:t_end])
            all_advantages.append(self.advantages[t_start:t_end])

        # Stack: [S, n_chunks*N, ...]
        obs = torch.cat(all_obs, dim=1)
        actions = torch.cat(all_actions, dim=1)
        dones = torch.cat(all_dones, dim=1)
        log_probs = torch.cat(all_log_probs, dim=1)
        values = torch.cat(all_values, dim=1)
        returns = torch.cat(all_returns, dim=1)
        advantages = torch.cat(all_advantages, dim=1)

        # States: per layer, cat chunks along batch dim
        chunk_actor_states = []
        chunk_critic_states = []
        for layer in range(n_layers):
            a_ssm = torch.cat([s[0] for s in all_chunk_actor[layer]], dim=0)
            a_conv = torch.cat([s[1] for s in all_chunk_actor[layer]], dim=0)
            chunk_actor_states.append((a_ssm, a_conv))
            c_ssm = torch.cat([s[0] for s in all_chunk_critic[layer]], dim=0)
            c_conv = torch.cat([s[1] for s in all_chunk_critic[layer]], dim=0)
            chunk_critic_states.append((c_ssm, c_conv))

        return (obs, actions, dones, log_probs, values, returns, advantages,
                chunk_actor_states, chunk_critic_states)


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
    seq_len=16,
    n_epochs=5,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2,
    n_mamba_layers=1,
    num_seq_per_batch=64,
    fixed_start=False,
    single_gate=False,
    random_segments=True,
    gate_size=None,
    lambda_prog=None,
    lambda_gate=None,
    lambda_gate_inc=None,
    n_steps_schedule=None,
    curriculum_advance_pct=85.0,
    lr_cycle=50_000_000,
    lr_min=1e-5,
    cosine_lr=True,
    save_dir="D:/drone2_training/mamba_latest",
    device="cuda",
    resume=None,
):
    import shutil
    from datetime import datetime
    device = torch.device(device)

    # Parse n_steps schedule
    if n_steps_schedule:
        schedule = sorted(n_steps_schedule, key=lambda x: x[0])
        current_n_steps = schedule[0][1]
    else:
        schedule = [(0, n_steps)]
        current_n_steps = n_steps

    assert current_n_steps % seq_len == 0, f"n_steps ({current_n_steps}) must be divisible by seq_len ({seq_len})"

    # Wipe previous run data (skip if resuming)
    if resume is None:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print(f"Cleaned up previous run in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    mode_parts = ["MAMBA-PPO"]
    if fixed_start:
        mode_parts.append("FIXED START")
    if random_segments:
        mode_parts.append("RANDOM SEGMENTS")
    elif single_gate:
        mode_parts.append("SINGLE GATE")
    if gate_size:
        mode_parts.append(f"GATE={gate_size}m")
    mode = ", ".join(mode_parts)
    d_inner = d_model * expand
    print(f"GPU Mamba-PPO Training: {num_envs} envs, {total_timesteps:,} timesteps [{mode}]")
    print(f"Mamba: {n_mamba_layers} layer(s), d_model={d_model}, d_inner={d_inner}, "
          f"d_state={d_state}, d_conv={d_conv}, seq_len={seq_len}")
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
                          lambda_gate_inc=lambda_gate_inc)

    # Policy
    policy = MambaActorCritic(
        obs_dim=OBS_DIM, act_dim=4,
        d_model=d_model, d_state=d_state,
        d_conv=d_conv, expand=expand,
        n_mamba_layers=n_mamba_layers,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"MambaActorCritic parameters: {param_count:,}")
    if n_steps_schedule:
        print(f"n_steps schedule: {schedule}")

    # Buffer (allocate for max n_steps)
    max_n_steps = max(s[1] for s in schedule)
    buffer = MambaRolloutBuffer(num_envs, max_n_steps, OBS_DIM, 4, device)

    # Observation normalizer
    obs_norm = RunningNorm(OBS_DIM, device)

    global_step = 0

    # Resume from checkpoint if provided
    if resume is not None:
        print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt['policy_state_dict'])
        obs_norm.mean = ckpt['obs_norm_mean']
        obs_norm.var = ckpt['obs_norm_var']
        global_step = ckpt.get('global_step', 0)
        # Fresh optimizer — don't load old Adam state, it's calibrated to the old reward/task
        print(f"  Loaded policy weights + obs_norm. Fresh optimizer. Resuming from step {global_step:,}")

    # Initial obs and states
    obs = env.reset_all()
    obs_norm.update(obs)
    obs = obs_norm.normalize(obs)
    actor_states, critic_states = policy.initial_state(num_envs, device)
    start_time = time.time()
    iteration = 0

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

    # Auto-plotting setup
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_tracking')
    save_tag = os.path.basename(save_dir.rstrip("/\\"))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M") + f"_{save_tag}"
    last_plot_time = 0.0

    # Sequential gate curriculum
    CURRICULUM_ADVANCE_PCT = curriculum_advance_pct
    CURRICULUM_STAGE_MIN = 1
    curriculum_stage = 1
    curriculum_stage_steps = 0  # steps at which current stage started

    # Gate size curriculum: 1.5m -> 0.5m over first 20% of training
    GATE_SIZE_START, GATE_SIZE_END = 1.5, 0.5
    GATE_CURRICULUM_FRAC = 0.20

    num_iterations = total_timesteps // (num_envs * current_n_steps)
    print(f"Starting Mamba-PPO training (~{num_iterations} iterations)...")

    while global_step < total_timesteps:
        iteration += 1

        # n_steps schedule
        new_n_steps = current_n_steps
        for threshold, ns in schedule:
            if global_step >= threshold:
                new_n_steps = ns
        if new_n_steps != current_n_steps:
            assert new_n_steps % seq_len == 0, f"n_steps ({new_n_steps}) must be divisible by seq_len ({seq_len})"
            print(f"  [Schedule] n_steps: {current_n_steps} -> {new_n_steps} at step {global_step:,}")
            current_n_steps = new_n_steps
            num_iterations = total_timesteps // (num_envs * current_n_steps)

        # Gate size curriculum (skip if fixed gate size provided)
        if gate_size is None:
            gate_frac = min(1.0, global_step / (total_timesteps * GATE_CURRICULUM_FRAC))
            difficulty = gate_frac
            env.gate_size = GATE_SIZE_START + (GATE_SIZE_END - GATE_SIZE_START) * gate_frac
        else:
            difficulty = 1.0

        # LR schedule
        if cosine_lr:
            cycle_progress = (global_step % lr_cycle) / lr_cycle
            lr_now = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(2.0 * math.pi * cycle_progress))
        else:
            lr_now = lr * (1.0 - global_step / total_timesteps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # Save rollout-start states for sequence generation
        rollout_actor_start = [(s.clone(), c.clone()) for s, c in actor_states]
        rollout_critic_start = [(s.clone(), c.clone()) for s, c in critic_states]

        # ── Collect rollout ──
        policy.eval()
        with torch.no_grad():
            for step in range(current_n_steps):
                action, log_prob, value, action_raw, actor_states, critic_states = \
                    policy.get_action(obs, actor_states, critic_states)

                next_obs, rewards, dones = env.step(action)
                obs_norm.update(next_obs)
                next_obs = obs_norm.normalize(next_obs)

                buffer.obs[step] = obs
                buffer.actions_raw[step] = action_raw
                buffer.rewards[step] = rewards
                buffer.dones[step] = dones.float()
                buffer.log_probs[step] = log_prob
                buffer.values[step] = value

                # Reset states for done envs
                if dones.any():
                    keep_mask = (~dones).float().unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
                    for layer in range(len(actor_states)):
                        ssm, conv = actor_states[layer]
                        actor_states[layer] = (ssm * keep_mask, conv * keep_mask)
                        ssm, conv = critic_states[layer]
                        critic_states[layer] = (ssm * keep_mask, conv * keep_mask)

                obs = next_obs
                global_step += num_envs

            # Bootstrap value
            _, _, last_value, _, _ = policy.forward_single(obs, actor_states, critic_states)

        buffer.compute_gae(last_value, gamma, gae_lambda, active_steps=current_n_steps)

        # ── PPO Update ──
        policy.train()
        (seq_obs, seq_actions, seq_dones, seq_old_lp, seq_old_v,
         seq_returns, seq_advantages,
         chunk_actor_states, chunk_critic_states) = \
            buffer.generate_sequences(seq_len, rollout_actor_start, rollout_critic_start, policy, active_steps=current_n_steps)

        total_seqs = seq_obs.shape[1]  # n_chunks * num_envs
        seq_indices = torch.arange(total_seqs, device=device)

        pg_losses, v_losses, ent_losses, clip_fracs = [], [], [], []

        for epoch in range(n_epochs):
            perm = torch.randperm(total_seqs, device=device)
            for start in range(0, total_seqs, num_seq_per_batch):
                end = min(start + num_seq_per_batch, total_seqs)
                idx = perm[start:end]

                mb_obs = seq_obs[:, idx]
                mb_act = seq_actions[:, idx]
                mb_done = seq_dones[:, idx]
                mb_old_lp = seq_old_lp[:, idx]
                mb_ret = seq_returns[:, idx]
                mb_adv = seq_advantages[:, idx]

                # Gather initial states for this minibatch
                mb_actor_states = [(s[idx], c[idx]) for s, c in chunk_actor_states]
                mb_critic_states = [(s[idx], c[idx]) for s, c in chunk_critic_states]

                new_lp, new_val, ent = policy.evaluate_sequences(
                    mb_obs, mb_act, mb_done, mb_actor_states, mb_critic_states)

                # Flatten time dimension for loss
                adv = mb_adv.reshape(-1)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                old_lp = mb_old_lp.reshape(-1)
                new_lp = new_lp.reshape(-1)
                ret = mb_ret.reshape(-1)
                new_val = new_val.reshape(-1)
                ent = ent.reshape(-1)

                ratio = (new_lp - old_lp).exp()
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (new_val - ret).pow(2).mean()
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

        if True:  # log every iteration
            avg_reward = buffer.rewards[:current_n_steps].sum(dim=0).mean().item()
            gates_passed = env.gates_passed_count
            eps_ended = env.episodes_ended_count
            pass_rate = gates_passed / max(eps_ended, 1)
            env.gates_passed_count = 0
            env.episodes_ended_count = 0

            # Reward components: per-env-per-step averages
            total_steps_in_period = current_n_steps * num_envs * 5
            rc = {k: v / max(total_steps_in_period, 1) for k, v in env.reward_components.items()}
            env.reward_components = {k: 0.0 for k in env.reward_components}

            # Gate distribution
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
            ckpt_path = os.path.join(save_dir, f"mamba_gpu_{global_step}.pt")
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'obs_norm_mean': obs_norm.mean,
                'obs_norm_var': obs_norm.var,
                'global_step': global_step,
            }, ckpt_path)

    # Save final
    tag = "randseg" if random_segments else ("single" if single_gate else ("fixed" if fixed_start else "gpu"))
    final_path = os.path.join(save_dir, f"mamba_m23_{tag}_final.pt")
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
    except Exception as e:
        print(f"Warning: final auto-plot failed: {e}")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Training complete! {global_step:,} steps in {elapsed:.1f}s ({global_step/elapsed:,.0f} FPS)")
    print(f"Model saved to {final_path}")
    print('To keep this run: python control/save_run.py "description"')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Mamba-PPO Training for MonoRace M23")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="D:/drone2_training/mamba_latest")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--mamba-layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--n-steps-schedule", type=str, default=None,
                        help="Format: '0:256,10000000:512'")
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--num-seq-per-batch", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--single-gate", action="store_true")
    parser.add_argument("--no-random-segments", action="store_true")
    parser.add_argument("--gate-size", type=float, default=None)
    parser.add_argument("--lambda-prog", type=float, default=None)
    parser.add_argument("--lambda-gate", type=float, default=None)
    parser.add_argument("--lambda-gate-inc", type=float, default=None)
    parser.add_argument("--curriculum-advance-pct", type=float, default=85.0,
                        help="Advance curriculum stage when this %% of episodes pass >= stage gates")
    parser.add_argument("--lr-cycle", type=int, default=50_000_000,
                        help="Cosine LR warm restart cycle length in steps")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="Minimum LR for cosine schedule")
    parser.add_argument("--cosine-lr", action=argparse.BooleanOptionalAction, default=True,
                        help="Use cosine LR with warm restarts (--no-cosine-lr to disable)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    args = parser.parse_args()

    # Parse n_steps schedule
    n_steps_schedule = None
    if args.n_steps_schedule:
        n_steps_schedule = []
        for part in args.n_steps_schedule.split(","):
            step_str, ns_str = part.strip().split(":")
            n_steps_schedule.append((int(step_str), int(ns_str)))

    train(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        device=args.device,
        save_dir=args.save_dir,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        n_mamba_layers=args.mamba_layers,
        seq_len=args.seq_len,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        num_seq_per_batch=args.num_seq_per_batch,
        ent_coef=args.ent_coef,
        fixed_start=args.fixed_start,
        single_gate=args.single_gate,
        random_segments=not args.no_random_segments,
        gate_size=args.gate_size,
        lambda_prog=args.lambda_prog,
        lambda_gate=args.lambda_gate,
        lambda_gate_inc=args.lambda_gate_inc,
        n_steps_schedule=n_steps_schedule,
        curriculum_advance_pct=args.curriculum_advance_pct,
        lr_cycle=args.lr_cycle,
        lr_min=args.lr_min,
        cosine_lr=args.cosine_lr,
        resume=args.resume,
    )
