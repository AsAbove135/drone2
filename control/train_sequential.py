"""
Sequential gate-to-gate training for MonoRace M23.
Trains segment-by-segment: start→G1, G1→G2, ..., G10→G11.
Advances to next segment when success rate exceeds threshold.
Tests final model on full 11-gate course.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.gpu_env import BatchedDroneEnv
from control.train_gpu import ActorCritic, RolloutBuffer, RunningNorm
from config import OBS_DIM, NUM_GATES


def evaluate_success_rate(env, policy, obs_norm, n_episodes=200):
    """Run n_episodes and return gate passage success rate."""
    env.gates_passed_count = 0
    env.episodes_ended_count = 0

    obs = env.reset_all()
    episodes_done = 0

    while episodes_done < n_episodes:
        obs_n = obs_norm.normalize(obs)
        with torch.no_grad():
            mean, std, val = policy(obs_n)
            action = torch.sigmoid(mean)
        obs, rew, done = env.step(action)
        episodes_done = env.episodes_ended_count

    passed = env.gates_passed_count
    total = env.episodes_ended_count
    return passed / max(total, 1)


def train_sequential(
    num_envs=2048,
    gate_size=0.8,
    success_threshold=0.90,
    max_steps_per_segment=30_000_000,
    eval_every=25,          # evaluate every N iterations
    eval_episodes=500,
    n_steps=64,
    batch_size=4096,
    n_epochs=10,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    max_segments=None,      # None = all segments, else stop after N
    save_dir="D:/drone2_training/sequential",
    device="cuda",
):
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Sequential Gate-to-Gate Training")
    print(f"  {num_envs} envs, gate_size={gate_size}m, threshold={success_threshold*100:.0f}%")
    print(f"  Max {max_steps_per_segment:,} steps per segment")
    print(f"  Device: {device}")
    print("=" * 70)

    # Policy (shared across all segments)
    policy = ActorCritic(obs_dim=OBS_DIM, act_dim=4).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    obs_norm = RunningNorm(OBS_DIM, device)
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"ActorCritic parameters: {param_count:,}")

    # Environment
    env = BatchedDroneEnv(num_envs=num_envs, device=device,
                          domain_randomize=False,
                          gate_size_override=gate_size)

    buffer = RolloutBuffer(num_envs, n_steps, OBS_DIM, 4, device)

    global_step = 0
    start_time = time.time()

    # Define segments: (from_gate, to_gate)
    # from_gate=None means start position (before gate 0)
    segments = [(None, 0)]  # start ->gate 1
    for i in range(NUM_GATES - 1):
        segments.append((i, i + 1))  # gate i ->gate i+1

    if max_segments is not None:
        segments = segments[:max_segments]

    print(f"Segments to train: {len(segments)}")
    for i, (fg, tg) in enumerate(segments):
        src = "START" if fg is None else f"G{fg+1}"
        print(f"  Segment {i}: {src} ->G{tg+1}")
    print("-" * 70)

    for seg_idx, (from_gate, to_gate) in enumerate(segments):
        src = "START" if from_gate is None else f"G{from_gate+1}"
        print(f"\n{'='*70}")
        print(f"SEGMENT {seg_idx}: {src} ->G{to_gate+1}")
        print(f"{'='*70}")

        env.set_segment(from_gate, to_gate)
        obs = env.reset_all()
        obs_norm.update(obs)
        obs = obs_norm.normalize(obs)

        seg_step = 0
        seg_start = time.time()
        iteration = 0
        best_success = 0.0

        # Reset exploration for each segment: restore log_std to 0
        # This prevents entropy collapse when moving to harder segments
        with torch.no_grad():
            policy.actor_log_std.zero_()

        # Reset LR and optimizer momentum for each segment
        optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

        while seg_step < max_steps_per_segment:
            iteration += 1

            # Constant LR within segment (reset at segment start)
            # No annealing — segments are variable length

            # Collect rollouts
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
                    seg_step += num_envs
                    global_step += num_envs

                _, _, last_value = policy(obs)

            buffer.compute_gae(last_value, gamma, gae_lambda)

            # PPO update
            policy.train()
            b_obs, b_actions, b_old_lp, b_old_val, b_returns, b_advantages = buffer.flatten()
            total_samples = num_envs * n_steps

            pg_losses, v_losses, ent_vals = [], [], []
            for epoch in range(n_epochs):
                indices = torch.randperm(total_samples, device=device)
                for start in range(0, total_samples, batch_size):
                    end = min(start + batch_size, total_samples)
                    mb_idx = indices[start:end]

                    mb_adv = b_advantages[mb_idx]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    new_lp, new_val, entropy = policy.evaluate(b_obs[mb_idx], b_actions[mb_idx])
                    ratio = (new_lp - b_old_lp[mb_idx]).exp()
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * ratio.clamp(1 - clip_range, 1 + clip_range)
                    pg_loss = torch.max(pg1, pg2).mean()
                    v_loss = 0.5 * (new_val - b_returns[mb_idx]).pow(2).mean()
                    ent_loss = entropy.mean()

                    loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    ent_vals.append(ent_loss.item())

            # Logging
            if iteration % 10 == 0 or iteration == 1:
                avg_reward = buffer.rewards.sum(dim=0).mean().item()
                elapsed = time.time() - start_time
                fps = global_step / elapsed
                print(
                    f"  Seg{seg_idx} Iter {iteration:4d} | "
                    f"Steps: {seg_step:>10,} | "
                    f"FPS: {fps:>8,.0f} | "
                    f"Reward: {avg_reward:>8.3f} | "
                    f"Entropy: {sum(ent_vals)/len(ent_vals):.3f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

            # Evaluate success rate periodically
            if iteration % eval_every == 0:
                policy.eval()
                success_rate = evaluate_success_rate(
                    env, policy, obs_norm, n_episodes=eval_episodes)
                best_success = max(best_success, success_rate)
                print(
                    f"  >>> EVAL: success={success_rate*100:.1f}% "
                    f"(best={best_success*100:.1f}%, "
                    f"threshold={success_threshold*100:.0f}%)",
                    flush=True,
                )

                # Save segment checkpoint
                ckpt_path = os.path.join(save_dir, f"seg{seg_idx}_sr{success_rate*100:.0f}.pt")
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'obs_norm_mean': obs_norm.mean,
                    'obs_norm_var': obs_norm.var,
                    'global_step': global_step,
                    'segment': seg_idx,
                    'success_rate': success_rate,
                }, ckpt_path)

                if success_rate >= success_threshold:
                    print(f"  >>> PASSED! Moving to next segment.")
                    break

                # Re-reset env for continued training
                obs = env.reset_all()
                obs_norm.update(obs)
                obs = obs_norm.normalize(obs)

        seg_elapsed = time.time() - seg_start
        print(f"  Segment {seg_idx} done: {seg_step:,} steps, {seg_elapsed:.1f}s, "
              f"best_success={best_success*100:.1f}%")

    # Save final model
    final_path = os.path.join(save_dir, "gcnet_sequential_final.pt")
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'obs_norm_mean': obs_norm.mean,
        'obs_norm_var': obs_norm.var,
        'global_step': global_step,
    }, final_path)

    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Sequential training complete! {global_step:,} total steps in {total_elapsed:.1f}s")
    print(f"Model saved to {final_path}")

    # Test on trained segments
    n_test_gates = len(segments)
    print(f"\n{'='*70}")
    print(f"COURSE TEST ({n_test_gates} gates)")
    print(f"{'='*70}")
    test_full_course(policy, obs_norm, device, gate_size, save_dir, max_gates=n_test_gates)


def test_full_course(policy, obs_norm, device, gate_size, save_dir, max_gates=None):
    """Test the trained policy on the course (or first max_gates gates)."""
    env = BatchedDroneEnv(num_envs=1, device=device,
                          domain_randomize=False,
                          gate_size_override=gate_size)
    # No segment — full course mode, spawn at start
    env.fixed_start = True
    env.single_gate = False
    test_gates = max_gates or NUM_GATES

    policy.eval()
    results = []

    for ep in range(10):
        env.reset_all()
        obs = env.get_obs()
        total_r = 0
        max_gate = 0
        positions = []

        for step in range(8000):  # 16 seconds
            obs_n = obs_norm.normalize(obs)
            with torch.no_grad():
                mean, std, val = policy(obs_n)
                action = torch.sigmoid(mean)

            obs, rew, done = env.step(action)
            total_r += rew[0].item()
            gi = env.gate_idx[0].item()
            max_gate = max(max_gate, gi)
            positions.append(env.prev_states[0, 0:3].cpu().numpy().copy())

            if done[0]:
                break

        import numpy as np
        positions = np.array(positions)
        results.append({
            'steps': step + 1,
            'reward': total_r,
            'gates': max_gate,
            'positions': positions,
        })
        print(f"  Ep{ep+1:2d}: {step+1:5d} steps, R={total_r:8.1f}, gates={max_gate}/{NUM_GATES}")

    # Save results for visualization
    import numpy as np
    np.savez(os.path.join(save_dir, "full_course_results.npz"),
             **{f"pos_{i}": r['positions'] for i, r in enumerate(results)},
             gates=[r['gates'] for r in results],
             rewards=[r['reward'] for r in results],
             steps=[r['steps'] for r in results])

    avg_gates = sum(r['gates'] for r in results) / len(results)
    max_gates = max(r['gates'] for r in results)
    print(f"\n  Average gates: {avg_gates:.1f}/{NUM_GATES}, Best: {max_gates}/{NUM_GATES}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sequential gate-to-gate training")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--gate-size", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max-steps", type=int, default=30_000_000,
                        help="Max steps per segment before giving up")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="Evaluate success rate every N iterations")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Max segments to train (default: all)")
    parser.add_argument("--save-dir", type=str, default="D:/drone2_training/sequential")
    args = parser.parse_args()

    train_sequential(
        num_envs=args.num_envs,
        gate_size=args.gate_size,
        success_threshold=args.threshold,
        max_steps_per_segment=args.max_steps,
        eval_every=args.eval_every,
        max_segments=args.max_segments,
        device=args.device,
        save_dir=args.save_dir,
    )
