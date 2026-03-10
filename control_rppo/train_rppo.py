"""
Recurrent PPO training for MonoRace M23 policy.
Uses sb3-contrib's RecurrentPPO with LSTM policy.
Reuses MonoRaceSimEnv from control/ppo_train.py.
"""
import time
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.ppo_train import MonoRaceSimEnv, GCNet
from config import (
    OBS_DIM, NUM_ENVS, TOTAL_TIMESTEPS,
    GATE_POSITIONS, NUM_LAPS, TRACKS,
)


def train_rppo(
    near_gate_spawn=True,
    num_envs=NUM_ENVS,
    total_timesteps=TOTAL_TIMESTEPS,
    lstm_hidden_size=128,
    n_lstm_layers=1,
    net_arch_pi=(64, 64),
    net_arch_vf=(64, 64),
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    ent_coef=0.005,
    save_dir="D:/drone2_training/rppo_latest",
):
    """Train M23 policy with Recurrent PPO (LSTM) via sb3-contrib."""
    import shutil
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    class GateCurriculumCallback(BaseCallback):
        """Linear curriculum: gate size 1.5->0.5m, spawn dist 2-3m->2-20m."""
        CURRICULUM_STEPS = 2_000_000
        GATE_START, GATE_END = 1.5, 0.5
        DIST_MAX_START, DIST_MAX_END = 3.0, 20.0

        def __init__(self, csv_path=None, **kwargs):
            super().__init__(verbose=0)
            self._last_log = 0
            self._ep_gate_counts = []
            self._ep_rewards = []
            self._csv_path = csv_path
            self._csv_initialized = False

        def _init_csv(self):
            if self._csv_path and not self._csv_initialized:
                num_gates = len(GATE_POSITIONS) * NUM_LAPS
                with open(self._csv_path, 'w') as f:
                    cols = ["steps", "reward_mean", "gate_pass_rate", "avg_gates",
                            "gate_size", "spawn_dist_max", "frac",
                            "entropy", "explained_var", "approx_kl", "clip_fraction"]
                    cols += [f"pct_ge_{i}" for i in range(1, num_gates + 1)]
                    f.write(",".join(cols) + "\n")
                self._csv_initialized = True

        def _on_step(self):
            self._init_csv()
            frac = min(1.0, self.num_timesteps / self.CURRICULUM_STEPS)

            MonoRaceSimEnv.gate_size_current = self.GATE_START + (self.GATE_END - self.GATE_START) * frac
            MonoRaceSimEnv.spawn_dist_min = 2.0
            MonoRaceSimEnv.spawn_dist_max = self.DIST_MAX_START + (self.DIST_MAX_END - self.DIST_MAX_START) * frac

            # Collect per-episode stats from info dicts
            for info in self.locals.get("infos", []):
                src = info
                if "terminal_info" in info:
                    src = info["terminal_info"]
                if "ep_gates_passed" in src:
                    self._ep_gate_counts.append(src["ep_gates_passed"])
                if "ep_reward" in src:
                    self._ep_rewards.append(src["ep_reward"])

            # Log every 100K steps
            if self.num_timesteps - self._last_log >= 100_000:
                self._last_log = self.num_timesteps
                try:
                    gates_list = self.training_env.get_attr('gates_passed')
                    eps_list = self.training_env.get_attr('episodes_count')
                    total_gates = sum(gates_list)
                    total_eps = sum(eps_list)
                    for i in range(len(gates_list)):
                        self.training_env.env_method('__setattr__', 'gates_passed', 0, indices=[i])
                        self.training_env.env_method('__setattr__', 'episodes_count', 0, indices=[i])
                except Exception:
                    total_gates = 0
                    total_eps = 0
                rate = total_gates / max(total_eps, 1)

                # Gate distribution from collected episode data
                num_gates_total = len(GATE_POSITIONS) * NUM_LAPS
                ep_counts = self._ep_gate_counts if self._ep_gate_counts else [0]
                avg_gates = np.mean(ep_counts)
                n_eps = len(ep_counts)
                pct_ge = []
                for threshold in range(1, num_gates_total + 1):
                    pct = sum(1 for c in ep_counts if c >= threshold) / max(n_eps, 1) * 100
                    pct_ge.append(pct)

                rew_mean = np.mean(self._ep_rewards) if self._ep_rewards else 0.0

                # PPO training stats
                logger_vals = getattr(self.model, 'logger', None)
                name_to_val = getattr(logger_vals, 'name_to_value', {}) if logger_vals else {}
                entropy = -name_to_val.get("train/entropy_loss", 0.0)
                explained_var = name_to_val.get("train/explained_variance", 0.0)
                approx_kl = name_to_val.get("train/approx_kl", 0.0)
                clip_fraction = name_to_val.get("train/clip_fraction", 0.0)

                # Log to tensorboard
                self.logger.record("curriculum/ep_reward_mean", rew_mean)
                self.logger.record("curriculum/gate_pass_rate", rate * 100)
                self.logger.record("curriculum/gate_size", MonoRaceSimEnv.gate_size_current)
                self.logger.record("curriculum/spawn_dist_max", MonoRaceSimEnv.spawn_dist_max)
                self.logger.record("curriculum/frac", frac)
                self.logger.record("curriculum/avg_gates", avg_gates)

                # Write CSV row
                if self._csv_path:
                    with open(self._csv_path, 'a') as f:
                        row = [self.num_timesteps, rew_mean, rate * 100, avg_gates,
                               MonoRaceSimEnv.gate_size_current, MonoRaceSimEnv.spawn_dist_max, frac,
                               entropy, explained_var, approx_kl, clip_fraction]
                        row += pct_ge
                        f.write(",".join(f"{v:.4f}" for v in row) + "\n")

                print(f"  [RPPO] Steps: {self.num_timesteps:>10,} | "
                      f"Size: {MonoRaceSimEnv.gate_size_current:.2f}m | "
                      f"Spawn: 2.0-{MonoRaceSimEnv.spawn_dist_max:.1f}m | "
                      f"Frac: {frac:.2f} | "
                      f"Passed: {total_gates}/{total_eps} eps ({rate:.1%}) | "
                      f"Avg gates: {avg_gates:.2f} | "
                      f">=1: {pct_ge[0]:.0f}%",
                      flush=True)

                self._ep_gate_counts = []
                self._ep_rewards = []
            return True

    # Wipe previous run data
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Cleaned up previous run in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    MonoRaceSimEnv.track_pool = None

    spawn_mode = "near-gate" if near_gate_spawn else "uniform"
    print(f"Setting up Recurrent PPO Training (spawn: {spawn_mode})...")
    print(f"LSTM: {n_lstm_layers} layer(s), {lstm_hidden_size} hidden units")

    def make_env(ngs):
        def _init():
            return MonoRaceSimEnv(near_gate_spawn=ngs)
        return _init

    env = SubprocVecEnv([make_env(near_gate_spawn) for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers,
        # Separate actor/critic MLP heads after the LSTM
        net_arch=dict(pi=list(net_arch_pi), vf=list(net_arch_vf)),
        activation_fn=nn.ReLU,
        shared_lstm=False,  # Separate LSTM for actor and critic
        enable_critic_lstm=True,  # Critic also gets its own LSTM
    )

    # Linear LR decay: 3e-4 -> 0 over training
    def lr_schedule(progress_remaining):
        return 3e-4 * progress_remaining

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        verbose=1,
        device="cpu",
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
    )

    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"RecurrentPPO parameters: {param_count:,}")

    checkpoint_cb = CheckpointCallback(
        save_freq=1_000_000 // num_envs,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="rppo_m23",
    )

    class VecNormSaveCallback(BaseCallback):
        """Save VecNormalize stats alongside each checkpoint."""
        def __init__(self, save_freq, save_path):
            super().__init__(verbose=0)
            self._save_freq = save_freq
            self._save_path = save_path
        def _on_step(self):
            if self.num_timesteps % self._save_freq < self.training_env.num_envs:
                vn_path = os.path.join(self._save_path,
                    f"vecnormalize_{self.num_timesteps}_steps.pkl")
                self.model.get_vec_normalize_env().save(vn_path)
            return True

    vecnorm_cb = VecNormSaveCallback(
        save_freq=1_000_000,
        save_path=os.path.join(save_dir, "checkpoints"),
    )
    csv_log_path = os.path.join(save_dir, "training_stats.csv")
    gate_cb = GateCurriculumCallback(csv_path=csv_log_path)

    class PeriodicPlotCallback(BaseCallback):
        """Regenerate result plots every N seconds during training."""
        def __init__(self, csv_path, results_dir, run_name, interval_sec=300, **kwargs):
            super().__init__(verbose=0)
            self._csv_path = csv_path
            self._results_dir = results_dir
            self._run_name = run_name
            self._interval = interval_sec
            self._last_plot_time = 0

        def _on_step(self):
            now = time.time()
            if now - self._last_plot_time >= self._interval:
                self._last_plot_time = now
                try:
                    from control.plot_training import generate_charts
                    generate_charts(self._csv_path, self._results_dir, self._run_name)
                    print(f"  [Plots] Updated charts in {self._results_dir}/{self._run_name}/",
                          flush=True)
                except Exception as e:
                    print(f"  [Plots] Failed: {e}", flush=True)
            return True

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_tracking')
    from datetime import datetime
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M") + "_rppo_lstm128"

    plot_cb = PeriodicPlotCallback(
        csv_path=csv_log_path, results_dir=results_dir,
        run_name=run_name, interval_sec=300,
    )

    print(f"Starting Recurrent PPO Training ({total_timesteps:,} timesteps, {num_envs} envs)...")
    print(f"Curriculum: gate 1.5m -> 0.5m, spawn 2-3m -> 2-20m, full by 2M steps")
    print(f"Plots updating every 5 min to: results_tracking/{run_name}/")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, vecnorm_cb, gate_cb, plot_cb],
    )

    model.save(os.path.join(save_dir, "rppo_m23_final"))
    env.save(os.path.join(save_dir, "vecnormalize_m23.pkl"))
    print(f"Training complete. Model saved to {save_dir}")
    print('To keep this run: python control/save_run.py "description"')

    # Final plot generation
    try:
        from control.plot_training import generate_charts
        generate_charts(csv_log_path, results_dir, run_name)
    except Exception as e:
        print(f"Warning: final auto-plot failed: {e}")


def _cleanup_children():
    """Kill any leftover subprocess workers on exit."""
    import multiprocessing
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=5)


if __name__ == "__main__":
    import argparse
    import signal
    import atexit

    atexit.register(_cleanup_children)
    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted — cleaning up..."), _cleanup_children(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup_children(), sys.exit(1)))

    parser = argparse.ArgumentParser(description="Recurrent PPO Training for MonoRace M23")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--lstm-hidden", type=int, default=128,
                        help="LSTM hidden size (default: 128)")
    parser.add_argument("--lstm-layers", type=int, default=1,
                        help="Number of LSTM layers (default: 1)")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="Rollout length per env (default: 512)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Minibatch size for LSTM sequences (default: 128)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--uniform-spawn", action="store_true",
                        help="Use uniform random spawn (default: near-gate)")
    parser.add_argument("--save-dir", type=str, default="D:/drone2_training/rppo_latest")
    args = parser.parse_args()

    try:
        train_rppo(
            near_gate_spawn=not args.uniform_spawn,
            num_envs=args.num_envs,
            total_timesteps=args.timesteps,
            lstm_hidden_size=args.lstm_hidden,
            n_lstm_layers=args.lstm_layers,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            save_dir=args.save_dir,
        )
    finally:
        _cleanup_children()
