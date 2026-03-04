import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from control.dynamics import QuadcopterDynamics

class GCNet(nn.Module):
    """
    Guidance & Control Network (G&CNet) policy architecture.
    A lightweight Multi-Layer Perceptron (MLP) mapping estimated states 
    to low-level motor commands (500 Hz control loop).
    """
    def __init__(self, state_dim=15, action_dim=4): # 15: [gate_rel_pos(3), v(3), q(4), w(3), last_action(2?)] 
        super(GCNet, self).__init__()
        
        # Simple 3-layer MLP as typically used for these aggressive quadcopter policies
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Sigmoid() # Outputs motor commands in [0, 1]
        )
        
    def forward(self, x):
        return self.net(x)

class MonoRaceSimEnv(gym.Env):
    """
    Custom Gymnasium Environment wrapping the PyTorch QuadcopterDynamics 
    for PPO training via Stable Baselines 3.
    """
    def __init__(self):
        super(MonoRaceSimEnv, self).__init__()
        
        # State: p(3), v(3), q(4), w(3), motor_speeds(4) = 17 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        # Actions: 4 motor commands [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.dynamics = QuadcopterDynamics(dt=0.01)
        self.state = None
        self.step_count = 0
        self.max_steps = 1000
        
    def _get_obs(self):
        return self.state.cpu().numpy().squeeze(0)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Domain Randomization can be applied here
        # e.g. self.dynamics.mass = np.random.uniform(0.9, 1.1) * 0.966
        
        # Ground initialization around (8, -22, 0)
        p = torch.tensor([[8.0 + np.random.uniform(-1, 1), 
                           -22.0 + np.random.uniform(-1, 1), 
                           0.0]])
        v = torch.zeros((1, 3))
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]) # Unit quaternion
        w = torch.zeros((1, 3))
        motors = torch.zeros((1, 4))
        
        self.state = torch.cat([p, v, q, w, motors], dim=-1)
        self.step_count = 0
        
        return self._get_obs(), {}
        
    def step(self, action):
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        # Advance simulation
        with torch.no_grad():
            self.state = self.dynamics(self.state, action_tensor)
            
        self.step_count += 1
        
        obs = self._get_obs()
        
        # Reward Function (Placeholder for progressing track)
        # E.g. penalty for control effort, reward for velocity towards next gate
        v_norm = np.linalg.norm(obs[3:6])
        reward = v_norm * 0.1 - 0.01 * np.sum(action**2)
        
        # Terminate if crashed (z < -0.5) or out of bounds
        terminated = bool(obs[2] > 0.5) # Z is down positive in NED, if it goes too high (negative) or hits ground
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, {}

def train_ppo():
    """
    Sets up the Gym environment and trains the PPO agent.
    """
    print("Setting up MonoRace Simulation Environment...")
    env = DummyVecEnv([lambda: MonoRaceSimEnv()])
    
    # Initialize PPO with GCNet policy (MlpPolicy is close enough for demonstration)
    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=256)
    
    # In a real scenario, this would train for millions of timesteps
    print("Starting Training (Dummy Run)...")
    # model.learn(total_timesteps=10000)
    # model.save("gcnet_policy")
    print("PPO framework setup complete.")

if __name__ == "__main__":
    train_ppo()
