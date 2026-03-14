"""
Experiment configuration: defines what to run, why, and how to evaluate it.
Each experiment is a JSON file that fully specifies a reproducible training run.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime


@dataclass
class ExperimentConfig:
    # ── Identity ──
    name: str                          # Short slug, e.g. "rppo_higher_ent"
    hypothesis: str                    # What we expect to learn
    rationale: str                     # Why we think this will work
    parent: Optional[str] = None      # Previous experiment this builds on

    # ── Model ──
    model_type: str = "fsppo"         # ppo | rppo | mamba | fsppo
    frame_stack: int = 16             # FSPPO only
    lstm_hidden: int = 128            # RPPO only
    lstm_layers: int = 1              # RPPO only
    seq_len: int = 16                 # RPPO/Mamba only

    # ── Training ──
    num_envs: int = 2048
    total_timesteps: int = 100_000_000
    n_steps: int = 512
    n_epochs: int = 5
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    cosine_lr: bool = True
    lr_cycle: int = 50_000_000
    lr_min: float = 1e-5

    # ── PPO-specific (feedforward only) ──
    batch_size: int = 4096            # PPO minibatch size
    ent_coef_start: float = 0.01     # PPO entropy decay start
    ent_coef_end: float = 0.001      # PPO entropy decay end
    log_std_max_start: float = 0.5
    log_std_max_end: float = -1.0

    # ── RPPO-specific ──
    num_seq_per_batch: int = 64       # RPPO sequences per minibatch

    # ── Environment ──
    random_segments: bool = True
    fixed_start: bool = False
    single_gate: bool = False
    gate_size: Optional[float] = None
    domain_randomize: bool = True
    track_names: Optional[List[str]] = None

    # ── Reward shaping ──
    lambda_prog: Optional[float] = None
    lambda_gate: Optional[float] = None
    lambda_gate_inc: Optional[float] = None
    lambda_rate: Optional[float] = None

    # ── Curriculum ──
    curriculum_advance_pct: float = 85.0

    # ── Resume ──
    resume_checkpoint: Optional[str] = None

    # ── Evaluation ──
    eval_episodes: int = 20
    eval_tracks: List[str] = field(default_factory=lambda: ["kidney", "figure8"])

    # ── Success criteria (for automated comparison) ──
    success_metric: str = "avg_gates"  # What to optimize
    success_threshold: Optional[float] = None  # Beat this to "pass"

    # ── Cost tracking ──
    gpu_cost_per_hour: float = 0.0    # $/hr for the GPU (0 = local/free)
    gpu_type: str = ""                # e.g. "RTX_4090"
    estimated_hours: float = 0.0      # Pre-training estimate
    estimated_cost: float = 0.0       # Pre-training estimate
    actual_hours: float = 0.0         # Post-training actual
    actual_cost: float = 0.0          # Post-training actual
    cost_efficiency: float = 0.0      # avg_gates / $ (higher = better)

    # ── Auto-populated at runtime ──
    experiment_id: Optional[str] = None
    git_hash: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"           # pending | running | completed | failed

    def save(self, path: str):
        """Save config as JSON."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def generate_id(self) -> str:
        """Generate unique experiment ID from timestamp + name."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{ts}_{self.name}"
        return self.experiment_id

    def to_train_args(self) -> list:
        """Convert config to CLI arguments for the appropriate training script."""
        args = [
            "--num-envs", str(self.num_envs),
            "--timesteps", str(self.total_timesteps),
            "--device", "cuda",
            "--n-steps", str(self.n_steps),
            "--n-epochs", str(self.n_epochs),
            "--ent-coef", str(self.ent_coef),
        ]

        if not self.random_segments:
            args.append("--no-random-segments")
        if self.fixed_start:
            args.append("--fixed-start")
        if self.single_gate:
            args.append("--single-gate")
        if self.gate_size is not None:
            args.extend(["--gate-size", str(self.gate_size)])

        # Reward lambdas
        for attr, flag in [
            ("lambda_prog", "--lambda-prog"),
            ("lambda_gate", "--lambda-gate"),
            ("lambda_gate_inc", "--lambda-gate-inc"),
            ("lambda_rate", "--lambda-rate"),
        ]:
            val = getattr(self, attr)
            if val is not None:
                args.extend([flag, str(val)])

        # Model-specific
        if self.model_type in ("rppo", "mamba"):
            args.extend(["--seq-len", str(self.seq_len)])
            if self.model_type == "rppo":
                args.extend([
                    "--lstm-hidden", str(self.lstm_hidden),
                    "--lstm-layers", str(self.lstm_layers),
                    "--num-seq-per-batch", str(self.num_seq_per_batch),
                ])

        if self.model_type == "fsppo":
            args.extend(["--frame-stack", str(self.frame_stack)])
            args.extend(["--minibatch-size", str(self.batch_size)])

        # LR schedule
        if self.cosine_lr:
            args.append("--cosine-lr")
        else:
            args.append("--no-cosine-lr")
        args.extend(["--lr-cycle", str(self.lr_cycle)])
        args.extend(["--lr-min", str(self.lr_min)])

        # Curriculum
        args.extend(["--curriculum-advance-pct", str(self.curriculum_advance_pct)])

        # Resume
        if self.resume_checkpoint:
            args.extend(["--resume", self.resume_checkpoint])

        # Track names
        if self.track_names and self.model_type in ("fsppo",):
            args.extend(["--track-names"] + self.track_names)

        return args

    @property
    def train_script(self) -> str:
        """Return the Python module path for training."""
        return {
            "ppo": "control.train_gpu",
            "rppo": "control_rppo.train_gpu_rppo",
            "mamba": "control_mamba.train_gpu_mamba",
            "fsppo": "control_fsppo.train_gpu_fsppo",
        }[self.model_type]

    @property
    def save_dir(self) -> str:
        """Experiment-specific save directory."""
        return f"D:/drone2_training/autoresearch/{self.experiment_id}"
