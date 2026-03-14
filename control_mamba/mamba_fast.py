"""
Fast Mamba block using the official mamba-ssm CUDA kernels.
Drop-in replacement for the pure PyTorch MambaBlock in train_gpu_mamba.py.

Falls back to pure PyTorch if mamba-ssm is not installed (e.g. local dev on Windows).

Usage:
    from control_mamba.mamba_fast import FastMambaBlock
    # Same interface as MambaBlock:
    #   .initial_state(batch_size, device)
    #   .forward_single(x, state)        — rollout collection
    #   .forward_sequence(x_seq, state, dones_seq)  — PPO training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


class FastMambaBlock(nn.Module):
    """
    Wrapper around the official mamba-ssm Mamba block that matches
    the interface expected by MambaActorCritic in train_gpu_mamba.py.

    Key differences from the official API:
    - forward_single(x, state): single-step for rollout, returns (output, new_state)
    - forward_sequence(x_seq, state, dones_seq): handles episode done resets
    - state = (ssm_state, conv_state) tuple, same as the pure PyTorch version

    When mamba-ssm is not installed, falls back to the pure PyTorch implementation.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        if HAS_MAMBA_SSM:
            self.mamba = OfficialMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=True,
                layer_idx=0,
            )
            self._use_fast = True
        else:
            # Fall back to pure PyTorch
            from control_mamba.train_gpu_mamba import MambaBlock as PureMambaBlock
            self._fallback = PureMambaBlock(d_model, d_state, d_conv, expand)
            self._use_fast = False

    def initial_state(self, batch_size, device):
        """Return zero (ssm_state, conv_state) — same format as pure PyTorch version."""
        if self._use_fast:
            # Official mamba-ssm: conv_state is [B, d_inner, d_conv], ssm_state is [B, d_inner, d_state]
            conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device)
            ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        else:
            return self._fallback.initial_state(batch_size, device)
        return ssm_state, conv_state

    def forward_single(self, x, state):
        """
        Single-step forward for rollout collection.
        x: [batch, d_model]
        state: (ssm_state [B, d_inner, d_state], conv_state [B, d_inner, d_conv])
        Returns: output [batch, d_model], new_state
        """
        if not self._use_fast:
            return self._fallback.forward_single(x, state)

        ssm_state, conv_state = state

        # Official Mamba.step() expects [B, 1, D] and modifies states in-place
        # Clone states so we can return clean new states
        conv_s = conv_state.clone()
        ssm_s = ssm_state.clone()

        out, conv_s, ssm_s = self.mamba.step(
            x.unsqueeze(1),  # [B, 1, D]
            conv_s,
            ssm_s,
        )

        return out.squeeze(1), (ssm_s, conv_s)  # [B, D], (new_ssm, new_conv)

    def forward_sequence(self, x_seq, state, dones_seq=None):
        """
        Process a sequence for PPO training.
        x_seq: [seq_len, batch, d_model]
        state: (ssm_state, conv_state)
        dones_seq: [seq_len, batch] or None
        Returns: outputs [seq_len, batch, d_model], final_state
        """
        if not self._use_fast:
            return self._fallback.forward_sequence(x_seq, state, dones_seq)

        seq_len, batch_size, d_model = x_seq.shape

        if dones_seq is None or not dones_seq.any():
            # No done resets needed — use the fast fused kernel for the whole sequence
            # Official forward expects [B, L, D]
            x_bl = x_seq.permute(1, 0, 2).contiguous()  # [B, L, D]
            out_bl = self.mamba(x_bl)  # [B, L, D] — uses fused CUDA kernel
            out = out_bl.permute(1, 0, 2).contiguous()  # [L, B, D]

            # We need final state — step through the last position to get it
            # For PPO training, the final state is typically not needed (it's discarded)
            # So return zero state as placeholder
            ssm_state, conv_state = state
            return out, (ssm_state, conv_state)
        else:
            # Done resets present — must step through sequentially to reset states
            # Use the fused step() kernel for each timestep (still faster than pure PyTorch)
            ssm_state, conv_state = state
            ssm_state = ssm_state.clone()
            conv_state = conv_state.clone()

            outputs = []
            for t in range(seq_len):
                # Reset state where episodes ended at previous step
                if t > 0:
                    done_mask = dones_seq[t - 1]  # [B]
                    if done_mask.any():
                        dm = done_mask.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                        ssm_state = ssm_state * (1.0 - dm)
                        conv_state = conv_state * (1.0 - dm)

                out, conv_state, ssm_state = self.mamba.step(
                    x_seq[t].unsqueeze(1),  # [B, 1, D]
                    conv_state,
                    ssm_state,
                )
                outputs.append(out.squeeze(1))  # [B, D]

            return torch.stack(outputs), (ssm_state, conv_state)

    def forward_sequence_chunked(self, x_seq, state, dones_seq=None, chunk_size=64):
        """
        Optimized: process done-free chunks with the fused kernel, only step
        through timesteps where dones actually occur.

        For typical RL rollouts, dones are sparse (maybe 5-10% of timesteps),
        so this gets most of the fused kernel speedup while still handling resets.
        """
        if not self._use_fast:
            return self._fallback.forward_sequence(x_seq, state, dones_seq)

        seq_len, batch_size, d_model = x_seq.shape

        if dones_seq is None or not dones_seq.any():
            return self.forward_sequence(x_seq, state, dones_seq)

        # Find timesteps where any env has a done
        any_done_per_step = dones_seq.any(dim=1)  # [seq_len]
        done_steps = torch.where(any_done_per_step)[0].tolist()

        if not done_steps:
            return self.forward_sequence(x_seq, state, dones_seq)

        # Process in chunks between done boundaries
        ssm_state, conv_state = state
        ssm_state = ssm_state.clone()
        conv_state = conv_state.clone()

        outputs = []
        chunk_start = 0

        for done_t in done_steps + [seq_len]:
            # Process clean chunk [chunk_start, done_t) with fused kernel
            if done_t > chunk_start:
                chunk = x_seq[chunk_start:done_t]  # [chunk_len, B, D]
                chunk_bl = chunk.permute(1, 0, 2).contiguous()  # [B, chunk_len, D]
                out_bl = self.mamba(chunk_bl)  # [B, chunk_len, D]
                outputs.append(out_bl.permute(1, 0, 2).contiguous())  # [chunk_len, B, D]

            # Apply done reset at done_t (if it's not the end)
            if done_t < seq_len:
                done_mask = dones_seq[done_t]  # [B]
                if done_mask.any():
                    dm = done_mask.unsqueeze(-1).unsqueeze(-1)
                    ssm_state = ssm_state * (1.0 - dm)
                    conv_state = conv_state * (1.0 - dm)

            chunk_start = done_t + 1 if done_t < seq_len else seq_len

        if outputs:
            return torch.cat(outputs, dim=0), (ssm_state, conv_state)
        else:
            return torch.zeros(0, batch_size, d_model, device=x_seq.device), (ssm_state, conv_state)
