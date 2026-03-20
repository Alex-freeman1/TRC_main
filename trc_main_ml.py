# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:39:18 2026

@author: alexa
"""

"""
Tiny Recursive Control (TRC)
============================
Jain & Linares, AIAA SciTech 2026

A small neural network (~1.5M params) that generates near-optimal control
sequences by iteratively refining them. The same weights are reused at every
iteration, so more iterations = better solutions without more memory.

Architecture:
    1. State Encoder    — encodes the problem (x0, goal, time) into latent z0
    2. Error Encoder    — encodes tracking error into latent z_err
    3. Reasoning Module — shared transformer blocks (the core of TRC)
    4. Initial Decoder  — first guess at controls from z0
    5. Residual Decoder — corrections to controls at each iteration

The reasoning module is inspired by TRM (Jolicoeur-Martineau 2024).
It receives multiple latent vectors as separate tokens and uses self-attention
to dynamically weight between them. The same module handles both:
    - Tactical updates: z_L = L_θ(z_L, z_H, z0, z_err, z_ctrl)  [5 tokens]
    - Strategic updates: z_H = L_θ(z_H, z_L)                     [2 tokens]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
import numpy as np


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    """Problem definition."""
    state_dim:   int          # dimension of state x
    control_dim: int          # dimension of control u
    horizon:     int          # number of time steps T
    dt:          float        # integration time step
    u_min:       float = -1.0 # control lower bound (scalar box or norm)
    u_max:       float =  1.0 # control upper bound (scalar box or norm)
    norm_bounded: bool = False # if True, bounds apply to ‖u‖ not each component

    def project_control(self, u: np.ndarray) -> np.ndarray:
        """Clip u to feasible set — box constraint or norm annulus."""
        if self.norm_bounded:
            # Moon lander: T_MIN ≤ ‖u‖ ≤ T_MAX
            T_norm = np.linalg.norm(u)
            if T_norm < self.u_min:
                return u * (self.u_min / T_norm)
            elif T_norm > self.u_max:
                return u * (self.u_max / T_norm)
            return u
        else:
            # VdP: simple elementwise box clip
            return np.clip(u, self.u_min, self.u_max)
        
        
@dataclass
class NetConfig:
    """Architecture settings."""
    d_z: int = 256          # latent dimension
    d_h: int = 512          # hidden dimension in MLPs
    n_heads: int = 8        # attention heads
    n_blocks: int = 3       # transformer blocks per reasoning call (L)
    K: int = 3              # outer refinement iterations
    n_inner: int = 4        # inner tactical cycles per iteration (n)
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    lambda_ps: float = 0.3  # process supervision weight
    grad_clip: float = 1.0


# ── Building Blocks ─────────────────────────────────────────────────────────

def rms_norm(x, eps=1e-6):
    """Root mean square normalization (used in TRM instead of LayerNorm)."""
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def make_mlp(in_dim, hidden_dim, out_dim):
    """Standard 2-layer MLP with LayerNorm + GELU (used for all encoders/decoders)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class ReasoningBlock(nn.Module):
    """One transformer block: self-attention + SwiGLU FFN, both with residual + RMSNorm.

    Operates on a short sequence of tokens (2-5 tokens, each of dimension d_z).
    Self-attention lets the model dynamically weight between different information
    sources (state, error, control history, etc).
    """

    def __init__(self, d_z, d_h, n_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_z, n_heads, dropout=dropout, batch_first=True)
        # SwiGLU FFN: gate and up are parallel projections, output is gate(x) * up(x)
        self.w_gate = nn.Linear(d_z, d_h, bias=False)
        self.w_up   = nn.Linear(d_z, d_h, bias=False)
        self.w_down = nn.Linear(d_h, d_z, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq):
        # Self-attention + residual + RMSNorm
        attn_out, _ = self.attn(seq, seq, seq)
        seq = rms_norm(seq + self.drop(attn_out))
        # SwiGLU FFN + residual + RMSNorm
        ffn_out = self.w_down(F.silu(self.w_gate(seq)) * self.w_up(seq))
        seq = rms_norm(seq + self.drop(ffn_out))
        return seq


class ReasoningModule(nn.Module):
    """The shared reasoning module L_θ — the heart of TRC.

    Takes any number of d_z vectors as separate tokens, stacks them into a
    sequence, runs through L transformer blocks, and returns the first token
    (the updated state).

    Called with 5 tokens for tactical updates, 2 tokens for strategic updates.
    Same weights handle both cases — the attention pattern adapts automatically.
    """

    def __init__(self, d_z, d_h, n_heads, n_blocks, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ReasoningBlock(d_z, d_h, n_heads, dropout) for _ in range(n_blocks)
        ])

    def forward(self, *tokens):
        seq = torch.stack(tokens, dim=1)   # (batch, num_tokens, d_z)
        for block in self.blocks:
            seq = block(seq)
        return seq[:, 0, :]                # updated first token


# ── Dynamics Simulator ──────────────────────────────────────────────────────

class Simulator(nn.Module):
    """Rolls out dynamics x_{t+1} = RK4(f, x_t, u_t) for T steps."""

    def __init__(self, dynamics_fn, dt, horizon):
        super().__init__()
        self.f = dynamics_fn
        self.dt = dt
        self.T = horizon

    def forward(self, x0, u_seq):
        """Returns full state trajectory: (batch, T+1, d_x)."""
        states = [x0]
        x = x0
        for t in range(self.T):
            u_t = u_seq[:, t, :]
            dt = self.dt
            k1 = self.f(x, u_t)
            k2 = self.f(x + 0.5 * dt * k1, u_t)
            k3 = self.f(x + 0.5 * dt * k2, u_t)
            k4 = self.f(x + dt * k3, u_t)
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            states.append(x)
        return torch.stack(states, dim=1)


# ── TRC Model ───────────────────────────────────────────────────────────────

class TRC(nn.Module):
    """Tiny Recursive Control.

    Given an initial state x0 and a goal, produces a control sequence u
    by iteratively refining an initial guess. Each iteration:
        1. Simulate dynamics with current controls
        2. Compute tracking error
        3. Update tactical latent z_L (n inner cycles)
        4. Update strategic latent z_H (1 step)
        5. Decode a correction Δu and apply it
    """

    def __init__(self, task: TaskConfig, net: NetConfig, dynamics_fn: Callable):
        super().__init__()
        self.task = task
        self.net = net

        d_x, d_u, T = task.state_dim, task.control_dim, task.horizon
        d_z, d_h = net.d_z, net.d_h

        # Module 1: State Encoder — Eq. 5
        self.state_encoder = make_mlp(2 * d_x + 1, d_h, d_z)

        # Module 2: Error Encoder — Eq. 6
        self.error_encoder = make_mlp(d_x, d_h, d_z)

        # Control embedding (flatten u and project)
        self.ctrl_embed = nn.Linear(T * d_u, d_z)

        # Module 3: Reasoning Module — Eq. 8-9 (shared across ALL iterations)
        self.reason = ReasoningModule(d_z, d_h, net.n_heads, net.n_blocks, net.dropout)

        # Module 4: Initial Decoder — first control guess from z0
        self.init_decoder = make_mlp(d_z, d_h, T * d_u)

        # Module 5: Residual Decoder — Eq. 10: corrections from [z_H; u_prev]
        self.res_decoder = make_mlp(d_z + T * d_u, d_h, T * d_u)

        # Learnable latent initializations + sample-specific projections
        self.H_init = nn.Parameter(torch.randn(d_z) * 0.02)
        self.L_init = nn.Parameter(torch.randn(d_z) * 0.02)
        self.h_proj = nn.Linear(d_z, d_z)
        self.l_proj = nn.Linear(d_z, d_z)

        # Differentiable dynamics rollout
        self.sim = Simulator(dynamics_fn, task.dt, T)

    def set_cost_matrices(self, Q, R, Qf):
        """Register quadratic cost matrices as buffers."""
        self.register_buffer('_Q', Q)
        self.register_buffer('_R', R)
        self.register_buffer('_Qf', Qf)

    def _clip(self, u):
        return torch.clamp(u, self.task.u_min, self.task.u_max)

    def _reshape_u(self, flat):
        return flat.view(-1, self.task.horizon, self.task.control_dim)

    def _cost_from_states(self, states, goal, u):
        """Quadratic tracking cost from precomputed rollout states."""
        if not hasattr(self, '_Q'):
            raise RuntimeError("Call set_cost_matrices(Q, R, Qf) before forward pass.")
        x_traj = states[:, :-1, :]
        x_T = states[:, -1, :]
        err_T = x_T - goal

        state_cost = torch.einsum('bti,ij,btj->b', x_traj, self._Q, x_traj)
        ctrl_cost  = torch.einsum('bti,ij,btj->b', u, self._R, u)
        term_cost  = torch.einsum('bi,ij,bj->b', err_T, self._Qf, err_T)
        return state_cost + ctrl_cost + term_cost

    def _cost(self, x0, goal, u):
        """Quadratic tracking cost: J = Σ(xᵀQx + uᵀRu) + (x_T-goal)ᵀQf(x_T-goal)."""
        states = self.sim(x0, u)
        return self._cost_from_states(states, goal, u)

    def forward(self, x0, goal, t_remaining, return_history=True):
        """Run TRC refinement loop (Algorithm 1).

        Args:
            x0:          (B, d_x)  initial state
            goal:        (B, d_x)  target state
            t_remaining: (B, 1)    time horizon

        Returns:
            dict with u_final and costs; plus histories if return_history=True
        """
        B = x0.shape[0]
        K, n = self.net.K, self.net.n_inner

        # Step 1: Encode the problem
        z0 = self.state_encoder(torch.cat([x0, goal, t_remaining], dim=-1))

        # Step 2-3: Initialize latents (learnable + sample-specific)
        z_H = self.H_init.expand(B, -1) + self.h_proj(z0)
        z_L = self.L_init.expand(B, -1) + self.l_proj(z0)

        # Step 4: Initial control guess
        u = self._clip(self._reshape_u(self.init_decoder(z0)))

        # Track everything (optionally keep full histories for analysis/plots)
        states = self.sim(x0, u)
        costs = [self._cost_from_states(states, goal, u)]
        if return_history:
            u_iters = [u]
            z_H_hist = [z_H.detach()]
            errors = []

        # Steps 5-15: Iterative refinement
        for k in range(K):
            # Simulate and compute error
            error = states[:, -1, :] - goal    # terminal tracking error
            if return_history:
                errors.append(error)

            # Encode feedback signals
            z_err  = self.error_encoder(error)
            z_ctrl = self.ctrl_embed(u.reshape(B, -1))

            # Tactical cycles: z_L attends to [z_L, z_H, z0, z_err, z_ctrl]
            for _ in range(n):
                z_L = self.reason(z_L, z_H, z0, z_err, z_ctrl)

            # Strategic update: z_H attends to [z_H, z_L]
            z_H = self.reason(z_H, z_L)

            # Decode correction and apply
            delta_u = self._reshape_u(
                self.res_decoder(torch.cat([z_H, u.reshape(B, -1)], dim=-1))
            )
            u = self._clip(u + delta_u)

            states = self.sim(x0, u)
            costs.append(self._cost_from_states(states, goal, u))
            if return_history:
                u_iters.append(u)
                z_H_hist.append(z_H.detach())

        out = {
            'u_final': u,
            'costs': costs,
            'terminal_error': states[:, -1, :] - goal,
        }
        if return_history:
            out.update({
                'u_iterations': u_iters,
                'errors': errors,
                'z_H_history': z_H_hist,
            })
        return out


# ── Loss ────────────────────────────────────────────────────────────────────

class TRCLoss(nn.Module):
    """Process supervision loss (Eq. 15).

    Three terms:
        1. Control accuracy:    ||u^(K) - u*||²
        2. Goal enforcement:    ||x_T - goal||²
        3. Improvement reward:  -λ * mean cost reduction per iteration

    The improvement reward teaches each iteration to actually reduce cost,
    not just produce the right final answer through arbitrary intermediate steps.
    """

    def __init__(self, lambda_ps=0.3, lambda_goal=0.0):
        super().__init__()
        self.lam = lambda_ps
        self.lam_goal = lambda_goal

    def forward(self, output, u_star):
        costs = output['costs']
        K = len(costs) - 1

        # Term 1: match optimal controls
        ctrl_loss = F.mse_loss(output['u_final'], u_star)

        # Term 2: explicitly enforce terminal goal x_T ≈ goal
        if 'terminal_error' in output:
            goal_loss = output['terminal_error'].pow(2).mean()
        else:
            goal_loss = torch.tensor(0.0, device=u_star.device)

        # Term 2: reward cost reduction at each step
        if K > 1:
            J0 = costs[0].detach().clamp(min=1e-8)
            normed = [c / J0 for c in costs]
            improvements = [normed[k-1] - normed[k] for k in range(1, K)]
            imp_loss = -self.lam * torch.stack(improvements).mean()
        else:
            imp_loss = torch.tensor(0.0, device=u_star.device)

        loss = ctrl_loss + self.lam_goal * goal_loss + imp_loss

        # Logging metrics
        with torch.no_grad():
            imp_metric = 0.0
            if K > 0:
                J0d = costs[0].clamp(min=1e-8)
                for k in range(K):
                    imp_metric += ((costs[k] - costs[k+1]) / J0d).mean().item()
                imp_metric /= K
            terminal_err = 0.0
            if 'terminal_error' in output:
                terminal_err = torch.norm(output['terminal_error'], dim=-1).mean().item()

        return loss, {
            'loss': loss.item(),
            'final_loss': ctrl_loss.item(),
            'ctrl_loss': ctrl_loss.item(),
            'goal_loss': goal_loss.item(),
            'imp_loss': imp_loss.item(),
            'imp_metric': imp_metric,
            'terminal_err': terminal_err,
            'cost_0': costs[0].mean().item(),
            'cost_K': costs[-1].mean().item(),
        }


# ── Helpers ─────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def vdp_dynamics(x, u, mu=1.0):
    """Van der Pol oscillator: ẍ - μ(1-x²)ẋ + x = u"""
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return torch.cat([x2, mu * (1 - x1**2) * x2 - x1 + u], dim=-1)

# def spacecraft_dynamics(x, u, mu=1.0):
  
#     x1, x2 = x[:, 0:1], x[:, 1:2]
#     return torch.cat([x2, mu * (1 - x1**2) * x2 - x1 + u], dim=-1)

# def j2_rhs(state, mu=MU_EARTH, r_earth=R_EARTH, j2=J2):
#     """Two-body + J2 equations of motion in ECI."""
#     r = state[:3]; v = state[3:6]
#     x, y, z = r
#     r2 = float(np.dot(r, r)); rmag = np.sqrt(r2)
#     a_grav = -mu * r / (rmag**3)
#     z2 = z*z; r5 = rmag**5
#     factor = 1.5 * j2 * mu * (r_earth**2) / r5
#     common = 5.0 * z2 / r2
#     a_j2 = factor * np.array([x*(common-1.0), y*(common-1.0), z*(common-3.0)])
#     return np.concatenate([v, a_grav + a_j2])




# ── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    task = TaskConfig(state_dim=2, control_dim=1, horizon=100, dt=0.05, u_min=-2, u_max=2)
    net = NetConfig(d_z=256, d_h=512, n_heads=8, n_blocks=3, K=3, n_inner=4)

    model = TRC(task, net, dynamics_fn=vdp_dynamics)
    model.set_cost_matrices(
        Q=torch.diag(torch.tensor([10., 5.])),
        R=torch.tensor([[0.5]]),
        Qf=torch.diag(torch.tensor([200., 100.])),
    )

    print(f'Parameters: {count_params(model):,}')

    B = 4
    out = model(
        x0=torch.randn(B, 2),
        goal=torch.zeros(B, 2),
        t_remaining=torch.ones(B, 1) * 5.0,
    )

    loss, metrics = TRCLoss()(out, torch.zeros(B, 100, 1))
    loss.backward()
    print(f'Loss: {metrics["loss"]:.4f}, cost: {metrics["cost_0"]:.0f} → {metrics["cost_K"]:.0f}')