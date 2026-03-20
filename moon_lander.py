# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:05:42 2026

@author: alexa
"""


import os, time
import contextlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


full_path_to_folder = r"C:\Users\alexa\Git Projects\Tiny Recursive Control"
sys.path.append(full_path_to_folder)

from trc_main_ml import TRC, TRCLoss, TaskConfig, NetConfig, vdp_dynamics, count_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = (device.type == 'cuda')
if use_cuda:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
print(f'Device: {device}')


QUICK = True

# Data
N_TRAIN = 64 if QUICK else 10_000
N_TEST  = 16 if QUICK else 1_000
DATA_DIR = Path('./data')
DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path('./checkpoints')
CKPT_DIR.mkdir(exist_ok=True)
FORCE_REGEN = True
PARALLEL_GEN = False
GEN_WORKERS = max(1, (os.cpu_count() or 1) - 1)
if QUICK:
    GEN_WORKERS = min(GEN_WORKERS, 4)

# Training
EPOCHS     = 5 if QUICK else 50
BATCH_SIZE = 16 if QUICK else 64
LR         = 1e-3
LAMBDA_PS  = 0.3


# Problem constants — Moon Landing (7 states, 3 controls)
G     = 1.62      # m/s², lunar gravity
DT    = 2.0       # seconds per timestep
T     = 150       # total timesteps (~5 min descent)
ISP = 200.7

# Control bounds — 3 thrusters [Tx, Ty, Tz]
U_MIN = np.array([0.0,  0.0,  0.0 ])   # thrust is non-negative in all axes
U_MAX = np.array([2.0,  2.0,  3.0 ])   # more vertical (z) thrust available

# State: x = [x, y, z, vx, vy, vz, mass]
#                x     y     z     vx    vy    vz    m
Q   = np.diag([1.0,  1.0,  5.0,  1.0,  1.0,  10.0, 0.1])   # running cost
Q_F = np.diag([10.0, 10.0, 50.0, 10.0, 10.0, 1000.0, 0.0]) # terminal cost

# Control cost — 3x3, lateral thrusters slightly more expensive (fuel efficiency)
R   = np.diag([3.0,  3.0,  1.0])   # [Tx, Ty, Tz]

# Initial conditions
x0 = np.array([100.0,   
               100.0,   
               1000.0,  
               5.0,     
               5.0,     
               -20.0,   
               200.0])

CKPT_PATH = CKPT_DIR / 'trc_vdp_best.pt'

print(f'N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}')
print(f'Parallel data gen: {PARALLEL_GEN} (workers={GEN_WORKERS})')


import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor


def mooon_landing(x,u):
    
    return np.array([
    x[3],                            # dx/dt   = vx
    x[4],                            # dy/dt   = vy
    x[5],                            # dz/dt   = vz
    u[0] / x[6],                     # dvx/dt  = Tx / mass
    u[1] / x[6],                     # dvy/dt  = Ty / mass
    u[2] / x[6] - G,                 # dvz/dt  = Tz / mass - gravity
    -np.linalg.norm(u) / (ISP * G)   # dm/dt   = fuel burn rate
])



# ── Physical constants (from paper) ──────────────────────────────────────────
G_MARS = 3.71          # m/s², Martian gravity
ISP    = 200.7         # s,    specific impulse
G0     = 9.81          # m/s², Earth gravity (for Tsiolkovsky denominator)

# ── Thrust bounds (from paper Eq. 21) ────────────────────────────────────────
T_MIN  = 4000.0        # N, minimum thrust (engine can't throttle below this)
T_MAX  = 13000.0       # N, maximum thrust

# ── Mass bounds (from paper Eq. 24) ──────────────────────────────────────────
M_DRY  = 1000.0        # kg, dry mass (no fuel)
M_WET  = 2000.0        # kg, wet mass (full fuel load) — also initial mass

# ── Glideslope constraint (from paper Eq. 22) ─────────────────────────────────
GAMMA_GS = 75.0                          # degrees from vertical
TAN_GS   = np.tan(np.radians(GAMMA_GS)) # precomputed for use in constraint

# ── Terminal constraints (from paper Eq. 23) ──────────────────────────────────
V_TOL  = 1.0           # m/s, maximum allowable landing velocity

# ── Time discretisation ───────────────────────────────────────────────────────
DT     = 1.0           # seconds per timestep
T      = 150           # total timesteps

# ── Gravity vector (from paper Eq. 20) ───────────────────────────────────────
g = np.array([0.0, 0.0, -G_MARS])

# ── Cost weights — fuel-optimal Mayer form J = m0 - m(tf) ────────────────────
# The paper minimises fuel, so we penalise state errors lightly and
# weight terminal velocity heavily to enforce the soft landing constraint.
# State: x = [x, y, z, vx, vy, vz, mass]
Q   = np.diag([0.0,  0.0,  0.0,                # no running position cost
               0.0,  0.0,  0.0,                # no running velocity cost
               0.0])                           # mass handled by Mayer term

Q_F = np.diag([100.0, 100.0, 100.0,            # terminal position → 0
               500.0, 500.0, 500.0,            # terminal velocity ≤ v_tol
               0.0  ])                         # mass free at terminal (Mayer)

R   = np.diag([1e-6, 1e-6, 1e-6])             # tiny — thrust cost via mass loss
                                               # not quadratic R in this formulation

# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.array([400.0,   # x (m), horizontal offset
               400.0,   # y (m), horizontal offset
               1500.0,  # z (m), altitude
               -10.0,   # vx (m/s)
               -10.0,   # vy (m/s)
               -50.0,   # vz (m/s), descending
               M_WET])  # mass (kg), start fully fuelled




def project_thrust(u: np.ndarray) -> np.ndarray:
    T_norm = np.linalg.norm(u)
    
    if T_norm < 1e-8:
        # Near-zero thrust — default to minimum thrust pointing upward
        return np.array([0.0, 0.0, T_MIN])
    elif T_norm < T_MIN:
        return u * (T_MIN / T_norm)
    elif T_norm > T_MAX:
        return u * (T_MAX / T_norm)
    return u


'''
Runge-kutta gen 4 method for integration
'''

def rk4_np(x, u):
    k1 = mooon_landing(x, u)
    k2 = mooon_landing(x + 0.5 * DT * k1, u)
    k3 = mooon_landing(x + 0.5 * DT * k2, u)
    k4 = mooon_landing(x + DT * k3, u)
    return x + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


'''
Extrapolate time sequence using rk4 integrator
'''
def rollout_np(x0, u_seq):
    u_seq = u_seq.reshape(T, 3)                      # was implicit (T,) for VdP
    states = np.zeros((T + 1, 7), dtype=np.float64)  # was (T + 1, 2)
    states[0] = x0
    for t in range(T):
        states[t + 1] = rk4_np(states[t], u_seq[t])
    return states


def cost_np(u_seq, x0):
    u_seq  = u_seq.reshape(T, 3)                     # was implicit (T,) for VdP
    states = rollout_np(x0, u_seq)

    # Running cost — Mayer formulation from paper means Q is zeros,
    # so this contributes nothing but is kept for structural consistency
    J = sum(
        states[t] @ Q @ states[t] +
        u_seq[t] @ R @ u_seq[t]                      # was R * u_seq[t]**2 (scalar)
        for t in range(T)
    )

    # Terminal cost
    J += states[T] @ Q_F @ states[T]

    return float(J)


'''
Find the gradient of J, \nabla J. 
Uses the costate (adjoint) method and sweeps back in time. 
'''
def cost_grad_np(u_seq, x0):
    u_seq  = u_seq.reshape(T, 3)                         # was implicit (T,)
    states = rollout_np(x0, u_seq)
    
    costate = 2.0 * Q_F @ states[T]                      # stays the same, Q_F now 7x7
    grad    = np.zeros((T, 3))                            # was np.zeros(T)
    eps     = 1e-6

    for t in range(T - 1, -1, -1):
        # Control gradient — u_seq[t] is now (3,), R is now (3x3)
        grad[t] = 2.0 * R @ u_seq[t]                     # was 2.0 * R * u_seq[t]

        # Gradient through control: perturb each thrust component separately
        du_grad = np.zeros(3)                             # was scalar
        for j in range(3):                                # was single scalar perturbation
            u_p, u_m = u_seq[t].copy(), u_seq[t].copy()
            u_p[j] += eps
            u_m[j] -= eps
            du_grad[j] = costate @ (rk4_np(states[t], u_p) - rk4_np(states[t], u_m)) / (2 * eps)
        grad[t] += du_grad

        # State Jacobian — now (7x7) instead of (2x2)
        dfdx = np.zeros((7, 7))                           # was (2, 2)
        for i in range(7):                                # was range(2)
            xp, xm = states[t].copy(), states[t].copy()
            xp[i] += eps
            xm[i] -= eps
            dfdx[:, i] = (rk4_np(xp, u_seq[t]) - rk4_np(xm, u_seq[t])) / (2 * eps)

        costate = 2.0 * Q @ states[t] + dfdx.T @ costate

    return grad.ravel()                                   # flatten to (T*3,) for scipy



'''
Solves one singular optimal control input u*
'''
def solve_one(x0):
    # Build warm start — minimum thrust pointing upward along z axis
    u_init = np.zeros(T * 3)
    u_init[2::3] = T_MIN    # every Tz component (indices 2, 5, 8, ...) = T_MIN
    
    return minimize(
        cost_np,
        u_init,                                  # pass it here as positional arg
        args=(x0,),
        jac=cost_grad_np,
        method='SLSQP',
        bounds=[(-T_MAX, T_MAX)] * (T * 3),
        options={'maxiter': 300, 'ftol': 1e-8},
    )

'''
This function solves the problem and collects everything required
'''


def _solve_one_payload(x0):
    print("1. solving...")
    result = solve_one(x0)
    
    print("2. projecting thrust...")
    u_raw = result.x.reshape(T, 3)
    u_opt = np.array([project_thrust(u) for u in u_raw])
    
    print("3. rolling out...")
    traj = rollout_np(x0, u_opt)
    
    print("4. computing cost...")
    cost_val = cost_np(u_opt, x0)
    print(f"   cost_val type: {type(cost_val)}, shape: {np.asarray(cost_val).shape}")
    
    print("5. packing...")
    return (
        x0.astype(np.float32),
        u_opt.reshape(T, 3).astype(np.float32),
        traj.astype(np.float32),
        float(np.asarray(cost_val).ravel()[0]),
        bool(result.success)
    )

def _pack_dataset(x0s, u_opts, trajs, costs_arr, success):
    n = len(x0s)
    return {
        'x0':           np.asarray(x0s,       dtype=np.float32),
        
        # Fix 2: target is 7D zero state — landed, stationary, any remaining mass
        'x_target':     np.zeros((n, 7),       dtype=np.float32),  # was (n, 2)
        
        't_remaining':  np.full((n, 1), T * DT, dtype=np.float32),
        'u_optimal':    np.asarray(u_opts,     dtype=np.float32),
        'x_trajectory': np.asarray(trajs,      dtype=np.float32),
        'costs':        np.asarray(costs_arr,  dtype=np.float32),
        'success':      np.asarray(success),
    }



def _generate_dataset_serial(x0_samples, verbose_every):
    x0s, u_opts, trajs, costs_arr, success = [], [], [], [], []
    n = len(x0_samples)
    for i, x0 in enumerate(x0_samples):
        x0_i, u_i, traj_i, cost_i, ok_i = _solve_one_payload(x0)
        if i == 0 or (i + 1) % verbose_every == 0 or i == n - 1:
            print(
                f'  [{i+1}/{n}] '
                f'z={x0_i[2]:+.1f}m '        # altitude
                f'vz={x0_i[5]:+.1f}m/s '     # vertical velocity
                f'mass={x0_i[6]:+.1f}kg '    # mass
                f'cost={cost_i:.1f} '
                f'{"OK" if ok_i else "FAIL"}'
            )
        x0s.append(x0_i)
        u_opts.append(u_i)
        trajs.append(traj_i)
        costs_arr.append(cost_i)
        success.append(ok_i)
    return _pack_dataset(x0s, u_opts, trajs, costs_arr, success)


class VDPDataset(Dataset):
    def __init__(self, path):
        d = np.load(path)
        self.x0        = torch.from_numpy(d['x0']).float()
        self.goal      = torch.from_numpy(d['x_target']).float()
        self.t_rem     = torch.from_numpy(d['t_remaining']).float()
        self.u_optimal = torch.from_numpy(d['u_optimal']).float()
        self.costs     = torch.from_numpy(d['costs']).float()

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, i):
        return {'x0': self.x0[i], 'goal': self.goal[i], 't_rem': self.t_rem[i],
                'u_opt': self.u_optimal[i], 'cost': self.costs[i]}
    
    





def sample_initial_conditions(n, seed=42):
    """
    Replace rng.uniform(-2, 2, size=(n, 2)) for VdP.
    Samples physically plausible 7D moon lander initial conditions.
    
    State: [x, y, z, vx, vy, vz, mass]
    """
    rng = np.random.RandomState(seed)

    # ── Position ──────────────────────────────────────────────────────────
    # Altitude: lander starts somewhere between 500m and 2000m up
    z  = rng.uniform(500.0,  2000.0, size=n)

    # Horizontal position: must satisfy glideslope ||r_xy|| ≤ z * tan(75°)
    # So max horizontal offset grows with altitude
    xy_max      = z * TAN_GS                        # per-sample horizontal limit
    x_pos       = rng.uniform(-1, 1, size=n) * xy_max * 0.5   # stay well inside
    y_pos       = rng.uniform(-1, 1, size=n) * xy_max * 0.5   # glideslope cone

    # ── Velocity ──────────────────────────────────────────────────────────
    # Vertical: always descending, between -80 and -10 m/s
    vz = rng.uniform(-80.0, -10.0, size=n)

    # Horizontal: small lateral drift, realistic for a descent scenario
    vx = rng.uniform(-20.0,  20.0, size=n)
    vy = rng.uniform(-20.0,  20.0, size=n)

    # ── Mass ──────────────────────────────────────────────────────────────
    # Start somewhere between 60% and 100% full of fuel
    mass = rng.uniform(0.6 * M_WET, M_WET, size=n)

    # ── Stack into (n, 7) array, same shape convention as VdP's (n, 2) ───
    x0_samples = np.column_stack([
        x_pos, y_pos, z,
        vx, vy, vz,
        mass
    ]).astype(np.float64)

    return x0_samples

def main():
    
    x0_test = sample_initial_conditions(1, seed=42)[0]
    print("x0_test shape:", x0_test.shape)
    
    try:
        result = _solve_one_payload(x0_test)
        print("x0:     ", result[0].shape)
        print("u_opt:  ", result[1].shape)
        print("traj:   ", result[2].shape)
        print("cost:   ", result[3])
        print("success:", result[4])
    except Exception as e:
        print(f"_solve_one_payload failed: {e}")
        import traceback
        traceback.print_exc()
        
    def generate_dataset(n, seed=42, verbose_every=100, parallel=False, workers=None):

        
        x0_samples = sample_initial_conditions(n, seed=seed)
        
        verbose_every = max(1, int(verbose_every))
        # if (not parallel) or n <= 1:
        #     return _generate_dataset_serial(x0_samples, verbose_every)
    


        if workers is None:
            workers = max(1, (os.cpu_count() or 1) - 1)
        workers = max(1, min(int(workers), n))
        
        if workers == 1:
            return _generate_dataset_serial(x0_samples, verbose_every)
        
        # Prefer fork in notebooks on Unix; fallback to default context.
        mp_ctx = None
        if os.name != 'nt':
            try:
                mp_ctx = mp.get_context('fork')
            except ValueError:
                mp_ctx = mp.get_context()
        else:
            mp_ctx = mp.get_context('spawn')
        
        x0s, u_opts, trajs, costs_arr, success = [], [], [], [], []
        #print(f'Generating dataset in parallel with {workers} workers...')
        try:
            chunksize = max(1, n // (workers * 8))
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
                for i, payload in enumerate(ex.map(_solve_one_payload, x0_samples, chunksize=chunksize)):
                    x0_i, u_i, traj_i, cost_i, ok_i = payload
                    if i == 0 or (i + 1) % verbose_every == 0 or i == n - 1:
                        print(f'  [{i+1}/{n}] x0=[{x0_i[0]:+.2f},{x0_i[1]:+.2f}] cost={cost_i:.1f} {"OK" if ok_i else "FAIL"}')
                    x0s.append(x0_i)
                    u_opts.append(u_i)
                    trajs.append(traj_i)
                    costs_arr.append(cost_i)
                    success.append(ok_i)
            return _pack_dataset(x0s, u_opts, trajs, costs_arr, success)
        except Exception as exc:
            print(f'Parallel generation failed ({exc}); falling back to serial mode.')
            return _generate_dataset_serial(x0_samples, verbose_every)
            
        
    train_path = DATA_DIR / 'moon_lander_train.npz'
    test_path  = DATA_DIR / 'moon_lander_test.npz'
    
    if FORCE_REGEN or not train_path.exists():
        print('Generating training data...')
        np.savez(
            train_path,
            **generate_dataset(
                N_TRAIN, seed=42,
                verbose_every=max(1, N_TRAIN // 8),
                parallel=PARALLEL_GEN, workers=GEN_WORKERS,
            ),
        )
    
    if FORCE_REGEN or not test_path.exists():
        print('Generating test data...')
        np.savez(
            test_path,
            **generate_dataset(
                N_TEST, seed=123,
                verbose_every=max(1, N_TEST // 4),
                parallel=PARALLEL_GEN, workers=GEN_WORKERS,
            ),
        )
            
    d = np.load(train_path)
    for k in d.files:
        print(f'  {k:15s} {str(d[k].shape):15s} {d[k].dtype}')
        
    # Visualize a few expert trajectories
    d = np.load(train_path)
    idxs = np.linspace(0, len(d['x_trajectory'])-1, min(6, len(d['x_trajectory'])), dtype=int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for i in idxs:
        ax1.plot(d['x_trajectory'][i, :, 0], d['x_trajectory'][i, :, 1], alpha=0.7)
        ax2.plot(d['u_optimal'][i, :, 0], alpha=0.7)
    ax1.scatter(0, 0, marker='x', s=80, c='red', zorder=5)
    ax1.set(xlabel='x', ylabel='ẋ', title='Expert phase portraits'); ax1.grid(alpha=0.3)
    ax2.set(xlabel='Time step', ylabel='u', title='Optimal control sequences'); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


    train_ds = VDPDataset(train_path)
    test_ds  = VDPDataset(test_path)
    
                
    loader_workers = min(8, os.cpu_count() or 1)
    if QUICK:
        loader_workers = min(loader_workers, 2)
    
    loader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': loader_workers,
        'pin_memory': False,
        'persistent_workers': loader_workers > 0,
        
    }
    if loader_workers > 0:
        loader_kwargs['prefetch_factor'] = None
                
        
    print(f'Train: {len(train_ds)}, Test: {len(test_ds)}')
    print(f'DataLoader workers={loader_workers}, pin_memory={use_cuda}')
    
    
    task = TaskConfig(
    state_dim    = 7,
    control_dim  = 3,
    horizon      = T,
    dt           = DT,
    u_min        = T_MIN,    # 4000.0 N
    u_max        = T_MAX,    # 13000.0 N
    norm_bounded = True      # signals annulus constraint, not box
    )

    '''
    
    Here to change how many iterations K
    '''
    if QUICK:
        net = NetConfig(d_z=64, d_h=128, n_heads=4, n_blocks=1, K=2, n_inner=2)
    else:
        net = NetConfig(d_z=256, d_h=512, n_heads=8, n_blocks=3, K=3, n_inner=4)
    
    model = TRC(task, net, dynamics_fn=vdp_dynamics).to(device)
    if use_cuda:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('torch.compile: enabled (reduce-overhead)')
        except Exception as exc:
            print(f'torch.compile unavailable, continuing without it: {exc}')
    
    model.set_cost_matrices(
        Q  = torch.diag(torch.tensor([10., 5.], device=device)),
        R  = torch.tensor([[0.5]], device=device),
        Qf = torch.diag(torch.tensor([200., 100.], device=device)),
    )
    
    criterion = TRCLoss(lambda_ps=LAMBDA_PS)
    warmup_epochs = 5
    # Safety: warmup can never meet or exceed total epochs
    warmup_epochs = min(warmup_epochs, max(1, EPOCHS - 1))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - warmup_epochs), eta_min=1e-5)
    
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    
    def autocast_ctx():
        if use_cuda:
            return torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
        return contextlib.nullcontext()
    
    print(f'Parameters: {count_params(model):,}')
    print(f'K={net.K} outer iters, n={net.n_inner} inner cycles, L={net.n_blocks} blocks')
    if use_cuda:
        print(f'AMP dtype: {amp_dtype}')
        
    def to_dev(batch):
        if use_cuda:
            return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return {k: v.to(device) for k, v in batch.items()}
    
    
    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        ctrl_losses, imps = [], []
        for b in loader:
            b = to_dev(b)
            with autocast_ctx():
                out = model(b['x0'], b['goal'], b['t_rem'], return_history=False)
                _, m = criterion(out, b['u_opt'])
            ctrl_losses.append(m['final_loss'])
            imps.append(m['imp_metric'])
        return np.mean(ctrl_losses), np.mean(imps)
    
    
    def save_checkpoint(model, optimizer, epoch, val_loss, history, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': history,
            'task': vars(task),
            'net': vars(net),
        }, path)


    history = {
        'train_ctrl_loss': [], 'val_ctrl_loss': [],
        'train_imp': [], 'val_imp': [],
    }
    best_val = float('inf')


    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_ctrl, ep_imp = [], []
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
    
        for b in train_loader:
            b = to_dev(b)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = model(b['x0'], b['goal'], b['t_rem'], return_history=False)
                loss, m = criterion(out, b['u_opt'])
    
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    
            ep_ctrl.append(m['final_loss'])
            ep_imp.append(m['imp_metric'])
            scheduler.step()
    
        val_ctrl, val_imp = evaluate(model, test_loader)
        train_ctrl, train_imp = np.mean(ep_ctrl), np.mean(ep_imp)
    
        history['train_ctrl_loss'].append(train_ctrl)
        history['val_ctrl_loss'].append(val_ctrl)
        history['train_imp'].append(train_imp)
        history['val_imp'].append(val_imp)
    
        if val_ctrl < best_val:
            best_val = val_ctrl
            save_checkpoint(model, optimizer, epoch, val_ctrl, history, CKPT_PATH)
            marker = ' ★ saved'
        else:
            marker = ''
    
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f'Epoch {epoch:3d}/{EPOCHS}  train_ctrl={train_ctrl:.4f}  imp={train_imp:.3f}  '
              f'val_ctrl={val_ctrl:.4f}  val_imp={val_imp:.3f}  ({dt:.1f}s){marker}')
    
    
    # Learning curves (Fig. 2 style)
    ep = np.arange(1, EPOCHS + 1)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Left axis: Control Loss (log scale)
    ax1.semilogy(ep, history['train_ctrl_loss'], 'b-', alpha=0.7, label='Train Control Loss')
    ax1.semilogy(ep, history['val_ctrl_loss'], 'b--', alpha=0.7, label='Val Control Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Control Loss (MSE)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Right axis: Improvement metric
    ax2 = ax1.twinx()
    ax2.plot(ep, history['train_imp'], 'g-', alpha=0.7, label='Train Improvement')
    ax2.plot(ep, history['val_imp'], 'g--', alpha=0.7, label='Val Improvement')
    ax2.set_ylabel('Improvement', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    ax1.grid(True, alpha=0.3)
    fig.suptitle('Fig. 2: Training Convergence', fontsize=14)
    plt.tight_layout(); plt.show()
    
    COLORS = ['#7b2d8e', '#2b8f8f', '#4ca64c', '#d4b830', '#C44E52']
    
    
    
    
    
    
    
    
    # Load best checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]} (val_loss={ckpt["val_loss"]:.4f})')
    
    # Rebuild model from saved config
    task_ck = TaskConfig(**ckpt['task'])
    net_ck  = NetConfig(**ckpt['net'])
    model_eval = TRC(task_ck, net_ck, dynamics_fn=vdp_dynamics).to(device)
    model_eval.set_cost_matrices(
        Q  = torch.diag(torch.tensor([10., 5.], device=device)),
        R  = torch.tensor([[0.5]], device=device),
        Qf = torch.diag(torch.tensor([200., 100.], device=device)),
    )
    model_eval.load_state_dict(ckpt['model_state_dict'])
    model_eval.eval()
    print(f'Model: {count_params(model_eval):,} params, K={net_ck.K}, n={net_ck.n_inner}')
    
    
    
    
    # Run on full test set
    all_x0   = test_ds.x0.to(device, non_blocking=use_cuda)
    all_goal = test_ds.goal.to(device, non_blocking=use_cuda)
    all_trem = test_ds.t_rem.to(device, non_blocking=use_cuda)
    all_uopt = test_ds.u_optimal.numpy()
    opt_costs = test_ds.costs.numpy()
    
    with torch.no_grad():
        with autocast_ctx():
            out = model_eval(all_x0, all_goal, all_trem, return_history=True)
    
    u_iters = [u.cpu().numpy() for u in out['u_iterations']]
    costs   = [c.cpu().numpy() for c in out['costs']]
    z_H     = [z.cpu().numpy() for z in out['z_H_history']]
    

    
    K = len(u_iters) - 1
    
    print(f'Test samples: {len(opt_costs)}')
    for k in range(K+1):
        print(f'  Iter {k}: cost = {costs[k].mean():.1f} ± {costs[k].std():.1f}')
    reduction = (1 - costs[-1].mean() / costs[0].mean()) * 100
    gap = (costs[-1].mean() / opt_costs.mean() - 1) * 100
    print(f'Cost reduction: {reduction:.1f}%')
    print(f'Optimal cost: {opt_costs.mean():.1f}, TRC cost: {costs[-1].mean():.1f}, gap: {gap:+.1f}%')
    
    
    # --- Figure 3: Trajectory results ---
    with torch.no_grad():
        states_final = model_eval.sim(all_x0, out['u_iterations'][-1]).cpu().numpy()
    
    time_ax = np.arange(T+1) * DT
    n_show = min(20, len(states_final))
    cmap = plt.cm.tab10
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i in range(n_show):
        c = cmap(i / max(n_show-1, 1))
        axes[0,0].plot(time_ax, states_final[i,:,0], color=c, alpha=0.7)
        axes[0,1].plot(time_ax, states_final[i,:,1], color=c, alpha=0.7)
        axes[1,0].plot(np.arange(T)*DT, u_iters[-1][i,:,0], color=c, alpha=0.7)
        axes[1,1].plot(states_final[i,:,0], states_final[i,:,1], color=c, alpha=0.7)
    axes[1,1].plot(0, 0, 'r*', ms=15, zorder=5)
    axes[0,0].set(xlabel='Time (s)', ylabel='Position')
    axes[0,1].set(xlabel='Time (s)', ylabel='Velocity')
    axes[1,0].set(xlabel='Time (s)', ylabel='Control')
    axes[1,1].set(xlabel='Position', ylabel='Velocity')
    fig.suptitle('Fig. 3: Trajectory Results', fontsize=14)
    plt.tight_layout(); plt.show()
    
    
    
    
    # --- Figure 4: Iterative refinement ---

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
   
    for k in range(K+1):
        ctrl = u_iters[k][:, :, 0]
        med = np.median(ctrl, axis=0)
        p25, p75 = np.percentile(ctrl, [25, 75], axis=0)
        ax1.plot(med, color=COLORS[k], lw=1.5, label=f'Iter {k}', alpha=0.9)
        ax1.fill_between(range(len(med)), p25, p75, color=COLORS[k], alpha=0.15)
    ax1.axhline(U_MAX, color='gray', ls='--', alpha=0.4)
    ax1.axhline(U_MIN, color='gray', ls='--', alpha=0.4)
    ax1.set(xlabel='Time Step', ylabel='Control', title='(a) Control Evolution')
    ax1.legend()
    
    mean_c = [costs[k].mean() for k in range(K+1)]
    std_c  = [costs[k].std() for k in range(K+1)]
    ax2.bar(range(K+1), mean_c, yerr=std_c, color=COLORS[:K+1], alpha=0.8, capsize=5, edgecolor='k', lw=0.5)
    ax2.set(xlabel='Iteration', ylabel='Cost', title='(b) Cost Reduction')
    if mean_c[0] > 0:
        red = (1 - mean_c[-1]/mean_c[0]) * 100
        ax2.annotate(f'{red:.0f}% reduction', xy=(K, mean_c[-1]),
                     xytext=(K-0.5, mean_c[0]*0.5),
                     arrowprops=dict(arrowstyle='->', color='red'), color='red', fontweight='bold')
    
    fig.suptitle('Fig. 4: Iterative Refinement', fontsize=14)
    plt.tight_layout(); plt.show()
    
    
    # --- Figure 5: Latent space evolution (PCA of z_H) ---
    all_z = np.concatenate(z_H, axis=0)
    mean_z = all_z.mean(axis=0)
    centered = all_z - mean_z
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:2].T
    var_exp = S[:2]**2 / (S**2).sum()
    
    z_2d = [(z - mean_z) @ basis for z in z_H]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for k in range(K+1):
        ax1.scatter(z_2d[k][:,0], z_2d[k][:,1], c=COLORS[k], s=20, alpha=0.5, label=f'Iter {k}')
        cx, cy = z_2d[k].mean(0)
        ax1.plot(cx, cy, 'x', color=COLORS[k], ms=10, mew=2)
    ax1.set(xlabel=f'PC1 ({var_exp[0]*100:.1f}%)', ylabel=f'PC2 ({var_exp[1]*100:.1f}%)',
            title='Latent Space (colored by iteration)')
    ax1.legend()
    
    final_cost = costs[-1]
    norm = plt.Normalize(final_cost.min(), final_cost.max())
    cmap_cost = plt.cm.RdYlGn_r
    N = len(final_cost)
    for i in range(min(N, 200)):
        px = [z_2d[k][i, 0] for k in range(K+1)]
        py = [z_2d[k][i, 1] for k in range(K+1)]
        ax2.plot(px, py, '-', color=cmap_cost(norm(final_cost[i])), alpha=0.4, lw=0.8)
        ax2.plot(px[0], py[0], 'o', color=cmap_cost(norm(final_cost[i])), ms=3)
        ax2.plot(px[-1], py[-1], 's', color=cmap_cost(norm(final_cost[i])), ms=4)
    sm = plt.cm.ScalarMappable(cmap=cmap_cost, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Final Cost')
    ax2.set(xlabel=f'PC1 ({var_exp[0]*100:.1f}%)', ylabel=f'PC2 ({var_exp[1]*100:.1f}%)',
            title='Refinement Paths (colored by cost)')
    
    fig.suptitle('Fig. 5: Latent Space Evolution', fontsize=14)
    plt.tight_layout(); plt.show()
    
    
    
    # --- Single sample: expert vs TRC ---
    idx = 0
    with torch.no_grad():
        x0_1 = all_x0[idx:idx+1]
        u_pred = out['u_final'][idx].cpu().numpy()
        x_pred = model_eval.sim(x0_1, out['u_final'][idx:idx+1]).cpu().numpy()[0]
    u_star = all_uopt[idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(u_star[:,0], label='Expert u*'); axes[0].plot(u_pred[:,0], label='TRC u', alpha=0.8)
    axes[0].set(xlabel='Step', ylabel='u', title='Control'); axes[0].legend(); axes[0].grid(alpha=0.3)
    
    axes[1].plot(x_pred[:,0], label='TRC x'); axes[1].plot(x_pred[:,1], label='TRC ẋ')
    axes[1].set(xlabel='Step', title='TRC State'); axes[1].legend(); axes[1].grid(alpha=0.3)
    
    axes[2].plot(x_pred[:,0], x_pred[:,1], label='TRC')
    axes[2].plot(x_pred[0,0], x_pred[0,1], 'go', ms=8); axes[2].plot(0, 0, 'r*', ms=12)
    axes[2].set(xlabel='x', ylabel='ẋ', title='Phase'); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    
    
    
    
    
    
if __name__ == '__main__':
    mp.freeze_support()  # Safe on Windows
    main()











