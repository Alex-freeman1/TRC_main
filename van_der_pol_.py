# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:37:51 2026

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



from trc_main_ import TRC, TRCLoss, TaskConfig, NetConfig, vdp_dynamics, count_params

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
FORCE_REGEN = False
PARALLEL_GEN = True
GEN_WORKERS = max(1, (os.cpu_count() or 1) - 1)
if QUICK:
    GEN_WORKERS = min(GEN_WORKERS, 4)

# Training
EPOCHS     = 5 if QUICK else 50
BATCH_SIZE = 16 if QUICK else 64
LR         = 1e-3
LAMBDA_PS  = 0.3

# Problem constants
MU    = 1.0
DT    = 0.05
T     = 100
U_MIN = -2.0
U_MAX = 2.0


Q     = np.diag([10.0, 5.0])
R     = 0.5
Q_F   = 20.0 * Q

CKPT_PATH = CKPT_DIR / 'trc_vdp_best.pt'

print(f'N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}')
print(f'Parallel data gen: {PARALLEL_GEN} (workers={GEN_WORKERS})')




import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor



'''
Van Der Pol dynamics 
'''
def vdp_np(x, u):
    return np.array([x[1], MU * (1 - x[0]**2) * x[1] - x[0] + float(u)])



'''
Runge-kutta gen 4 method for integration
'''

def rk4_np(x, u):
    k1 = vdp_np(x, u)
    k2 = vdp_np(x + 0.5 * DT * k1, u)
    k3 = vdp_np(x + 0.5 * DT * k2, u)
    k4 = vdp_np(x + DT * k3, u)
    return x + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


'''
Extrapolate time sequence using rk4 integrator
'''
def rollout_np(x0, u_seq):
    states = np.zeros((T + 1, 2), dtype=np.float64)
    states[0] = x0
    for t in range(T):
        states[t + 1] = rk4_np(states[t], u_seq[t])
    return states


'''
Calculate the integral, J,  using discrete time 
'''
def cost_np(u_seq, x0):
    states = rollout_np(x0, u_seq)
    
    # Discretly adds state vectors and control inputs weighting
    J = sum(states[t] @ Q @ states[t] + R * u_seq[t]**2 for t in range(T))
    
    # Adds the final weighting, Q_f
    J += states[T] @ Q_F @ states[T]
    return float(J)



'''
Find the gradient of J, \nabla J. 
Uses the costate (adjoint) method and sweeps back in time. 
'''
def cost_grad_np(u_seq, x0):
    states = rollout_np(x0, u_seq)
    costate = 2.0 * Q_F @ states[T]
    grad = np.zeros(T)
    eps = 1e-6
    for t in range(T - 1, -1, -1):
        grad[t] = 2.0 * R * u_seq[t]
        grad[t] += costate @ (rk4_np(states[t], u_seq[t] + eps) - rk4_np(states[t], u_seq[t] - eps)) / (2 * eps)
        dfdx = np.zeros((2, 2))
        for i in range(2):
            xp, xm = states[t].copy(), states[t].copy()
            xp[i] += eps
            xm[i] -= eps
            dfdx[:, i] = (rk4_np(xp, u_seq[t]) - rk4_np(xm, u_seq[t])) / (2 * eps)
        costate = 2.0 * Q @ states[t] + dfdx.T @ costate
    return grad



'''
Solves one singular optimal control input u*
'''
def solve_one(x0):
    return minimize(
        cost_np,
        np.zeros(T),
        args=(x0,),
        jac=cost_grad_np,
        method='SLSQP',
        bounds=[(U_MIN, U_MAX)] * T,
        options={'maxiter': 300, 'ftol': 1e-8},
    )

'''
This function solves the problem and collects everything required
'''
def _solve_one_payload(x0):
    result = solve_one(x0)
    u_opt = np.clip(result.x, U_MIN, U_MAX)
    traj = rollout_np(x0, u_opt)
    cost_val = cost_np(u_opt, x0)
    return x0.astype(np.float32), u_opt.reshape(T, 1).astype(np.float32), traj.astype(np.float32), float(cost_val), bool(result.success)


def _pack_dataset(x0s, u_opts, trajs, costs_arr, success):
    n = len(x0s)
    return {
        'x0': np.asarray(x0s, dtype=np.float32),
        'x_target': np.zeros((n, 2), dtype=np.float32),
        't_remaining': np.full((n, 1), T * DT, dtype=np.float32),
        'u_optimal': np.asarray(u_opts, dtype=np.float32),
        'x_trajectory': np.asarray(trajs, dtype=np.float32),
        'costs': np.asarray(costs_arr, dtype=np.float32),
        'success': np.asarray(success),
    }


def _generate_dataset_serial(x0_samples, verbose_every):
    x0s, u_opts, trajs, costs_arr, success = [], [], [], [], []
    n = len(x0_samples)
    for i, x0 in enumerate(x0_samples):
        x0_i, u_i, traj_i, cost_i, ok_i = _solve_one_payload(x0)
        if i == 0 or (i + 1) % verbose_every == 0 or i == n - 1:
            print(f'  [{i+1}/{n}] x0=[{x0_i[0]:+.2f},{x0_i[1]:+.2f}] cost={cost_i:.1f} {"OK" if ok_i else "FAIL"}')
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
    
    




def main():
    
    def generate_dataset(n, seed=42, verbose_every=100, parallel=True, workers=None):
        rng = np.random.RandomState(seed)
        
        x0_samples = rng.uniform(-2, 2, size=(n, 2)).astype(np.float64)
        verbose_every = max(1, int(verbose_every))

        if (not parallel) or n <= 1:
            return _generate_dataset_serial(x0_samples, verbose_every)

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
        print(f'Generating dataset in parallel with {workers} workers...')
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
        



    ''' 
        
        This determines which dataset to use
        my_vdp_train and my_vdp_test are from Minduli's dataset and are 
        N_TRAIN = 10_000
        N_TEST  = 1_000
        
        From playing around, it seems as though a neural network needs more data than that 
        of the smaller set and actually needs the full 10000 and 1000 dataset. 
        
    '''

    train_path = DATA_DIR / 'vdp_train.npz'
    test_path  = DATA_DIR / 'vdp_test.npz'

   #  train_path = Path(r"C:\Users\alexa\Git Projects\Tiny Recursive Control\my_vdp_train.npz")
   #  test_path = Path(r"C:\Users\alexa\Git Projects\Tiny Recursive Control\my_vdp_test.npz")
    
    
    

    if FORCE_REGEN or not train_path.exists():
        print('Generating training data...')
        np.savez(
            train_path,
            **generate_dataset(
                N_TRAIN,
                seed=42,
                verbose_every=max(1, N_TRAIN // 8),
                parallel=PARALLEL_GEN,
                workers=GEN_WORKERS,
            ),
        )
    else:
        print(f'Using existing: {train_path}')

    if FORCE_REGEN or not test_path.exists():
        print('Generating test data...')
        np.savez(
            test_path,
            **generate_dataset(
                N_TEST,
                seed=123,
                verbose_every=max(1, N_TEST // 4),
                parallel=PARALLEL_GEN,
                workers=GEN_WORKERS,
            ),
        )
    else:
        print(f'Using existing: {test_path}')

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
    
    
    task = TaskConfig(state_dim=2, control_dim=1, horizon=T, dt=DT, u_min=U_MIN, u_max=U_MAX)
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

