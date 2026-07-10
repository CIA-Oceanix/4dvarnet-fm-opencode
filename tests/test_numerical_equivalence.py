"""
Numerical equivalence test: compare refactored baselines (with dynamics)
against the old inline L63 code.
Runs a single trajectory through both code paths and checks bit-level match.
"""
import os, sys
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz63 import generate_long_trajectory, generate_observations
from models.lorenz63_dynamics import Lorenz63Dynamics, _apply_coupling
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dt = 0.01
num_steps = 300
seed = 42
sigma, rho, beta = 10.0, 28.0, 8/3
c1 = 1.0
R_var = 0.5

# Generate reference trajectory (GPU is fast)
traj_path = "/tmp/test_traj.pt"
if os.path.exists(traj_path):
    traj_full = torch.load(traj_path, map_location=device)
    print(f"Loaded trajectory from {traj_path}")
else:
    traj_full = generate_long_trajectory(num_steps=num_steps + 10000, dt=dt, seed=seed,
                                          sigma=sigma, rho=rho, beta=beta,
                                          gamma=0.05, W_L_bar=0.0, c1=c1, c2=0.1,
                                          sigma_0=0.08, sigma_L=0.20,
                                          coupling_exponent=1.6, device=device)
    torch.save(traj_full.cpu(), traj_path)
state_traj = traj_full[-num_steps:, :3]
forcing_traj = traj_full[-num_steps:, 3]

# Build observations
obs, obs_mask = generate_observations(state_traj, obs_interval=1, R_var=R_var, seed=43, device=device)

# --- Test: Single Euler step equality ---
print("=== Test 1: Single Euler step (Lorenz63Dynamics.step vs inline) ===")
dynamics = Lorenz63Dynamics(dt=dt, coupling_exponent=1.6)
s = state_traj[0]
W = forcing_traj[0]
new_dyn = dynamics.step(s.unsqueeze(0), W.unsqueeze(0), sigma, rho, beta)

X, Y, Z = s[0], s[1], s[2]
coupling = _apply_coupling(W.unsqueeze(0), c1, 1.6).squeeze(0)
dX = sigma * (Y - X) + coupling
dY = X * (rho - Z) - Y
dZ = X * Y - beta * Z
new_inline = torch.stack([X + dX * dt, Y + dY * dt, Z + dZ * dt], dim=-1)

diff = (new_dyn - new_inline).abs().max().item()
print(f"  Max diff: {diff:.3e}  {'PASS' if diff < 1e-10 else 'FAIL'}")

# --- Test: Full trajectory rollout ---
print("=== Test 2: Full trajectory rollout ===")
def inline_rollout(s0, forcing, steps, sigma, rho, beta, c1, dt):
    traj = [s0.clone()]
    s = s0.clone()
    for t in range(1, steps):
        X, Y, Z = s[0], s[1], s[2]
        W = forcing[t-1]
        coupling_line = _apply_coupling(W.unsqueeze(0), c1, 1.6).squeeze(0)
        dX = sigma * (Y - X) + coupling_line
        dY = X * (rho - Z) - Y
        dZ = X * Y - beta * Z
        s = torch.stack([X + dX * dt, Y + dY * dt, Z + dZ * dt], dim=-1)
        traj.append(s)
    return torch.stack(traj, dim=0)

inline_traj = inline_rollout(state_traj[0], forcing_traj, num_steps, sigma, rho, beta, c1, dt)
dyn_traj = [state_traj[0].unsqueeze(0)]
s = state_traj[0].unsqueeze(0)
for t in range(1, num_steps):
    s = dynamics.step(s, forcing_traj[t-1:t], sigma, rho, beta)
    dyn_traj.append(s)
dyn_traj = torch.cat(dyn_traj, dim=0)

rollout_diff = (dyn_traj - inline_traj).abs().max().item()
print(f"  Max diff: {rollout_diff:.3e}  {'PASS' if rollout_diff < 1e-10 else 'FAIL'}")

import signal
class TimeoutError(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")
signal.signal(signal.SIGALRM, timeout_handler)

# --- Test: Weak4DVar equivalence ---
print("=== Test 3: Weak4DVar (with dynamics) produces non-NaN RMSE ===")
weak = Weak4DVar(da_window_steps=num_steps, dt=dt, device=device, coupling_exponent=1.6, dynamics=dynamics)
signal.alarm(30)
result = weak.assimilate(obs, obs_mask, forcing_traj, state_traj, sigma=sigma, rho=rho, beta=beta, c1=c1)
signal.alarm(0)
print(f"  RMSE: X={result.rmse[0]:.4f} Y={result.rmse[1]:.4f} Z={result.rmse[2]:.4f}")
print(f"  Has NaN: {np.any(np.isnan(result.trajectory))}")
print(f"  Has non-finite: {np.any(~np.isfinite(result.trajectory))}")

# --- Test: Strong4DVar ---
print("=== Test 4: Strong4DVar (with dynamics) produces non-NaN RMSE ===")
strong = Strong4DVar(da_window_steps=num_steps, dt=dt, device=device, coupling_exponent=1.6, dynamics=dynamics)
signal.alarm(30)
result_s = strong.assimilate(obs, obs_mask, forcing_traj, state_traj, sigma=sigma, rho=rho, beta=beta, c1=c1)
signal.alarm(0)
print(f"  RMSE: X={result_s.rmse[0]:.4f} Y={result_s.rmse[1]:.4f} Z={result_s.rmse[2]:.4f}")
print(f"  Has NaN: {np.any(np.isnan(result_s.trajectory))}")

# --- Test: EnKF ---
print("=== Test 5: EnKF (with dynamics) produces non-NaN RMSE ===")
enkf = EnKF(N_ensemble=30, inflation=1.0, dt=dt, device=device, coupling_exponent=1.6, dynamics=dynamics)
signal.alarm(30)
result_e = enkf.assimilate(obs, obs_mask, forcing_traj, state_traj, sigma=sigma, rho=rho, beta=beta, c1=c1)
signal.alarm(0)
print(f"  RMSE: X={result_e.rmse[0]:.4f} Y={result_e.rmse[1]:.4f} Z={result_e.rmse[2]:.4f}")

# --- Test: ETKF ---
print("=== Test 6: ETKF (with dynamics) produces non-NaN RMSE ===")
etkf = ETKF(N_ensemble=30, inflation=1.0, dt=dt, device=device, coupling_exponent=1.6, dynamics=dynamics)
signal.alarm(30)
result_t = etkf.assimilate(obs, obs_mask, forcing_traj, state_traj, sigma=sigma, rho=rho, beta=beta, c1=c1)
signal.alarm(0)
print(f"  RMSE: X={result_t.rmse[0]:.4f} Y={result_t.rmse[1]:.4f} Z={result_t.rmse[2]:.4f}")

print()
print(f"Step rollout diff: {rollout_diff:.3e}")
print(f"Weak4DVar OK: {not np.any(np.isnan(result.trajectory))}")
print(f"Strong4DVar OK: {not np.any(np.isnan(result_s.trajectory))}")
print(f"EnKF OK: {not np.any(np.isnan(result_e.trajectory))}")
print(f"ETKF OK: {not np.any(np.isnan(result_t.trajectory))}")
print("=== ALL TESTS DONE ===")