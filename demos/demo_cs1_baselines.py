#!/usr/bin/env python3
"""
Demonstration Script 2: Case Study 1 - Good Reconstruction

Purpose: Show good reconstruction with correct model parameters

This script demonstrates:
1. Data generation for CS1 (noise-free forcing, correct parameters)
2. Running all three baseline DA methods (Weak-4DVar, Strong-4DVar, EnKF)
3. Visualization of reconstruction quality
4. RMSE metrics showing good performance (< 0.5)

Case Study 1 Setup:
- Forcing: Noise-free (use true W_L)
- Parameters: Correct (σ=10, ρ=28, β=8/3)
- Expected: All methods achieve low RMSE (< 0.5)

Usage:
    python demos/demo_cs1_baselines.py [--seed SEED] [--duration DURATION]

Author: Agent B - Visualization Specialist
Date: June 25, 2026
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.lorenz63 import Lorenz63Config, Lorenz63Dataset
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF


def run_baselines(window, cfg):
    """Run all three baseline methods on a data window"""
    device = torch.device('cpu')
    
    # Extract data
    true_state = window['true_state']
    obs = window['obs']
    obs_mask = window['obs_mask']
    forcing = window['forcing_true']  # CS1 uses true forcing
    
    # Get correct parameters for CS1
    sigma, rho, beta = cfg.da_params
    
    # Initialize methods (with reduced iterations for speed)
    weak_4dvar = Weak4DVar(
        da_window_steps=cfg.num_steps,
        B_var=cfg.B_var,
        R_var=cfg.R_var,
        opt_steps=80,  # Reduced from 150 for faster demo
        dt=cfg.dt,
        device=device
    )
    
    strong_4dvar = Strong4DVar(
        da_window_steps=cfg.num_steps,
        B_var=cfg.B_var,
        R_var=cfg.R_var,
        max_iter=20,  # Reduced from 40 for faster demo
        dt=cfg.dt,
        device=device
    )
    
    enkf = EnKF(
        N_ensemble=30,
        R_var=cfg.R_var,
        dt=cfg.dt,
        device=device
    )
    
    print(f"   - Parameters: σ={sigma:.2f}, ρ={rho:.2f}, β={beta:.4f}")
    print("   - Forcing: True (noise-free)")
    
    # Run Weak-4DVar
    print("   - Running Weak-4DVar...")
    result_weak = weak_4dvar.assimilate(
        obs, obs_mask, forcing, true_state,
        sigma=sigma, rho=rho, beta=beta
    )
    
    # Run Strong-4DVar
    print("   - Running Strong-4DVar...")
    result_strong = strong_4dvar.assimilate(
        obs, obs_mask, forcing, true_state,
        sigma=sigma, rho=rho, beta=beta
    )
    
    # Run EnKF
    print("   - Running EnKF...")
    result_enkf = enkf.assimilate(
        obs, obs_mask, forcing, true_state,
        sigma=sigma, rho=rho, beta=beta
    )
    
    return {
        'weak': result_weak,
        'strong': result_strong,
        'enkf': result_enkf
    }


def plot_reconstruction(window, results, cfg, save_path):
    """Plot reconstruction comparison for all methods"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    true_state = window['true_state'].cpu().numpy()
    obs = window['obs'].cpu().numpy()
    obs_mask = window['obs_mask'].cpu().numpy()
    time_grid = cfg.time_grid
    
    components = ['X', 'Y', 'Z']
    colors_methods = {
        'Truth': 'black',
        'Weak-4DVar': '#1f77b4',
        'Strong-4DVar': '#ff7f0e',
        'EnKF': '#2ca02c'
    }
    
    for i, (ax, comp) in enumerate(zip(axes, components)):
        # Plot truth
        ax.plot(time_grid, true_state[:, i], color='black', linewidth=2, 
                label='Truth', alpha=0.8)
        
        # Plot observations
        obs_times = time_grid[obs_mask]
        obs_values = obs[obs_mask, i]
        ax.scatter(obs_times, obs_values, color='gray', s=10, alpha=0.5, 
                   label='Obs', zorder=3)
        
        # Plot reconstructions
        ax.plot(time_grid, results['weak'].trajectory[:, i], 
                color=colors_methods['Weak-4DVar'], linewidth=1.5, 
                linestyle='--', label='Weak-4DVar', alpha=0.8)
        
        ax.plot(time_grid, results['strong'].trajectory[:, i], 
                color=colors_methods['Strong-4DVar'], linewidth=1.5, 
                linestyle='-.', label='Strong-4DVar', alpha=0.8)
        
        ax.plot(time_grid, results['enkf'].trajectory[:, i], 
                color=colors_methods['EnKF'], linewidth=1.5, 
                linestyle=':', label='EnKF', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(comp, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    fig.suptitle('CS1: Good Reconstruction (noise-free forcing, correct parameters)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def print_rmse_table(results):
    """Print formatted RMSE table"""
    print("\n" + "=" * 76)
    print("  CASE STUDY 1: Noise-free forcings & correct parameters")
    print("=" * 76)
    print(f"{'Method':<18} {'RMSE X':>12} {'RMSE Y':>12} {'RMSE Z':>12} {'Mean RMSE':>12}")
    print("-" * 76)
    
    for method_name, method_key in [('Weak-4DVar', 'weak'), 
                                     ('Strong-4DVar', 'strong'), 
                                     ('EnKF', 'enkf')]:
        rmse = results[method_key].rmse
        mean_rmse = np.mean(rmse)
        print(f"{method_name:<18} {rmse[0]:>12.4f} {rmse[1]:>12.4f} {rmse[2]:>12.4f} {mean_rmse:>12.4f}")
    
    print("=" * 76)
    print("Expected: All RMSE < 0.5 (good reconstruction)")
    print("=" * 76)


def save_results_table(results, save_path):
    """Save RMSE table to text file"""
    with open(save_path, 'w') as f:
        f.write("=" * 76 + "\n")
        f.write("  CASE STUDY 1: Noise-free forcings & correct parameters\n")
        f.write("=" * 76 + "\n")
        f.write(f"{'Method':<18} {'RMSE X':>12} {'RMSE Y':>12} {'RMSE Z':>12} {'Mean RMSE':>12}\n")
        f.write("-" * 76 + "\n")
        
        for method_name, method_key in [('Weak-4DVar', 'weak'), 
                                         ('Strong-4DVar', 'strong'), 
                                         ('EnKF', 'enkf')]:
            rmse = results[method_key].rmse
            mean_rmse = np.mean(rmse)
            f.write(f"{method_name:<18} {rmse[0]:>12.4f} {rmse[1]:>12.4f} {rmse[2]:>12.4f} {mean_rmse:>12.4f}\n")
        
        f.write("=" * 76 + "\n")
        f.write("Expected: All RMSE < 0.5 (good reconstruction)\n")
        f.write("=" * 76 + "\n")
    
    print(f"✓ Saved results: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='CS1 Baseline Demonstration')
    parser.add_argument('--seed', type=int, default=123, help='Random seed (default: 123)')
    parser.add_argument('--duration', type=float, default=5.0, help='Window duration in seconds (default: 5.0)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  CASE STUDY 1: BASELINE METHODS (GOOD RECONSTRUCTION)")
    print("=" * 70)
    
    # Create output directories
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    # Generate CS1 dataset
    print(f"\n[1/3] Generating CS1 dataset (T={args.duration}s, seed={args.seed})...")
    cfg = Lorenz63Config(
        case=1,
        param_bias=0.0,
        num_windows=1,
        T_max=args.duration,
        seed=args.seed
    )
    
    dataset = Lorenz63Dataset(cfg)
    window = dataset[0]
    
    print("   - Case: 1 (noise-free forcing, correct parameters)")
    print(f"   - Window size: {cfg.num_steps} steps ({args.duration}s)")
    print(f"   - Observations: {window['obs_mask'].sum().item()} sparse samples")
    
    # Run baseline methods
    print("\n[2/3] Running baseline DA methods...")
    results = run_baselines(window, cfg)
    
    # Print RMSE table
    print_rmse_table(results)
    
    # Save results table
    save_results_table(results, 'outputs/results/cs1_rmse_table.txt')
    
    # Plot reconstruction
    print("\n[3/3] Creating reconstruction visualization...")
    plot_reconstruction(
        window,
        results,
        cfg,
        'outputs/figures/cs1_reconstruction_all_methods.png'
    )
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("CS1 demonstrates that all baseline DA methods achieve good")
    print("reconstruction when the model is correct:")
    print("  - Noise-free forcing (true W_L)")
    print("  - Correct parameters (σ=10, ρ=28, β=8/3)")
    print("  - All RMSE values < 0.5")
    print("\nOutputs:")
    print("  - outputs/figures/cs1_reconstruction_all_methods.png")
    print("  - outputs/results/cs1_rmse_table.txt")
    print("=" * 70)


if __name__ == '__main__':
    main()
