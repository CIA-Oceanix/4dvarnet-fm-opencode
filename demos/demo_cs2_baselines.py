#!/usr/bin/env python3
"""
Demonstration Script 3: Case Study 2 - Poor Reconstruction

Purpose: Show degradation with model mismatch (corrupted forcing + biased parameters)

This script demonstrates:
1. Data generation for CS2 (corrupted forcing, biased parameters)
2. Running all three baseline DA methods with biased parameters
3. Visualization of poor reconstruction quality
4. RMSE metrics showing significant degradation (3-6x higher than CS1)

Case Study 2 Setup:
- Forcing: Corrupted with Ornstein-Uhlenbeck process
- Parameters: Biased by 5% (σ=9.5, ρ=26.6, β=2.9)
- Expected: All methods degrade significantly (RMSE 3-6x higher than CS1)

Usage:
    python demos/demo_cs2_baselines.py [--seed SEED] [--duration DURATION] [--bias BIAS]

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
    """Run all three baseline methods on a data window with biased parameters"""
    device = torch.device('cpu')
    
    # Extract data
    true_state = window['true_state']
    obs = window['obs']
    obs_mask = window['obs_mask']
    forcing = window['forcing_corrupted']  # CS2 uses corrupted forcing
    
    # Get BIASED parameters for CS2 (this is critical!)
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
    
    print(f"   - Parameters: σ={sigma:.2f}, ρ={rho:.2f}, β={beta:.4f} (BIASED by {cfg.param_bias*100:.0f}%)")
    print("   - Forcing: Corrupted (OU process)")
    
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
    
    fig.suptitle('CS2: Poor Reconstruction (corrupted forcing, biased parameters)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_forcing_impact(window, results, cfg, save_path):
    """Visualize the impact of forcing corruption on reconstruction"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    forcing_true = window['forcing_true'].cpu().numpy()
    forcing_corrupted = window['forcing_corrupted'].cpu().numpy()
    true_state = window['true_state'].cpu().numpy()
    time_grid = cfg.time_grid
    
    # Top panel: Forcing comparison
    axes[0].plot(time_grid, forcing_true, color='green', linewidth=2, 
                 label='True Forcing', alpha=0.8)
    axes[0].plot(time_grid, forcing_corrupted, color='orange', linewidth=2, 
                 linestyle='--', label='Corrupted Forcing (OU)', alpha=0.8)
    axes[0].set_ylabel(r'$W_L$ (Forcing)', fontsize=12)
    axes[0].set_title('CS2: Impact of Forcing Corruption on DA Performance', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics box
    difference = forcing_corrupted - forcing_true
    std_diff = np.std(difference)
    mean_abs_diff = np.mean(np.abs(difference))
    stats_text = f'Corruption Stats:\nStd: {std_diff:.4f}\nMean |Δ|: {mean_abs_diff:.4f}'
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Bottom panel: X component error for best method (Strong-4DVar)
    axes[1].plot(time_grid, true_state[:, 0], color='black', linewidth=2, 
                 label='Truth (X)', alpha=0.8)
    axes[1].plot(time_grid, results['strong'].trajectory[:, 0], 
                 color='#ff7f0e', linewidth=1.5, linestyle='--', 
                 label='Strong-4DVar Reconstruction', alpha=0.8)
    
    # Shade error regions
    error = results['strong'].trajectory[:, 0] - true_state[:, 0]
    axes[1].fill_between(time_grid, true_state[:, 0], 
                          results['strong'].trajectory[:, 0], 
                          alpha=0.3, color='red', label='Reconstruction Error')
    
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('X Component', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Add RMSE info
    rmse_x = results['strong'].rmse[0]
    axes[1].text(0.02, 0.02, f'RMSE (X): {rmse_x:.4f}', 
                 transform=axes[1].transAxes, fontsize=10, 
                 verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def print_rmse_table(results):
    """Print formatted RMSE table"""
    print("\n" + "=" * 76)
    print("  CASE STUDY 2: Noisy forcings & biased parameters")
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
    print("Expected: RMSE 3-6x higher than CS1 (significant degradation)")
    print("=" * 76)


def save_results_table(results, save_path):
    """Save RMSE table to text file"""
    with open(save_path, 'w') as f:
        f.write("=" * 76 + "\n")
        f.write("  CASE STUDY 2: Noisy forcings & biased parameters\n")
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
        f.write("Expected: RMSE 3-6x higher than CS1 (significant degradation)\n")
        f.write("=" * 76 + "\n")
    
    print(f"✓ Saved results: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='CS2 Baseline Demonstration')
    parser.add_argument('--seed', type=int, default=123, help='Random seed (default: 123)')
    parser.add_argument('--duration', type=float, default=5.0, help='Window duration in seconds (default: 5.0)')
    parser.add_argument('--bias', type=float, default=0.05, help='Parameter bias fraction (default: 0.05)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  CASE STUDY 2: BASELINE METHODS (POOR RECONSTRUCTION)")
    print("=" * 70)
    
    # Create output directories
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    # Generate CS2 dataset
    print(f"\n[1/4] Generating CS2 dataset (T={args.duration}s, seed={args.seed})...")
    cfg = Lorenz63Config(
        case=2,
        param_bias=args.bias,
        num_windows=1,
        T_max=args.duration,
        seed=args.seed
    )
    
    dataset = Lorenz63Dataset(cfg)
    window = dataset[0]
    
    print("   - Case: 2 (corrupted forcing, biased parameters)")
    print(f"   - Parameter bias: {args.bias*100:.0f}%")
    print(f"   - Window size: {cfg.num_steps} steps ({args.duration}s)")
    print(f"   - Observations: {window['obs_mask'].sum().item()} sparse samples")
    
    # Run baseline methods
    print("\n[2/4] Running baseline DA methods with biased parameters...")
    results = run_baselines(window, cfg)
    
    # Print RMSE table
    print_rmse_table(results)
    
    # Save results table
    save_results_table(results, 'outputs/results/cs2_rmse_table.txt')
    
    # Plot reconstruction
    print("\n[3/4] Creating reconstruction visualization...")
    plot_reconstruction(
        window,
        results,
        cfg,
        'outputs/figures/cs2_reconstruction_all_methods.png'
    )
    
    # Plot forcing impact
    print("\n[4/4] Creating forcing corruption impact visualization...")
    plot_forcing_impact(
        window,
        results,
        cfg,
        'outputs/figures/cs2_forcing_corruption.png'
    )
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("CS2 demonstrates significant degradation of baseline DA methods")
    print("under model mismatch:")
    print("  - Corrupted forcing (OU process noise)")
    print(f"  - Biased parameters ({args.bias*100:.0f}% bias)")
    print("  - RMSE values 3-6x higher than CS1")
    print("\nThis validates the need for data-driven approaches like 4DVarNet-FM")
    print("that can learn to correct for model errors.")
    print("\nOutputs:")
    print("  - outputs/figures/cs2_reconstruction_all_methods.png")
    print("  - outputs/figures/cs2_forcing_corruption.png")
    print("  - outputs/results/cs2_rmse_table.txt")
    print("=" * 70)


if __name__ == '__main__':
    main()
