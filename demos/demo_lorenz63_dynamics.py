#!/usr/bin/env python3
"""
Demonstration Script 1: Lorenz-63 Dynamics Visualization

Purpose: Visualize the chaotic Lorenz-63 system and data generation process

This script demonstrates:
1. 3D attractor visualization
2. Time series of state variables (X, Y, Z, W_L)
3. Sparse observation sampling
4. Forcing corruption (CS1 vs CS2)

Usage:
    python demos/demo_lorenz63_dynamics.py [--seed SEED] [--duration DURATION]

Author: Agent B - Visualization Specialist
Date: June 25, 2026
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.lorenz63 import Lorenz63Config, Lorenz63Dataset


def plot_3d_attractor(trajectory, save_path):
    """Plot 3D phase space attractor with time coloring"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y, Z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    
    # Color by time
    time_steps = np.arange(len(X))
    colors = plt.cm.viridis(time_steps / len(time_steps))
    
    # Scatter plot with time coloring
    scatter = ax.scatter(X, Y, Z, c=time_steps, cmap='viridis', s=1, alpha=0.6)
    
    # Add trajectory line
    ax.plot(X, Y, Z, color='gray', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Lorenz-63 Attractor (3D Phase Space)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Time Step', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_time_series(trajectory, time_grid, save_path):
    """Plot time series of all state variables"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    labels = ['X', 'Y', 'Z', r'$W_L$ (Forcing)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time_grid, trajectory[:, i], color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(time_grid[0], time_grid[-1])
        
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_title('Lorenz-63 State Variables (Time Series)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_observations(true_state, obs, obs_mask, time_grid, save_path):
    """Visualize sparse observation sampling"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot true X component
    ax.plot(time_grid, true_state[:, 0], color='black', linewidth=2, 
            label='True State (X)', alpha=0.8)
    
    # Overlay sparse observations
    obs_times = time_grid[obs_mask.cpu().numpy()]
    obs_values = obs[obs_mask.cpu().numpy(), 0].cpu().numpy()
    ax.scatter(obs_times, obs_values, color='red', s=50, zorder=5, 
               label=f'Observations (N={len(obs_times)})', edgecolors='darkred')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('X Component', fontsize=12)
    ax.set_title('Sparse Observations (every 20 time steps)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_forcing_comparison(forcing_cs1, forcing_cs2, time_grid, save_path):
    """Compare CS1 (true) vs CS2 (corrupted) forcing"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top panel: Forcing comparison
    axes[0].plot(time_grid, forcing_cs1, color='green', linewidth=2, 
                 label='CS1: True Forcing', alpha=0.8)
    axes[0].plot(time_grid, forcing_cs2, color='orange', linewidth=2, 
                 linestyle='--', label='CS2: Corrupted Forcing (OU process)', alpha=0.8)
    axes[0].set_ylabel(r'$W_L$ (Forcing)', fontsize=12)
    axes[0].set_title('Case Study Comparison: Forcing Corruption', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Bottom panel: Difference (corruption magnitude)
    difference = forcing_cs2 - forcing_cs1
    axes[1].plot(time_grid, difference, color='purple', linewidth=1.5)
    axes[1].fill_between(time_grid, 0, difference, color='purple', alpha=0.3, 
                          label='Corruption Magnitude')
    axes[1].axhline(0, color='black', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel(r'$\Delta W_L$ (Difference)', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics
    std_diff = np.std(difference)
    axes[1].text(0.02, 0.98, f'Corruption Std: {std_diff:.4f}', 
                 transform=axes[1].transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Lorenz-63 Dynamics Visualization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--duration', type=float, default=30.0, help='Simulation duration in seconds (default: 30.0)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  LORENZ-63 DYNAMICS VISUALIZATION")
    print("=" * 70)
    
    # Create output directory
    os.makedirs('outputs/figures', exist_ok=True)
    
    # Generate long trajectory for visualization
    print(f"\n[1/5] Generating long trajectory (T={args.duration}s, seed={args.seed})...")
    cfg_long = Lorenz63Config(case=1, num_windows=1, T_max=args.duration, seed=args.seed)
    dataset_long = Lorenz63Dataset(cfg_long)
    window_long = dataset_long[0]
    
    trajectory_full = window_long['true_state'].cpu().numpy()
    forcing_true = window_long['forcing_true'].cpu().numpy()
    
    # Add forcing as 4th dimension for visualization
    trajectory_with_forcing = np.concatenate([
        trajectory_full,
        forcing_true.reshape(-1, 1)
    ], axis=1)
    
    time_grid_long = cfg_long.time_grid
    num_steps = len(trajectory_full)
    num_obs = np.sum(window_long['obs_mask'].cpu().numpy())
    
    print(f"   - Trajectory: {num_steps} steps over {args.duration:.1f} seconds")
    print(f"   - Observations: {num_obs} sparse samples (every {cfg_long.obs_interval} steps)")
    
    # Plot 1: 3D Attractor
    print("\n[2/5] Creating 3D attractor visualization...")
    plot_3d_attractor(
        trajectory_full,
        'outputs/figures/lorenz63_attractor_3d.png'
    )
    
    # Plot 2: Time Series
    print("\n[3/5] Creating time series plots...")
    plot_time_series(
        trajectory_with_forcing,
        time_grid_long,
        'outputs/figures/lorenz63_timeseries.png'
    )
    
    # Plot 3: Observations
    print("\n[4/5] Creating observation visualization...")
    plot_observations(
        window_long['true_state'].cpu().numpy(),
        window_long['obs'],
        window_long['obs_mask'],
        time_grid_long,
        'outputs/figures/observations_visualization.png'
    )
    
    # Plot 4: Forcing Comparison (CS1 vs CS2)
    print("\n[5/5] Creating forcing comparison (CS1 vs CS2)...")
    # Generate CS2 data for comparison
    cfg_cs1 = Lorenz63Config(case=1, num_windows=1, T_max=5.0, seed=123)
    cfg_cs2 = Lorenz63Config(case=2, num_windows=1, T_max=5.0, seed=123, param_bias=0.05)
    
    dataset_cs1 = Lorenz63Dataset(cfg_cs1)
    dataset_cs2 = Lorenz63Dataset(cfg_cs2)
    
    forcing_cs1 = dataset_cs1[0]['forcing_true'].cpu().numpy()
    forcing_cs2 = dataset_cs2[0]['forcing_corrupted'].cpu().numpy()
    time_grid_cs = cfg_cs1.time_grid
    
    plot_forcing_comparison(
        forcing_cs1,
        forcing_cs2,
        time_grid_cs,
        'outputs/figures/forcing_comparison.png'
    )
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"Generated trajectory: {num_steps} steps over {args.duration:.1f} seconds")
    print(f"Observations: {num_obs} sparse samples (every {cfg_long.obs_interval} steps)")
    print(f"Observation ratio: {100.0 * num_obs / num_steps:.1f}%")
    print("\nFigures saved to outputs/figures/:")
    print("  - lorenz63_attractor_3d.png")
    print("  - lorenz63_timeseries.png")
    print("  - observations_visualization.png")
    print("  - forcing_comparison.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
