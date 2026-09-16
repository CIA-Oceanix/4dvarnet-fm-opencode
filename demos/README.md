# Demo Scripts: DA Baseline Visualization

This directory contains 4 standalone demonstration scripts showcasing baseline Data Assimilation (DA) methods on Lorenz-63 case studies.

## Overview

The demos validate the **degradation hypothesis**: classical DA methods (Weak-4DVar, Strong-4DVar, EnKF) perform well under perfect model conditions (CS1) but degrade significantly (3-6x RMSE increase) under model mismatch (CS2: corrupted forcing + biased parameters).

## Quick Start

```bash
# Activate environment
conda activate fdv-monai-proto

# Run all demos (fast mode: 2-second windows)
python demos/demo_lorenz63_dynamics.py
python demos/demo_cs1_baselines.py --duration 2.0
python demos/demo_cs2_baselines.py --duration 2.0
python demos/demo_comparison.py --num-windows 3 --duration 2.0

# For publication-quality results (5-second windows, slower)
python demos/demo_cs1_baselines.py --duration 5.0
python demos/demo_cs2_baselines.py --duration 5.0
python demos/demo_comparison.py --num-windows 10 --duration 5.0
```

## Scripts

### 1. `demo_lorenz63_dynamics.py` (~180 lines)
**Purpose**: Visualize the Lorenz-63 chaotic system and data generation

**Outputs** (4 figures):
- `lorenz63_attractor_3d.png` - 3D phase space visualization
- `lorenz63_timeseries.png` - Time series of X, Y, Z, W_L
- `observations_visualization.png` - Sparse observation sampling
- `forcing_comparison.png` - CS1 vs CS2 forcing corruption

**Runtime**: ~10 seconds

### 2. `demo_cs1_baselines.py` (~200 lines)
**Purpose**: Show good reconstruction with correct model (CS1)

**Outputs**:
- `cs1_reconstruction_all_methods.png` - Comparison of all 3 methods
- `cs1_rmse_table.txt` - RMSE metrics

**Expected Results** (5s window):
- Weak-4DVar RMSE: ~0.3-0.4
- Strong-4DVar RMSE: ~0.2-0.3
- EnKF RMSE: ~0.3-0.4

**Runtime**: 
- 2s window: ~30 seconds
- 5s window: ~90 seconds

### 3. `demo_cs2_baselines.py` (~200 lines)
**Purpose**: Show poor reconstruction with model mismatch (CS2)

**Outputs**:
- `cs2_reconstruction_all_methods.png` - Poor reconstruction visualization
- `cs2_forcing_corruption.png` - Impact of forcing corruption
- `cs2_rmse_table.txt` - RMSE metrics

**Expected Results** (5s window):
- Weak-4DVar RMSE: ~1.5-2.0 (3-5x higher than CS1)
- Strong-4DVar RMSE: ~1.2-1.8 (4-6x higher than CS1)
- EnKF RMSE: ~1.5-2.5 (4-6x higher than CS1)

**Runtime**: 
- 2s window: ~30 seconds
- 5s window: ~90 seconds

### 4. `demo_comparison.py` (~250 lines)
**Purpose**: Direct CS1 vs CS2 comparison with degradation analysis

**Outputs**:
- `cs1_vs_cs2_comparison.png` - Side-by-side reconstruction
- `degradation_barplot.png` - Degradation ratio visualization
- `degradation_summary.txt` - Comprehensive analysis

**Expected Results** (N=10 windows, 5s each):
- Degradation ratios: 3-6x for all methods
- Statistical significance with mean ± std

**Runtime**: 
- 3 windows, 2s: ~3 minutes
- 10 windows, 5s: ~15 minutes

## Case Study Definitions

### CS1: Perfect Model (Baseline)
- **Forcing**: Noise-free (use true W_L)
- **Parameters**: Correct (σ=10, ρ=28, β=8/3)
- **Expected**: Good reconstruction (RMSE < 0.5)

### CS2: Model Mismatch (Degradation Test)
- **Forcing**: Corrupted with Ornstein-Uhlenbeck process
- **Parameters**: Biased by 5% (σ=9.5, ρ=26.6, β=2.8)
- **Expected**: Significant degradation (RMSE 3-6x higher than CS1)

## Command-Line Options

All scripts support:
- `--seed SEED` - Random seed for reproducibility (default: varies)
- `--duration DURATION` - Window duration in seconds (default: varies)

Additional options:
- `demo_comparison.py --num-windows N` - Number of windows to process (default: 10)
- `demo_cs2_baselines.py --bias BIAS` - Parameter bias fraction (default: 0.05)

## Output Directory Structure

```
outputs/
├── figures/          # All PNG figures (300 DPI)
│   ├── lorenz63_attractor_3d.png
│   ├── lorenz63_timeseries.png
│   ├── observations_visualization.png
│   ├── forcing_comparison.png
│   ├── cs1_reconstruction_all_methods.png
│   ├── cs2_reconstruction_all_methods.png
│   ├── cs2_forcing_corruption.png
│   ├── cs1_vs_cs2_comparison.png
│   └── degradation_barplot.png
└── results/          # Text summaries
    ├── cs1_rmse_table.txt
    ├── cs2_rmse_table.txt
    └── degradation_summary.txt
```

## Performance Notes

### Speed Optimizations
The demo scripts use reduced iteration counts for faster execution:
- Weak-4DVar: 80 iterations (vs 150 in production)
- Strong-4DVar: 20 iterations (vs 40 in production)
- EnKF: 30 ensemble members

### Known Limitations
1. **Short windows (2s)**: Show reduced degradation (~2x instead of 3-6x) due to fewer observations
2. **Numerical stability**: Some windows may produce NaN values due to chaotic dynamics
3. **Stochasticity**: Results may vary slightly across runs due to optimization randomness

### Recommended Settings
For **fast exploration** (development):
```bash
--duration 2.0 --num-windows 3
```

For **publication-quality** (validation):
```bash
--duration 5.0 --num-windows 10
```

## Key Findings

From the degradation analysis (`degradation_summary.txt`):

> **KEY FINDING**: All classical DA methods degrade 4-5x under model mismatch 
> (corrupted forcing + biased parameters), validating the need for data-driven 
> approaches like 4DVarNet-FM.

This consistent degradation pattern across all baseline methods motivates the development of learned DA approaches that can adapt to model uncertainties.

## Dependencies

- PyTorch 2.4.1+
- NumPy 1.26.4+
- Matplotlib 3.9.2+

See `requirements.txt` for full dependencies.

## Troubleshooting

### NaN values in results
- Try reducing window duration: `--duration 2.0`
- Use fewer windows: `--num-windows 3`
- Check for numerical overflow in chaotic regions

### Slow execution
- Reduce window duration: `--duration 2.0`
- Reduce number of windows: `--num-windows 3`
- The optimization is CPU-bound and may take several minutes

### Import errors
- Ensure conda environment is activated: `conda activate fdv-monai-proto`
- Verify all dependencies are installed: `pip install -r requirements.txt`

## References

- Lorenz, E. N. (1963). Deterministic nonperiodic flow.
- Le Dimet, F.-X., & Talagrand, O. (1986). Variational algorithms for analysis and assimilation.
- Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model.

## Author

Agent B - Visualization Specialist
Date: June 25, 2026
