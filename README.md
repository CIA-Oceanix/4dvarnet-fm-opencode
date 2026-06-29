# 4DVarNet-FM — Physics-Informed Neural Ensemble DA via Conditional Flow Matching

Implementation of the 4DVarNet-FM framework for stochastic Lorenz-63 data assimilation, combining conditional flow matching with a Tweedie-inspired decomposition for ensemble-based state estimation.

## Reference

This implementation is based on the manuscript:

> **4DVarNet-FM: Physics-Informed Neural Ensemble Data Assimilation via Conditional Flow Matching**  
> R. Fablet, W. de Tinguy, M. Cuzol, B. Chapron  
> *Preprint, 2025*

See [`notes_4dvarnet_fm.pdf`](./notes_4dvarnet_fm.pdf) for the full manuscript.

The baseline DA methods (Weak-4DVar, Strong-4DVar, EnKF) are adapted from the reference Colab notebook [`notebook.ipynb`](./notebook.ipynb).

## Project Structure

```
├── data/               # Lorenz-63 SDE dataset generation
│   ├── lorenz63.py     # Trajectory generation with train/val/test splits
│   └── dataloader.py   # Flow-matching batch collation
├── models/             # Neural DA components
│   ├── interpolant.py  # Linear interpolant (ατ=1-τ, βτ=τ)
│   ├── unet.py         # 1D U-Net backbone
│   ├── residual.py     # IterativeUpdateCell (Rᴺᴳ) & MeanEstimatorCell (Ψ̄θ̄)
│   └── solver.py       # TweedieSolver with Eq. 14 decomposition
├── training/           # Two-stage training pipeline
│   ├── losses.py       # State MSE + gradient loss
│   ├── stage1.py       # Train Gaussian mean estimator Ψ̄θ̄
│   ├── stage2.py       # Train non-Gaussian residual Rᴺᴳ (freeze Ψ̄θ̄)
│   ├── lightning_module.py  # PyTorch Lightning wrapper
│   └── pipeline.py     # pl.Trainer orchestration
├── evaluation/         # Baselines and metrics
│   ├── baselines.py    # Weak-4DVar, Strong-4DVar, EnKF
│   ├── metrics.py      # RMSE, spread, CRPS
│   ├── experiment.py   # Experiment orchestration
│   └── run.py          # Baseline runner with caching
├── conf/               # Hydra structured config schemas
│   └── schema.py       # DataConfig, ModelConfig, TrainingConfig
├── config/             # YAML configuration presets
│   ├── lorenz63_default.yaml
│   ├── baselines/      # DWS presets (dws20 … dws500)
│   └── experiment/     # FM experiment configs (A1, B1, C1, C4, D1)
├── reports/            # Report generation scripts and outputs (tracked)
│   ├── generate_report.py
│   ├── generate_baseline_report.py
│   ├── generate_synthesis.py
│   └── outputs/        # Synthesis PDFs (versioned)
├── batch/              # SLURM batch and shell scripts
│   ├── *.sbatch        # run_baselines, run_enkf_inflation, …
│   ├── launch.sh
│   └── run_with_scheduler.sh
├── notebooks/          # Jupyter notebooks
│   └── demo_baselines.ipynb  # CS1 vs CS2 baseline comparison
├── eval_baselines.py   # Hydra entry point for baseline evaluation
├── train.py            # Hydra entry point for FM training
├── run_experiment.py   # End-to-end entry point
├── run_experiments.py  # Multi-experiment orchestrator
└── notes_4dvarnet_fm.pdf  # Manuscript PDF
```

## Quick Start

```bash
pip install torch numpy matplotlib

# Generate data and run baselines
python run_experiment.py

# Generate baseline synthesis report
python reports/generate_baseline_report.py \
    --json experiments/baselines_dws50_inf1.2.json \
    --trajs experiments/baselines_trajectories_dws50_inf1.2.npz \
    --output reports/outputs/synthesis_dws50_inf12.pdf

# Or explore interactively
jupyter notebook notebooks/demo_baselines.ipynb
```

## Case Studies

- **CS1** — Noise-free forcings, correct parameters → expected good reconstruction
- **CS2** — OU-corrupted forcings, biased parameters → expected degradation

The key validation criterion is that CS1 RMSE << CS2 RMSE across all methods.

## Citation

If you use this code, please cite the original manuscript:

```bibtex
@article{fablet2025dvarnetfm,
  title={4DVarNet-FM: Physics-Informed Neural Ensemble Data Assimilation via Conditional Flow Matching},
  author={Fablet, R. and de Tinguy, W. and Cuzol, M. and Chapron, B.},
  year={2025}
}
```

## License

See the original manuscript for licensing details.
