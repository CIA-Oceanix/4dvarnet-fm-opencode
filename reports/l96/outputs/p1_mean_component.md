# `Psi_mean` alone vs the full CFM — S0, group `all_obs`

200 windows, mean ± std across windows. Same conventions as `m4_per_window_metrics.md`: RMSE is the repo's `mean_d sqrt(mse_d)`, `pooled` is `evaluate_estimates`'s.

CRPS of a point forecast is the absolute error, so that column is MAE and is NOT comparable to the full models' ensemble CRPS. `full RMSE` is the ens30 ensemble-mean RMSE of the same run, for the mean-vs-full comparison; `draws` is the across-draw dispersion of `m` itself (0 = deterministic in y).

| model | RMSE | pooled | MSE | CRPS (=MAE) | full RMSE | mean−full | draws |
|---|---|---|---|---|---|---|---|
| VanillaCFM S+ | 0.4726 ± 0.0849 | 0.4932 | 0.2547 ± 0.0975 | 0.3244 ± 0.0584 | 0.4124 | +14.6% | 0.1227 |
| VanillaCFM M | 0.4101 ± 0.0689 | 0.4251 | 0.1992 ± 0.0734 | 0.2637 ± 0.0401 | 0.3445 | +19.0% | 0.0919 |
| VanillaCFM L | 0.4183 ± 0.0672 | 0.4333 | 0.2080 ± 0.0743 | 0.2654 ± 0.0384 | 0.3563 | +17.4% | 0.0739 |
| PredictStateCFM S+ | 0.4481 ± 0.0740 | 0.4632 | 0.2341 ± 0.0828 | 0.2922 ± 0.0440 | 0.4124 | +8.7% | 0.0732 |
| PredictStateCFM M | 0.4081 ± 0.0669 | 0.4229 | 0.1973 ± 0.0713 | 0.2582 ± 0.0381 | 0.3584 | +13.9% | 0.0546 |
| PredictStateCFM L (lr3e-4) | 0.4072 ± 0.0658 | 0.4222 | 0.1972 ± 0.0706 | 0.2555 ± 0.0368 | 0.3508 | +16.1% | 0.0497 |
| FDV1CFM S+ | 0.3732 ± 0.0767 | 0.3924 | 0.1666 ± 0.0729 | 0.2513 ± 0.0494 | 0.3482 | +7.2% | 0.0826 |
| FDV1CFM M | 0.3635 ± 0.0586 | 0.3795 | 0.1572 ± 0.0584 | 0.2379 ± 0.0328 | 0.3887 | -6.5% | 0.0639 |
| FDV1CFM S+ (+aug) | 0.4878 ± 0.0825 | 0.5069 | 0.2776 ± 0.0964 | 0.3347 ± 0.0556 | 0.4766 | +2.4% | 0.1056 |
| FDV1CFM M (+aug) | 0.4478 ± 0.0560 | 0.4604 | 0.2312 ± 0.0660 | 0.3088 ± 0.0305 | 0.4747 | -5.7% | 0.0859 |
