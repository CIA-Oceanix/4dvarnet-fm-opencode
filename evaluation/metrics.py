import numpy as np


def rmse(analysis: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((analysis - truth) ** 2, axis=0))


def spread(ensemble_variance: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(ensemble_variance, axis=0))


def crps(ensemble: np.ndarray, truth: np.ndarray) -> float:
    N, T, D = ensemble.shape
    scores = np.zeros(D)
    truth_expanded = truth[np.newaxis, :, :]
    for d in range(D):
        e = ensemble[:, :, d]
        t = truth[:, d]
        abs_diff = np.abs(e[np.newaxis, :, :] - e[:, np.newaxis, :])
        pairwise = np.mean(abs_diff, axis=(0, 1))
        abs_err = np.mean(np.abs(e - t[np.newaxis, :]), axis=0)
        scores[d] = np.mean(pairwise - abs_err)
    return scores


def print_metrics_table(results: dict, case_name: str):
    print(f"\n{'=' * 70}")
    print(f"  {case_name}")
    print(f"{'=' * 70}")
    print(f"{'Method':<20} {'RMSE X':<12} {'RMSE Y':<12} {'RMSE Z':<12}")
    print(f"{'-' * 56}")
    for name, res in results.items():
        r = res.rmse
        print(f"{name:<20} {r[0]:<12.4f} {r[1]:<12.4f} {r[2]:<12.4f}")
    print(f"{'=' * 70}")
