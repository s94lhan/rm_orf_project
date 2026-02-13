
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Tuple, Optional


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    c = spearmanr(a, b).correlation
    return float(c) if c == c else np.nan


def point_metrics(tau_hat: np.ndarray, tau_true: np.ndarray) -> Dict[str, float]:
    err = tau_hat - tau_true
    return {
        "bias": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "abs_err_p90": float(np.quantile(np.abs(err), 0.90))
    }


def gates_stats(tau_hat: np.ndarray, tau_true: np.ndarray, K: int = 5):
    qs = np.quantile(tau_hat, np.linspace(0, 1, K + 1))
    qs = np.unique(qs)
    if len(qs) < K + 1:
        jitter = 1e-8 * np.random.randn(*tau_hat.shape)
        qs = np.quantile(tau_hat + jitter, np.linspace(0, 1, K + 1))
    groups = np.digitize(tau_hat, qs[1:-1], right=True)  # 0..K-1

    est_means, true_means = [], []
    for k in range(K):
        m = groups == k
        est_means.append(np.mean(tau_hat[m]))
        true_means.append(np.mean(tau_true[m]))
    est_means = np.array(est_means)
    true_means = np.array(true_means)
    rmse_g = float(np.sqrt(np.mean((est_means - true_means) ** 2)))
    return rmse_g, est_means, true_means, groups


def clan_diffs(X5: np.ndarray, tau_hat: np.ndarray, top_q: float = 0.2,
               bottom_q: float = 0.2, s: int = 5) -> Dict[str, float]:
    n = X5.shape[0]
    order = np.argsort(tau_hat)
    k_top = max(1, int(np.floor(n * top_q)))
    k_bot = max(1, int(np.floor(n * bottom_q)))
    bot = order[:k_bot]
    top = order[-k_top:]
    diffs = np.mean(X5[top, :s], axis=0) - np.mean(X5[bot, :s], axis=0)
    return {f"clan_diff_x{j + 1}": float(diffs[j]) for j in range(s)}


def top_bottom_10(tau_hat: np.ndarray, tau_true: np.ndarray) -> Dict[str, float]:
    n = len(tau_hat)
    k = max(1, int(np.floor(0.1 * n)))
    order = np.argsort(tau_hat)
    bot = order[:k]
    top = order[-k:]
    true_top = float(np.mean(tau_true[top]))
    true_bot = float(np.mean(tau_true[bot]))
    est_top = float(np.mean(tau_hat[top]))
    est_bot = float(np.mean(tau_hat[bot]))
    return {
        "top10_true": true_top, "bot10_true": true_bot,
        "top10_est": est_top, "bot10_est": est_bot,
        "top10_err": float(est_top - true_top),
        "bot10_err": float(est_bot - true_bot),
        "topbot_gap_true": float(true_top - true_bot),
        "topbot_gap_est": float(est_top - est_bot),
        "topbot_gap_err": float((est_top - est_bot) - (true_top - true_bot)),
    }


def ci_pointwise_coverage(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    se_hat: Optional[np.ndarray],
    K: int = 5,
    z: float = 1.96,
) -> Dict[str, float]:

    if se_hat is None:
        return {
            "pointwise_cover_rate": np.nan,
            "pointwise_ci_len": np.nan,
            "pointwise_se_cv": np.nan,
            "pointwise_cover_q1": np.nan,
            "pointwise_cover_q2": np.nan,
            "pointwise_cover_q3": np.nan,
            "pointwise_cover_q4": np.nan,
            "pointwise_cover_q5": np.nan,
        }

    tau_hat = np.asarray(tau_hat).reshape(-1)
    tau_true = np.asarray(tau_true).reshape(-1)
    se_hat = np.asarray(se_hat).reshape(-1)

    if not (tau_hat.shape[0] == tau_true.shape[0] == se_hat.shape[0]):
        raise ValueError("tau_hat, tau_true, se_hat must have the same length")

    lo = tau_hat - z * se_hat
    hi = tau_hat + z * se_hat
    cover_i = (lo <= tau_true) & (tau_true <= hi)

    cover_rate = float(np.mean(cover_i))
    ci_len = float(np.mean(hi - lo))
    se_mean = float(np.mean(se_hat))
    se_cv = float(np.std(se_hat, ddof=1) / (se_mean + 1e-12)) if se_hat.size > 1 else np.nan

    order = np.argsort(tau_true)
    idx_groups = np.array_split(order, K)
    group_coverages = []
    for gidx in idx_groups:
        if gidx.size == 0:
            group_coverages.append(np.nan)
        else:
            group_coverages.append(float(np.mean(cover_i[gidx])))

    while len(group_coverages) < 5:
        group_coverages.append(np.nan)

    return {
        "pointwise_cover_rate": cover_rate,
        "pointwise_ci_len": ci_len,
        "pointwise_se_cv": se_cv,
        "pointwise_cover_q1": group_coverages[0],
        "pointwise_cover_q2": group_coverages[1],
        "pointwise_cover_q3": group_coverages[2],
        "pointwise_cover_q4": group_coverages[3],
        "pointwise_cover_q5": group_coverages[4],
    }


import numpy as np

def clan_nonlinear_diffs(X, tau_hat, top_q=0.2, bottom_q=0.2):
    n = len(tau_hat)
    order = np.argsort(tau_hat)

    k_top = int(np.floor(top_q * n))
    k_bot = int(np.floor(bottom_q * n))

    top_idx = order[-k_top:]
    bot_idx = order[:k_bot]

    x2 = X[:, 1]   # X2 is the 2nd column

    sin_x2 = np.sin(np.pi * x2)
    abs_x2 = np.abs(x2 - 0.5)

    return {
        "clan_diff_sin_pi_x2": sin_x2[top_idx].mean() - sin_x2[bot_idx].mean(),
        "clan_diff_abs_x2_m05": abs_x2[top_idx].mean() - abs_x2[bot_idx].mean()
    }
