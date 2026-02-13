
import numpy as np
from scipy.special import expit as sigmoid
from typing import Tuple
from config import DGPConfig


# DGP
def mu0(X5: np.ndarray) -> np.ndarray:
    """μ₀(X)"""
    X1, X2, X3, X4, X5 = X5[:, 0], X5[:, 1], X5[:, 2], X5[:, 3], X5[:, 4]
    return 2.0 * X1 + 0.5 * (X2 ** 2) - np.sin(2 * np.pi * X3) + 0.2 * X4 * X5


def tau_strong(X5: np.ndarray) -> np.ndarray:
    """strong τ(X)"""
    X1, X2 = X5[:, 0], X5[:, 1]
    return 1.0 + 1.0 * X1 + np.sin(np.pi * X2)


def tau_weak(X5: np.ndarray) -> np.ndarray:
    """weak τ(X)"""
    X1 = X5[:, 0]
    return 1.0 + 0.2 * X1


def tau_threshold(X5: np.ndarray) -> np.ndarray:
    """threshold τ(X)"""
    X1, X2 = X5[:, 0], X5[:, 1]
    return 0.5 + (X1 > 0.5).astype(float) + 0.5 * (X2 > 0.7).astype(float)


def g_high_alignment(X5: np.ndarray) -> np.ndarray:
    """high alignment g(X)"""
    X1, X2, X3 = X5[:, 0], X5[:, 1], X5[:, 2]
    return -0.5 + 1.5 * X1 - 1.5 * X2 + 1.0 * X3


def g_low_alignment_mild(X5: np.ndarray) -> np.ndarray:
    """low alignment g(X)"""
    X4, X5 = X5[:, 3], X5[:, 4]
    return -0.5 + 2.0 * X4 - 2.0 * X5


def g_low_alignment_hard(X5: np.ndarray) -> np.ndarray:
    """low alignment g(X)"""
    X4, X5 = X5[:, 3], X5[:, 4]
    return -2.0 + 5.0 * X4 - 5.0 * X5


def tau_from_type(X5: np.ndarray, tau_type: str) -> np.ndarray:
    if tau_type == "strong":
        return tau_strong(X5)
    if tau_type == "weak":
        return tau_weak(X5)
    if tau_type == "threshold":
        return tau_threshold(X5)
    raise ValueError(f"未知的tau_type: {tau_type}")


def g_from_alignment(cfg: DGPConfig, X5: np.ndarray) -> np.ndarray:
    if cfg.alignment == "high":
        return g_high_alignment(X5)
    if cfg.alignment == "low":
        return g_low_alignment_mild(X5) if cfg.name == "DGP2" else g_low_alignment_hard(X5)
    raise ValueError("RCT没有g(X)")


def clip_prob(e: np.ndarray, a: float, b: float) -> np.ndarray:
    """Truncation probability value"""
    return np.minimum(np.maximum(e, a), b)


def gen_data(cfg: DGPConfig, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, cfg.p))  # X_j ~ U(0,1)

    X5 = X[:, :5]
    tau_true = tau_from_type(X5, cfg.tau_type)
    mu = mu0(X5)

    if cfg.is_rct:
        e = np.full(n, 0.5)
    else:
        g = g_from_alignment(cfg, X5)
        e = sigmoid(g)
        e = clip_prob(e, cfg.clip_a, cfg.clip_b)

    T = rng.binomial(1, e, size=n).astype(int)
    eps = rng.normal(0.0, cfg.sigma, size=n)
    Y = mu + tau_true * T + eps

    return X, T, Y, tau_true, e