
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Utilities
def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1))


def _bootstrap_indices(T: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, T, size=T, endpoint=False)


def _loss_transform(series: pd.Series, transform: str, nominal: float) -> pd.Series:
    """
    Ensure "smaller is better" loss.
    """
    if transform == "identity":
        return series.astype(float)
    if transform == "neg":
        return (-series.astype(float))
    if transform == "abs":
        return series.astype(float).abs()
    if transform == "abs_dev_from_nominal":
        return (series.astype(float) - nominal).abs()
    if transform == "sq_dev_from_nominal":
        return (series.astype(float) - nominal) ** 2
    raise ValueError(f"Unknown loss_transform: {transform}")


@dataclass
class MCSResult:
    dgp: str
    n: int
    metric: str
    loss_transform: str
    alpha: float
    B: int
    included: List[str]
    elimination_order: List[str]
    elimination_pvalues: List[float]
    T_used: int


# Core MCS
def mcs(
    losses: pd.DataFrame,
    alpha: float = 0.05,
    B: int = 2000,
    seed: int = 123,
) -> Tuple[List[str], List[str], List[float]]:

    if losses.shape[1] < 2:
        cols = list(losses.columns)
        return cols, [], []

    rng = np.random.default_rng(seed)

    # Drop rows with any NaNs to keep aligned series
    L = losses.copy()
    L = L.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    T = int(L.shape[0])
    if T < 10:
        # Too few reps to do meaningful MCS; return everything.
        return list(L.columns), [], []

    methods = list(L.columns)
    elimination_order: List[str] = []
    elimination_pvalues: List[float] = []

    # Precompute loss matrix
    Lmat_full = L.to_numpy(dtype=float)  # (T, M)

    # Iteratively eliminate
    alive = list(range(len(methods)))  # indices into methods / columns

    while len(alive) > 1:
        # Current alive loss matrix
        Lmat = Lmat_full[:, alive]  # (T, m)
        m = Lmat.shape[1]

        # Pairwise loss differentials: d_ij,t = L_i,t - L_j,t
        # We'll compute mean and standard error for each (i, j)
        dbar = np.zeros((m, m), dtype=float)
        se = np.zeros((m, m), dtype=float)

        for i in range(m):
            for j in range(m):
                if i == j:
                    dbar[i, j] = 0.0
                    se[i, j] = np.nan
                else:
                    dij = Lmat[:, i] - Lmat[:, j]
                    dbar[i, j] = float(np.mean(dij))
                    s = _safe_std(dij)
                    se[i, j] = s / math.sqrt(T) if np.isfinite(s) and s > 0 else np.nan

        # Observed t_ij and Tmax
        tmat = np.full((m, m), np.nan, dtype=float)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                if np.isfinite(se[i, j]) and se[i, j] > 0:
                    tmat[i, j] = dbar[i, j] / se[i, j]

        # For each method i, t_i = max_j t_ij (how much i is worse than others)
        t_i = np.nanmax(tmat, axis=1)
        Tmax_obs = float(np.nanmax(t_i))

        # Bootstrap distribution of Tmax under null (recentering)
        Tmax_star = np.empty(B, dtype=float)
        for b in range(B):
            idx = _bootstrap_indices(T, rng)
            Lb = Lmat[idx, :]  # (T, m)

            # Recentered mean differentials: mean(dij*) - mean(dij)
            tmat_b = np.full((m, m), np.nan, dtype=float)
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    dij_b = Lb[:, i] - Lb[:, j]
                    dbar_b = float(np.mean(dij_b) - dbar[i, j])
                    if np.isfinite(se[i, j]) and se[i, j] > 0:
                        tmat_b[i, j] = dbar_b / se[i, j]
            t_i_b = np.nanmax(tmat_b, axis=1)
            Tmax_star[b] = float(np.nanmax(t_i_b))

        # p-value for EPA (Equal Predictive Ability) null
        pval = float(np.mean(Tmax_star >= Tmax_obs))

        if pval >= alpha:
            break  # fail to reject EPA: current set is the MCS

        # Eliminate the "worst" method: the one with largest t_i
        worst_local = int(np.nanargmax(t_i))
        worst_global_idx = alive[worst_local]
        elimination_order.append(methods[worst_global_idx])
        elimination_pvalues.append(pval)

        # Remove it
        alive.pop(worst_local)

    included = [methods[i] for i in alive]
    return included, elimination_order, elimination_pvalues


# Runner on your outputs
def run_mcs_from_metrics_long(
    metrics_long: pd.DataFrame,
    n_ref: int,
    methods: List[str],
    metric: str,
    loss_transform: str,
    nominal: float,
    alpha: float,
    B: int,
    seed: int,
) -> List[MCSResult]:
    out: List[MCSResult] = []
    df = metrics_long.copy()

    # Basic checks
    need_cols = {"dgp", "n", "rep", "method", metric}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"metrics_long missing columns: {missing}")

    df = df[df["n"] == n_ref].copy()
    df = df[df["method"].isin(methods)].copy()

    # Ensure numeric
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    for dgp, ddf in df.groupby("dgp"):
        # Pivot to (rep x method)
        pv = ddf.pivot_table(index="rep", columns="method", values=metric, aggfunc="mean")
        pv = pv.reindex(columns=methods)  # keep order
        pv = pv.dropna(axis=0, how="any")  # aligned reps only

        # Apply loss transform to make "smaller is better"
        L = pv.apply(lambda s: _loss_transform(s, loss_transform, nominal))

        included, elim_order, elim_pvals = mcs(L, alpha=alpha, B=B, seed=seed)

        out.append(
            MCSResult(
                dgp=str(dgp),
                n=int(n_ref),
                metric=metric,
                loss_transform=loss_transform,
                alpha=float(alpha),
                B=int(B),
                included=included,
                elimination_order=elim_order,
                elimination_pvalues=elim_pvals,
                T_used=int(L.shape[0]),
            )
        )
    return out


def results_to_dataframe(res: List[MCSResult]) -> pd.DataFrame:
    rows = []
    for r in res:
        rows.append(
            {
                "dgp": r.dgp,
                "n": r.n,
                "metric": r.metric,
                "loss_transform": r.loss_transform,
                "alpha": r.alpha,
                "B": r.B,
                "T_used": r.T_used,
                "mcs_included": ",".join(r.included),
                "elim_order": ",".join(r.elimination_order) if r.elimination_order else "",
                "elim_pvalues": ",".join([f"{p:.4g}" for p in r.elimination_pvalues]) if r.elimination_pvalues else "",
                "n_methods_included": len(r.included),
            }
        )
    return pd.DataFrame(rows).sort_values(["dgp"])


def main():
    parser = argparse.ArgumentParser(description="Run MCS tests on metrics_long.csv")
    parser.add_argument("--in_csv", type=str, required=True, help="Path to metrics_long.csv")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV for MCS results")
    parser.add_argument("--n", type=int, default=1000, help="Sample size n to filter (default 1000)")
    parser.add_argument("--methods", type=str, nargs="+", default=["ORF", "CF", "DML-CF"])
    parser.add_argument("--metric", type=str, default="rmse", help="Metric column in metrics_long (default rmse)")
    parser.add_argument(
        "--loss_transform",
        type=str,
        default="identity",
        choices=["identity", "neg", "abs", "abs_dev_from_nominal", "sq_dev_from_nominal"],
        help="Transform metric into a loss where smaller is better.",
    )
    parser.add_argument(
        "--nominal",
        type=float,
        default=0.95,
        help="Nominal value used by *_dev_from_nominal transforms (default 0.95).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="MCS alpha (default 0.05)")
    parser.add_argument("--B", type=int, default=2000, help="Bootstrap draws (default 2000)")
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    res = run_mcs_from_metrics_long(
        metrics_long=df,
        n_ref=args.n,
        methods=args.methods,
        metric=args.metric,
        loss_transform=args.loss_transform,
        nominal=args.nominal,
        alpha=args.alpha,
        B=args.B,
        seed=args.seed,
    )
    out_df = results_to_dataframe(res)
    out_df.to_csv(args.out_csv, index=False)

    print("[DONE] MCS results saved to:", args.out_csv)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
