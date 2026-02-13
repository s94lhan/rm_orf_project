
import os
import warnings
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from typing import Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

#Figure 5-2 CATE Calibration Across Different DGPs
def plot_calibration_curve_one(dgp: str, n: int, pred_dir: str,
                               outdir: str, K: int = 20, rep: int = 1) -> None:

    ensure_dir(outdir)

    fn = os.path.join(pred_dir, f"pred_{dgp}_n{n}_rep{rep}.npz")
    if not os.path.exists(fn):
        warnings.warn(f"Calibration plot skipped: missing saved predictions file: {fn}")
        return

    pack = np.load(fn)

    if "tau_true_test" in pack.files:
        tau_true = np.asarray(pack["tau_true_test"]).reshape(-1)
    else:
        tau_true = np.asarray(pack["tau_true"]).reshape(-1)

    if "n_test" in pack.files:
        n_test_saved = int(pack["n_test"])
        if tau_true.shape[0] != n_test_saved:
            warnings.warn(
                f"Calibration plot skipped: tau_true length {tau_true.shape[0]} != saved n_test {n_test_saved} in {fn}"
            )
            return

    methods = []
    for m in ["ORF", "CF", "DML-CF"]:
        key = f"tau_hat_{m}_test"

        if key in pack.files:
            methods.append(m)

    if not methods:
        warnings.warn(f"Calibration plot skipped: no method predictions in {fn}")
        return

    curves = {}
    for m in methods:
        tau_hat = np.asarray(pack[f"tau_hat_{m}_test"]).reshape(-1)


        # Binning: Binn the data by the quantile point tau_hat, then calculate the mean within each bin for (tau_hat, tau_true).
        qs = np.quantile(tau_hat, np.linspace(0, 1, K + 1))
        qs = np.unique(qs)
        if len(qs) < K + 1:
            jitter = 1e-8 * np.random.randn(*tau_hat.shape)
            qs = np.quantile(tau_hat + jitter, np.linspace(0, 1, K + 1))
        g = np.digitize(tau_hat, qs[1:-1], right=True)

        x_bin, y_bin = [], []
        for k in range(K):
            msk = g == k
            if np.sum(msk) == 0:
                continue
            x_bin.append(float(np.mean(tau_hat[msk])))
            y_bin.append(float(np.mean(tau_true[msk])))
        curves[m] = (np.array(x_bin), np.array(y_bin))

    plt.figure()

    all_vals = [tau_true]
    for m in methods:
        all_vals.append(np.asarray(pack[f"tau_hat_{m}_test"]).reshape(-1))
    lo = float(np.min(np.concatenate(all_vals)))
    hi = float(np.max(np.concatenate(all_vals)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="Perfect calibration")

    colors = {"ORF": "C0", "CF": "C1", "DML-CF": "C2"}
    for m in methods:
        x, y = curves[m]
        plt.plot(x, y, marker="o", label=m, color=colors.get(m, None))

    plt.xlabel("Predicted value (bin mean)")
    plt.ylabel("True value (bin mean)")
    plt.title(f"Calibration - {dgp}, n={n}, rep={rep}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig1_calibration_{dgp}_n{n}_rep{rep}.png"), dpi=200)
    plt.close()

#Figure 5-9: RMSE Increase under Nuisance Misspecification across DGPs (n=1000)
def plot_misspec_robustness(metrics_long: pd.DataFrame, outdir: str,
                            dgp_ref: str = "DGP0", n_ref: int = 1000) -> None:

    ensure_dir(outdir)
    df = metrics_long[(metrics_long["dgp"] == dgp_ref) & (metrics_long["n"] == n_ref)]
    methods = ["ORF", "ORF_misspec", "DML-CF", "DML-CF_misspec", "CF"]
    df = df[df["method"].isin(methods)]

    if df.empty:
        warnings.warn("Misspec robustness plot skipped (no data). Run with --diag_misspec.")
        return

    # pivot per rep
    pv = df.pivot_table(index="rep", columns="method", values="rmse")
    bars = {}
    for base, ms in [("ORF", "ORF_misspec"), ("DML-CF", "DML-CF_misspec")]:
        if base in pv.columns and ms in pv.columns:
            ratio = (pv[ms] - pv[base]) / (pv[base] + 1e-12)
            bars[base] = float(np.nanmean(ratio.values))

    plt.figure()
    xs = list(bars.keys())
    ys = [bars[k] for k in xs]
    plt.bar(xs, ys)
    plt.ylabel("RMSE increase ratio under misspec")
    plt.title(f"Figure 4: Robustness to nuisance misspec - {dgp_ref}, n={n_ref}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig4_misspec_robustness_{dgp_ref}_n{n_ref}.png"), dpi=200)
    plt.close()


# Figure 5-1: Bias, RMSE, tail RMSE(p90), MAE across DGPs (n=1000)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"[Figure 5-1] 找不到 {label} 对应列。候选={candidates}，现有列={list(df.columns)}")


def plot_fig5_1_accuracy_panels(
    summary: pd.DataFrame,
    outdir: str,
    n_ref: int = 1000,
    dgps_order: list[str] | None = None,
    methods_order: list[str] | None = None,
) -> str:

    #Figure 5-1 (a)(b)(c)(d)
    _ensure_dir(outdir)

    if dgps_order is None:
        dgps_order = ["RCT", "DGP0", "DGP4", "DGP7", "DGP3"]
    if methods_order is None:
        methods_order = ["CF", "DML-CF", "ORF"]

    df = summary.copy()
    if "n" not in df.columns or "dgp" not in df.columns or "method" not in df.columns:
        raise KeyError("[Figure 5-1] summary 必须包含列：dgp, n, method")

    df = df[df["n"] == n_ref].copy()
    if df.empty:
        raise ValueError(f"[Figure 5-1] summary 中找不到 n={n_ref} 的数据。")


    bias_col = _pick_col(df, ["bias_mean", "bias", "mean_bias"], "Bias(mean)")
    rmse_col = _pick_col(df, ["rmse_mean", "rmse", "RMSE_mean"], "RMSE(mean)")
    p90_col = _pick_col(df, ["rmse_p90", "rmse90", "rmse_p90_mean", "tail_rmse_p90"], "RMSE(p90)")
    mae_col  = _pick_col(df, ["mae_mean", "mae", "MAE_mean"], "MAE(mean)")

    metrics = [
        ("(a) Bias (mean)", bias_col, "Bias (mean)"),
        ("(b) RMSE (mean)", rmse_col, "RMSE (mean)"),
        ("(c) RMSE (p90)",  p90_col,  "RMSE (p90)"),
        ("(d) MAE (mean)",  mae_col,  "MAE (mean)"),
    ]

    dgps = [d for d in dgps_order if d in set(df["dgp"].astype(str))]
    if not dgps:
        raise ValueError(f"[Figure 5-1] summary 里没找到指定 dgps_order={dgps_order} 的任何 DGP。")

    methods = [m for m in methods_order if m in set(df["method"].astype(str))]
    if not methods:
        raise ValueError(f"[Figure 5-1] summary 里没找到指定 methods_order={methods_order} 的任何 method。")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    x = np.arange(len(dgps))
    width = 0.8 / max(1, len(methods))

    for ax, (panel_title, col, ylab) in zip(axes, metrics):
        for i, m in enumerate(methods):
            ys = []
            for dgp in dgps:
                sub = df[(df["dgp"].astype(str) == dgp) & (df["method"].astype(str) == m)]
                if len(sub) == 0:
                    ys.append(np.nan)
                else:
                    ys.append(float(pd.to_numeric(sub[col], errors="coerce").iloc[0]))
            xpos = x + (i - (len(methods) - 1) / 2) * width
            bars = ax.bar(xpos, ys, width=width, label=m)


            for b in bars:
                h = b.get_height()
                if np.isfinite(h):
                    ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}",
                            ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(dgps)
        ax.set_title(panel_title)
        ax.set_ylabel(ylab)


        if col == bias_col:
            ax.axhline(0.0, linewidth=1)

        ax.legend(fontsize=8)

    fig.suptitle(f"Figure 5-1 Bias, RMSE, tail RMSE (p90), and MAE across DGPs (n = {n_ref})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    save_path = os.path.join(outdir, f"fig5_1_accuracy_panels_n{n_ref}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_fig5_1_from_merged_csv(
    summary_csv: str = "summary_all.csv",
    outdir: str = "plots_report",
    n_ref: int = 1000
) -> str:

    summary = pd.read_csv(summary_csv)
    path = plot_fig5_1_accuracy_panels(summary, outdir=outdir, n_ref=n_ref)
    print("[DONE] Figure 5-1 saved to:", path)
    return path



# Figure 5-3: CLAN Decomposition of Linear and Nonlinear Treatment Effect Heterogeneity

def plot_fig5_3_clan_decomposition(
    summary: pd.DataFrame,
    outdir: str,
    n_ref: int = 1000,
    dgps_order: Optional[list] = None,
    methods_order: Optional[list] = None,
) -> str:

    ensure_dir(outdir)

    if dgps_order is None:
        dgps_order = ["DGP0", "DGP3", "DGP4", "DGP7"]
    if methods_order is None:
        methods_order = ["CF", "DML-CF", "ORF"]


    need_cols = {"dgp", "n", "method", "clan_diff_x1_mean", "clan_diff_abs_x2_m05_mean"}
    miss = need_cols - set(summary.columns)
    if miss:
        raise KeyError(f"[Figure 5-3] summary 缺少必要列: {miss}. 现有列: {list(summary.columns)}")

    df = summary[summary["n"] == n_ref].copy()
    if df.empty:
        raise ValueError(f"[Figure 5-3] summary 中找不到 n={n_ref} 的数据。")

    dgps = [d for d in dgps_order if d in set(df["dgp"].astype(str))]
    methods = [m for m in methods_order if m in set(df["method"].astype(str))]
    if not dgps:
        raise ValueError(f"[Figure 5-3] 未找到指定 DGP: {dgps_order}")
    if not methods:
        raise ValueError(f"[Figure 5-3] 未找到指定方法: {methods_order}")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [
        ("Linear heterogeneity (CLAN Δx1)", "clan_diff_x1_mean", "CLAN Δx1"),
        ("Nonlinear heterogeneity (CLAN Δ|x2 − 0.5|)", "clan_diff_abs_x2_m05_mean", "CLAN Δ|x2 − 0.5|"),
    ]

    x = np.arange(len(dgps))
    width = 0.8 / max(1, len(methods))

    for ax, (title, col, ylab) in zip(axes, panels):
        for i, m in enumerate(methods):
            ys = []
            for dgp in dgps:
                sub = df[(df["dgp"].astype(str) == dgp) & (df["method"].astype(str) == m)]
                ys.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
            xpos = x + (i - (len(methods) - 1) / 2) * width
            bars = ax.bar(xpos, ys, width=width, label=m)


            for b in bars:
                h = b.get_height()
                if np.isfinite(h):
                    ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}",
                            ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(dgps)
        ax.set_title(title)
        ax.set_xlabel("DGP")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)

    fig.suptitle("Figure 5-3 CLAN Decomposition of Linear and Nonlinear Treatment Effect Heterogeneity",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    save_path = os.path.join(outdir, f"fig5_3_clan_decomposition_n{n_ref}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_fig5_3_from_merged_csv(
    summary_csv: str = "summary_all.csv",
    outdir: str = "plots_report",
    n_ref: int = 1000,
) -> str:

    summary = pd.read_csv(summary_csv)
    path = plot_fig5_3_clan_decomposition(summary, outdir=outdir, n_ref=n_ref)
    print("[DONE] Figure 5-3 saved to:", path)
    return path


def plot_fig5_5_inference_tradeoffs(
    summary: pd.DataFrame,
    outdir: str,
    n_ref: int = 1000,
    dgps_order: Optional[list] = None,
    methods_order: Optional[list] = None,
) -> str:

#Figure 5-5. Pointwise Inference Trade-offs: Coverage, Interval Length, and Uncertainty

    ensure_dir(outdir)

    if dgps_order is None:
        dgps_order = ["DGP0", "DGP3", "DGP4", "DGP7"]
    if methods_order is None:
        methods_order = ["CF", "DML-CF", "ORF"]

    need_cols = {
        "dgp", "n", "method",
        "pointwise_ci_len_mean",
        "pointwise_cover_rate_mean",
        "pointwise_se_cv_mean"
    }
    miss = need_cols - set(summary.columns)
    if miss:
        raise KeyError(f"[Figure 5-5] summary 缺少必要列: {miss}. 现有列: {list(summary.columns)}")

    df = summary[summary["n"] == n_ref].copy()
    if df.empty:
        raise ValueError(f"[Figure 5-5] summary 中找不到 n={n_ref} 的数据。")

    dgps = [d for d in dgps_order if d in set(df["dgp"].astype(str))]
    methods = [m for m in methods_order if m in set(df["method"].astype(str))]
    if not dgps:
        raise ValueError(f"[Figure 5-5] 未找到指定 DGP: {dgps_order}")
    if not methods:
        raise ValueError(f"[Figure 5-5] 未找到指定方法: {methods_order}")

    # v[dgp][method] = (ci_len, cover, se_cv)
    v = {}
    for dgp in dgps:
        v[dgp] = {}
        for m in methods:
            sub = df[(df["dgp"].astype(str) == dgp) & (df["method"].astype(str) == m)]
            if len(sub) == 0:
                continue
            ci_len = float(sub["pointwise_ci_len_mean"].iloc[0])
            cover  = float(sub["pointwise_cover_rate_mean"].iloc[0])
            se_cv  = float(sub["pointwise_se_cv_mean"].iloc[0])
            v[dgp][m] = (ci_len, cover, se_cv)

    all_ci = [v[d][m][0] for d in v for m in v[d]]
    all_cv = [v[d][m][2] for d in v for m in v[d]]
    if len(all_ci) == 0:
        raise ValueError("[Figure 5-5] 没有可用数据绘制。")

    max_ci = max(all_ci)
    xlim_seg = (-1.15 * max_ci / 2, 1.15 * max_ci / 2)


    y_low, y_high = 0.5, 2.0

    cv_min, cv_max = (min(all_cv), max(all_cv)) if len(all_cv) else (0.0, 1.0)
    def bubble_size(se_cv: float) -> float:
        if cv_max <= cv_min + 1e-12:
            return 140.0
        t = (se_cv - cv_min) / (cv_max - cv_min)
        return 90.0 + 600.0 * (t ** 1.2)

    color_map = {"CF": "C0", "DML-CF": "C1", "ORF": "C2"}


    fig = plt.figure(figsize=(14, 6))

    gs_left = fig.add_gridspec(
        2, 2,
        left=0.05, right=0.49,
        top=0.88, bottom=0.30,
        wspace=0.25, hspace=0.60  # 👈 从0.40改成0.60
    )

    gs_right = fig.add_gridspec(
        2, 2,
        left=0.55, right=0.98,
        top=0.88, bottom=0.30,
        wspace=0.25, hspace=0.60  # 👈 同样改
    )

    for idx, dgp in enumerate(dgps):
        ax = fig.add_subplot(gs_left[idx // 2, idx % 2])
        ax.set_title(dgp, fontsize=10)

        methods_y = ["CF", "DML-CF", "ORF"]
        methods_y = [m for m in methods_y if m in methods]
        y_pos = np.arange(len(methods_y))

        ax.axvline(0.0, linestyle="--", linewidth=1)

        for yi, m in enumerate(methods_y):
            if m not in v[dgp]:
                continue
            ci_len, cover, se_cv = v[dgp][m]
            half = ci_len / 2.0
            ax.hlines(yi, -half, half, colors=color_map.get(m, None), linewidth=3)
            ax.plot([0.0], [yi], marker="o", markersize=4, color=color_map.get(m, None))
            ax.text(half, yi, f" {ci_len:.3f}", va="center", ha="left", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods_y, fontsize=9)
        ax.set_xlim(*xlim_seg)
        ax.grid(axis="x", alpha=0.2)

    fig.text(0.27, 0.92, f"Segment Plot of Average CI Length (n={n_ref})", ha="center", fontsize=11)

    # Trade-off View: Coverage vs Average CI Length
    for idx, dgp in enumerate(dgps):
        ax = fig.add_subplot(gs_right[idx // 2, idx % 2])
        ax.set_title(dgp, fontsize=10)

        ax.axvline(0.95, linestyle="--", linewidth=1)

        for m in methods:
            if m not in v[dgp]:
                continue
            ci_len, cover, se_cv = v[dgp][m]
            ax.scatter(cover, ci_len, s=bubble_size(se_cv), color=color_map.get(m, None))

        ax.set_xlim(0.50, 1.00)
        ax.set_ylim(y_low, y_high)
        ax.set_yticks([0.5, 1.0, 1.5, 2.0])

        ax.set_xlabel("Coverage", fontsize=9)
        ax.set_ylabel("Avg CI Length", fontsize=9)
        ax.grid(alpha=0.2)

    fig.text(0.77, 0.92, "Trade-off View: Coverage vs Average CI Length", ha="center", fontsize=11)

    method_handles, method_labels = [], []
    for m in ["CF", "DML-CF", "ORF"]:
        if m in methods:
            h = plt.Line2D([0], [0], marker="o", linestyle="",
                           color=color_map.get(m, None), markersize=7)
            method_handles.append(h)
            method_labels.append(m)

    se_min = float(min(all_cv)) if len(all_cv) else 0.250
    se_max = float(max(all_cv)) if len(all_cv) else 0.556
    sizes_demo = [se_min, se_max]
    demo_handles = [plt.scatter([], [], s=bubble_size(x), color="gray") for x in sizes_demo]
    demo_labels = [f"SE CV = {x:.3f}" for x in sizes_demo]

    all_handles = method_handles + demo_handles
    all_labels = method_labels + demo_labels

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.08),
        ncol=len(all_handles),
        fontsize=9
    )

    fig.suptitle("Figure 5-5. Pointwise Inference Trade-offs: Coverage, Interval Length, and Uncertainty",
                 fontsize=14, fontweight="bold", y=0.98)

    save_path = os.path.join(outdir, f"fig5_5_inference_tradeoffs_n{n_ref}.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_fig5_5_from_merged_csv(
    summary_csv: str = "summary_all.csv",
    outdir: str = "plots_report",
    n_ref: int = 1000,
) -> str:
    summary = pd.read_csv(summary_csv)
    path = plot_fig5_5_inference_tradeoffs(summary, outdir=outdir, n_ref=n_ref)
    print("[DONE] Figure 5-5 saved to:", path)
    return path



# Figure 5-6: Coverage diagnostics across DGPs

def _pick_col_any(df: pd.DataFrame, candidates: list, label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"[Figure 5-6] 找不到 {label} 列。候选={candidates}，现有列={list(df.columns)}")


def plot_fig5_6_coverage_diagnostics(
    summary: pd.DataFrame,
    outdir: str,
    n_ref: int = 1000,
    nominal: float = 0.95,
    methods_order: Optional[list] = None,
) -> str:
    ensure_dir(outdir)

    if methods_order is None:
        methods_order = ["CF", "DML-CF", "ORF"]

    for col in ["dgp", "n", "method"]:
        if col not in summary.columns:
            raise KeyError(f"[Figure 5-6] summary 缺少必要列: {col}")

    cover_col = _pick_col_any(
        summary,
        ["pointwise_cover_rate_mean", "pointwise_coverage_mean", "pointwise_cover_mean"],
        "overall pointwise coverage mean",
    )

    # quantile cover Q1..Q5
    q_cols = []
    for k in range(1, 6):
        q_cols.append(_pick_col_any(
            summary,
            [f"pointwise_cover_q{k}_mean", f"pointwise_cover_q{k}", f"cover_q{k}_mean"],
            f"quantile coverage q{k}"
        ))

    # (a)
    cover_se_col = None
    for cand in ["pointwise_cover_rate_se", "pointwise_cover_rate_sem", "pointwise_cover_rate_std"]:
        if cand in summary.columns:
            cover_se_col = cand
            break

    df_ref = summary[summary["n"] == n_ref].copy()
    if df_ref.empty:
        raise ValueError(f"[Figure 5-6] summary 中找不到 n={n_ref} 的数据。")

    def _dgp_sort_key(x: str):
        if x == "RCT":
            return (-1, -1)
        if x.startswith("DGP"):
            try:
                return (0, int(x.replace("DGP", "")))
            except:
                return (0, 999)
        return (1, 999)

    dgps = sorted(df_ref["dgp"].astype(str).unique().tolist(), key=_dgp_sort_key)
    methods = [m for m in methods_order if m in set(df_ref["method"].astype(str))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    x = np.arange(len(dgps))
    width = 0.8 / max(1, len(methods))

    for i, m in enumerate(methods):
        gaps = []
        errs = []
        for dgp in dgps:
            sub = df_ref[(df_ref["dgp"].astype(str) == dgp) & (df_ref["method"].astype(str) == m)]
            if len(sub) == 0:
                gaps.append(np.nan)
                errs.append(np.nan)
                continue
            cov = float(sub[cover_col].iloc[0])
            gaps.append(cov - nominal)
            if cover_se_col is not None:
                val = float(sub[cover_se_col].iloc[0])
                # 如果是std而不是se，仍然给一个误差棒（图形层面）
                errs.append(val)
            else:
                errs.append(np.nan)

        xpos = x + (i - (len(methods) - 1) / 2) * width
        ax_a.errorbar(
            xpos, gaps,
            yerr=None if cover_se_col is None else errs,
            fmt="o",
            capsize=3,
            label=m
        )

    ax_a.axhline(0.0, linestyle="--", linewidth=1)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(dgps, rotation=0, fontsize=8)
    ax_a.set_ylabel("Coverage gap (coverage - 0.95)")
    ax_a.set_title("(a) Overall pointwise coverage gap")
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.2)

    # (b) Quantile-group coverage gaps: boxplots across DGPs
    data_by_q = {q: {m: [] for m in methods} for q in range(1, 6)}
    for dgp in dgps:
        for m in methods:
            sub = df_ref[(df_ref["dgp"].astype(str) == dgp) & (df_ref["method"].astype(str) == m)]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
            for qi, col in enumerate(q_cols, start=1):
                data_by_q[qi][m].append(float(row[col]) - nominal)

    # boxplot
    positions = []
    box_data = []
    colors = []
    color_map = {"CF": "C0", "DML-CF": "C1", "ORF": "C2"}

    base = 1
    for qi in range(1, 6):
        for j, m in enumerate(methods):
            positions.append(base + j * 0.25)
            box_data.append(data_by_q[qi][m] if len(data_by_q[qi][m]) else [np.nan])
            colors.append(color_map.get(m, None))
        base += 1.2

    bp = ax_b.boxplot(box_data, positions=positions, widths=0.18, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)


    q_centers = [1 + (k - 1) * 1.2 + 0.25 for k in range(1, 6)]
    ax_b.set_xticks(q_centers)
    ax_b.set_xticklabels([f"Q{k}" for k in range(1, 6)], fontsize=9)
    ax_b.axhline(0.0, linestyle="--", linewidth=1)
    ax_b.set_ylabel("Coverage gap (q-coverage - 0.95)")
    ax_b.set_title("(b) Quantile-group coverage gaps")

    handles = [plt.Line2D([0], [0], color=color_map[m], marker="s", linestyle="", label=m) for m in methods]
    ax_b.legend(handles=handles, fontsize=8, loc="upper center", ncol=3)
    ax_b.grid(alpha=0.2)


    # (c) ORF advantage heatmap in overall pointwise coverage
    rivals = [r for r in ["CF", "DML-CF"] if r in methods]
    heat = np.full((len(dgps), len(rivals)), np.nan)

    for i, dgp in enumerate(dgps):
        sub_orf = df_ref[(df_ref["dgp"].astype(str) == dgp) & (df_ref["method"].astype(str) == "ORF")]
        if len(sub_orf) == 0:
            continue
        cov_orf = float(sub_orf[cover_col].iloc[0])
        for j, rv in enumerate(rivals):
            sub_rv = df_ref[(df_ref["dgp"].astype(str) == dgp) & (df_ref["method"].astype(str) == rv)]
            if len(sub_rv) == 0:
                continue
            cov_rv = float(sub_rv[cover_col].iloc[0])
            heat[i, j] = cov_orf - cov_rv

    im = ax_c.imshow(heat, aspect="auto")
    ax_c.set_yticks(np.arange(len(dgps)))
    ax_c.set_yticklabels(dgps, fontsize=8)
    ax_c.set_xticks(np.arange(len(rivals)))
    ax_c.set_xticklabels([f"ORF - {rv}" for rv in rivals], fontsize=9)
    ax_c.set_title("(c) ORF advantage heatmap")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Difference in coverage")

    for i in range(len(dgps)):
        for j in range(len(rivals)):
            if np.isfinite(heat[i, j]):
                ax_c.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center", fontsize=7)


    # (d) Coverage vs sample size (mean across DGPs; band=IQR)
    df_all = summary.copy()
    ns = sorted(df_all["n"].dropna().unique().tolist())

    for m in methods:
        means = []
        q25s = []
        q75s = []
        for n in ns:
            tmp = df_all[(df_all["n"] == n) & (df_all["method"].astype(str) == m)]
            vals = tmp[cover_col].astype(float).values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                means.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)
            else:
                means.append(float(np.mean(vals)))
                q25s.append(float(np.quantile(vals, 0.25)))
                q75s.append(float(np.quantile(vals, 0.75)))

        ax_d.plot(ns, means, marker="o", label=m)
        ax_d.fill_between(ns, q25s, q75s, alpha=0.15)

    ax_d.axhline(nominal, linestyle="--", linewidth=1, label="Nominal 0.95")
    ax_d.set_xlabel("Sample size n")
    ax_d.set_ylabel("Coverage")
    ax_d.set_title("(d) Coverage vs sample size")
    ax_d.set_ylim(0.5, 1.02)
    ax_d.legend(fontsize=8, loc="lower right")
    ax_d.grid(alpha=0.2)

    fig.suptitle("Figure 5-6 Coverage diagnostics across DGPs.", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(outdir, f"fig5_6_coverage_diagnostics_n{n_ref}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_fig5_6_from_merged_csv(
    summary_csv: str = "summary_all.csv",
    outdir: str = "plots_report",
    n_ref: int = 1000,
) -> str:

    summary = pd.read_csv(summary_csv)
    path = plot_fig5_6_coverage_diagnostics(summary, outdir=outdir, n_ref=n_ref)
    print("[DONE] Figure 5-6 saved to:", path)
    return path



# Figure 5-7: One-Factor Sensitivity of ORF Relative Performance

def plot_fig5_7_one_factor_sensitivity(
    summary: pd.DataFrame,
    outdir: str,
    n_ref: int = 1000,
    baseline_dgp: str = "DGP0",
    factors: Optional[list] = None,
) -> str:

    ensure_dir(outdir)

    if factors is None:

        factors = [
            ("Overlap", "DGP1"),
            ("Alignment", "DGP2"),
            ("Overlap + alignment", "DGP3"),
            ("Noise", "DGP4"),
            ("Dimension", "DGP7"),
        ]


    rmse_col = None
    for cand in ["rmse_mean", "rmse", "RMSE_mean"]:
        if cand in summary.columns:
            rmse_col = cand
            break
    if rmse_col is None:
        raise KeyError(f"[Figure 5-7] summary 找不到 rmse 列。现有列={list(summary.columns)}")

    for col in ["dgp", "n", "method"]:
        if col not in summary.columns:
            raise KeyError(f"[Figure 5-7] summary 缺少必要列: {col}")

    df = summary[summary["n"] == n_ref].copy()
    if df.empty:
        raise ValueError(f"[Figure 5-7] summary 中找不到 n={n_ref} 的数据。")

    def get_rmse(dgp: str, method: str) -> float:
        sub = df[(df["dgp"].astype(str) == dgp) & (df["method"].astype(str) == method)]
        if len(sub) == 0:
            raise KeyError(f"[Figure 5-7] 找不到 (dgp={dgp}, method={method}, n={n_ref}) 的 rmse")
        return float(sub[rmse_col].iloc[0])

    def rel_gap(dgp: str, rival: str) -> float:
        rmse_orf = get_rmse(dgp, "ORF")
        rmse_rv = get_rmse(dgp, rival)
        return (rmse_orf - rmse_rv) / rmse_rv

    base_gap_cf = rel_gap(baseline_dgp, "CF")
    base_gap_dml = rel_gap(baseline_dgp, "DML-CF")

    labels = [name for name, _ in factors]
    gaps_cf = [rel_gap(dgp, "CF") for _, dgp in factors]
    gaps_dml = [rel_gap(dgp, "DML-CF") for _, dgp in factors]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

    def draw_panel(ax, gaps, base_gap, title, xlabel):
        y = np.arange(len(labels))

        ax.axvline(base_gap, linestyle="--", linewidth=1, label=f"Baseline DGP ({baseline_dgp}) : {base_gap:.3f}")

        ax.axvline(0.0, linestyle="-", linewidth=1, alpha=0.3)


        for yi, g in enumerate(gaps):
            # 从 base 到 g 画一条线
            ax.hlines(yi, xmin=min(base_gap, g), xmax=max(base_gap, g), colors="C1", linewidth=2)
            ax.plot(g, yi, marker="o", color="C1")
            ax.text(g, yi, f" {g:.3f}", va="center", ha="left", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()  # 让 Dimension 在最上面更像报告（你也可注释掉）
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=7, loc="lower right")

    draw_panel(
        axes[0],
        gaps_cf,
        base_gap_cf,
        "(a) ORF relative to CF",
        f"Relative RMSE gap: (RMSE_ORF − RMSE_CF) / RMSE_CF   (n={n_ref})"
    )
    draw_panel(
        axes[1],
        gaps_dml,
        base_gap_dml,
        "(b) ORF relative to DML-CF",
        f"Relative RMSE gap: (RMSE_ORF − RMSE_DMLCF) / RMSE_DMLCF   (n={n_ref})"
    )

    fig.suptitle("Figure 5-7 One-Factor Sensitivity of ORF Relative Performance",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    save_path = os.path.join(outdir, f"fig5_7_one_factor_sensitivity_n{n_ref}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_fig5_7_from_merged_csv(
    summary_csv: str = "summary_all.csv",
    outdir: str = "plots_report",
    n_ref: int = 1000,
) -> str:

    summary = pd.read_csv(summary_csv)
    path = plot_fig5_7_one_factor_sensitivity(summary, outdir=outdir, n_ref=n_ref)
    print("[DONE] Figure 5-7 saved to:", path)
    return path


#Figure 5-8: Overlap–alignment interaction in relative RMSE (n=1000)
# (a) ORF vs CF: Interaction plot
# (b) ORF vs DML-CF: Interaction plot
# (c) ORF vs CF: Cell-wise boxplots of ΔRMSE
# (d) ORF vs DML-CF: Cell-wise boxplots of ΔRMSE

def plot_fig5_8_overlap_alignment_interaction(
    metrics_long: pd.DataFrame,
    outdir: str,
    n_ref: int = 1000,
    cell_map: Optional[dict] = None,
) -> str:
    """
    画报告 Figure 5-8（interaction + cell-wise boxplots）。
    """
    ensure_dir(outdir)

    # --- 必要列检查 ---
    for col in ["dgp", "n", "method"]:
        if col not in metrics_long.columns:
            raise KeyError(f"[Figure 5-8] metrics_long 缺少必要列: {col}")

    # rmse列名兼容
    rmse_col = None
    for cand in ["rmse", "rmse_mean", "rmse_val"]:
        if cand in metrics_long.columns:
            rmse_col = cand
            break
    if rmse_col is None:
        raise KeyError(f"[Figure 5-8] metrics_long 找不到 rmse 列。现有列={list(metrics_long.columns)}")

    # Good overlap / Weak overlap × Alignment high / low
    if cell_map is None:
        cell_map = {
            ("Good overlap", "Alignment: high"): "DGP0",
            ("Weak overlap", "Alignment: high"): "DGP1",
            ("Good overlap", "Alignment: low"):  "DGP2",
            ("Weak overlap", "Alignment: low"):  "DGP3",
        }

    df = metrics_long[metrics_long["n"] == n_ref].copy()
    if df.empty:
        raise ValueError(f"[Figure 5-8] metrics_long 中找不到 n={n_ref} 的数据。")

    needed_dgps = set(cell_map.values())
    df = df[df["dgp"].astype(str).isin(needed_dgps)].copy()
    if df.empty:
        raise ValueError(f"[Figure 5-8] metrics_long 中找不到 {sorted(needed_dgps)} 的数据。")

    rep_col = None
    for cand in ["rep", "r", "mc_id", "trial", "run_id", "seed"]:
        if cand in df.columns:
            rep_col = cand
            break
    if rep_col is None:
        rep_col = "__rep__"
        df[rep_col] = np.arange(len(df))

    pvt = df.pivot_table(index=["dgp", rep_col], columns="method", values=rmse_col, aggfunc="mean")

    for m in ["ORF", "CF", "DML-CF"]:
        if m not in pvt.columns:
            raise KeyError(
                f"[Figure 5-8] 需要方法 {m}，但 metrics_long 中没有。现有方法={list(pvt.columns)}"
            )

    gap_cf = ((pvt["ORF"] - pvt["CF"]) / pvt["CF"]).rename("gap_cf")
    gap_dml = ((pvt["ORF"] - pvt["DML-CF"]) / pvt["DML-CF"]).rename("gap_dml")
    gap_df = pd.concat([gap_cf, gap_dml], axis=1).reset_index()  # dgp, rep_col, gap_cf, gap_dml

    overlap_order = ["Good overlap", "Weak overlap"]
    align_order = ["Alignment: high", "Alignment: low"]

    def cell_series(metric: str, overlap: str, align: str) -> np.ndarray:
        dgp = cell_map[(overlap, align)]
        arr = gap_df[gap_df["dgp"].astype(str) == dgp][metric].astype(float).values
        return arr[np.isfinite(arr)]

    def mean_iqr(arr: np.ndarray):
        if len(arr) == 0:
            return np.nan, np.nan, np.nan
        m = float(np.mean(arr))
        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        return m, q1, q3


    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    def draw_interaction(ax, metric: str, title: str):
        x = np.arange(len(overlap_order))  # 0,1
        for align in align_order:
            means = []
            yerr_low = []
            yerr_high = []
            for ov in overlap_order:
                arr = cell_series(metric, ov, align)
                m, q1, q3 = mean_iqr(arr)
                means.append(m)
                yerr_low.append(m - q1 if np.isfinite(m) and np.isfinite(q1) else np.nan)
                yerr_high.append(q3 - m if np.isfinite(m) and np.isfinite(q3) else np.nan)

            ax.errorbar(
                x, means,
                yerr=[yerr_low, yerr_high],
                marker="o",
                linewidth=2,
                capsize=4,
                label=align
            )

        ax.set_xticks(x)
        ax.set_xticklabels(overlap_order, fontsize=9)
        ax.axhline(0.0, linewidth=1, alpha=0.3)
        ax.set_xlabel("Overlap regime", fontsize=9)

        ax.set_ylabel("")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.2)

    def draw_cellwise_box(ax, metric: str, title: str):
        positions = []
        data = []
        colors = []
        color_map = {"Alignment: high": "C0", "Alignment: low": "C1"}

        for gi, ov in enumerate(overlap_order, start=1):
            for align in align_order:
                pos = gi + (-0.18 if align == "Alignment: high" else 0.18)
                arr = cell_series(metric, ov, align)
                data.append(arr if len(arr) else np.array([np.nan]))
                positions.append(pos)
                colors.append(color_map[align])

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.28,
            patch_artist=True,
            showfliers=False
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.25)

        ax.axhline(0.0, linewidth=1, alpha=0.3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(overlap_order, fontsize=9)
        ax.set_xlabel("Overlap regime", fontsize=9)

        ax.set_ylabel("")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)

        handles = [
            plt.Line2D([0], [0], color="C0", lw=6, alpha=0.25, label="Alignment = high"),
            plt.Line2D([0], [0], color="C1", lw=6, alpha=0.25, label="Alignment = low"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper center", ncol=2)

    draw_interaction(
        ax_a,
        metric="gap_cf",
        title=f"(a) ORF vs CF: Interaction plot"
    )
    draw_interaction(
        ax_b,
        metric="gap_dml",
        title=f"(b) ORF vs DML-CF: Interaction plot"
    )
    draw_cellwise_box(
        ax_c,
        metric="gap_cf",
        title=f"(c) ORF vs CF: Cell-wise boxplots of ΔRMSE"
    )
    draw_cellwise_box(
        ax_d,
        metric="gap_dml",
        title=f"(d) ORF vs DML-CF: Cell-wise boxplots of ΔRMSE"
    )

    fig.suptitle(f"Figure 5-8: Overlap–alignment interaction in relative RMSE (n = {n_ref})",
                 fontsize=14, fontweight="bold", y=0.98)

    fig.text(
        0.5, 0.01,
        "ΔRMSE = (RMSE_ORF − RMSE_rival) / RMSE_rival",
        ha="center",
        fontsize=10
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    save_path = os.path.join(outdir, f"fig5_8_overlap_alignment_interaction_n{n_ref}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_fig5_8_from_merged_csv(
    metrics_csv: str = "metrics_long_all.csv",
    outdir: str = "plots_report",
    n_ref: int = 1000,
) -> str:
    metrics_long = pd.read_csv(metrics_csv)
    path = plot_fig5_8_overlap_alignment_interaction(metrics_long, outdir=outdir, n_ref=n_ref)
    print("[DONE] Figure 5-8 saved to:", path)
    return path
