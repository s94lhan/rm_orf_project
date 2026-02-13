
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

MISSPEC_DGPS = {"DGP0", "DGP3", "DGP4", "DGP7"}
MISSPEC_N = 1000

import os
import numpy as np
import pandas as pd
import zlib
import pickle
from pathlib import Path
#misspec diagnostic control
MISSPEC_DIAG_DGPS = {"DGP0", "DGP3", "DGP4", "DGP7"}
MISSPEC_DIAG_N = 1000

import warnings
from typing import List, Optional, Dict, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm
from metrics import clan_diffs, clan_nonlinear_diffs

from config import DGPConfig
from dgp import gen_data
from models import fit_orf_econml, fit_cf_econml, fit_dml_cf_econml, get_method_hparams
from metrics import (
    point_metrics, safe_spearman, gates_stats, clan_diffs, top_bottom_10,
    ci_pointwise_coverage,
)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def generate_fixed_test_set(
        cfg: DGPConfig,
        N_test: int = 500,
        seed: int = 42,
        cache_dir: str = "./test_sets"
) -> Tuple[np.ndarray, np.ndarray]:
    #Generate a fixed test set for each DGP and cache it.
    os.makedirs(cache_dir, exist_ok=True)


    name_tag = zlib.adler32(cfg.name.encode("utf-8")) % 1000
    cache_file = os.path.join(cache_dir, f"test_set_{cfg.name}_{name_tag}_n{N_test}_seed{seed}.pkl")


    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            Xte, tau_true_te = pickle.load(f)
        print(f"加载缓存的测试集: {cache_file}")
        return Xte, tau_true_te


    print(f"生成新的测试集并缓存: {cache_file}")
    set_global_seed(seed)
    Xte, _, _, tau_true_te, _ = gen_data(cfg, n=N_test, seed=seed)
    tau_true_te = np.asarray(tau_true_te).ravel()


    with open(cache_file, 'wb') as f:
        pickle.dump((Xte, tau_true_te), f)

    return Xte, tau_true_te


def run_one_rep(
        cfg: DGPConfig,
        n: int,
        rep: int,
        seed_base: int,
        fixed_test_data: Tuple[np.ndarray, np.ndarray],
        methods: List[str],
        diag_misspec: bool,
        save_pred_dir: Optional[str] = None,
        save_pred_rep: int = 1,
) -> pd.DataFrame:
    Xte, tau_true_te = fixed_test_data

    name_tag = zlib.adler32(cfg.name.encode("utf-8")) % 1000
    seed = seed_base + rep * 10_000 + name_tag + n
    set_global_seed(seed)

    # Generate a training set (sample size = n), and regenerate it for each repetition.
    Xtr, Ttr, Ytr, _, _ = gen_data(cfg, n=n, seed=seed)
    Ttr = np.asarray(Ttr).ravel()
    Ytr = np.asarray(Ytr).ravel()

    rows = []

    # save first-rep test set & predictions for calibration plots
    pred_pack = None
    if save_pred_dir is not None and rep == save_pred_rep:
        os.makedirs(save_pred_dir, exist_ok=True)
        pred_pack = {
            "tau_true_test": tau_true_te.astype(float),
            "n_test": int(Xte.shape[0]),
        }

    Xte_first5 = Xte[:, :5]  # Only the first 5 features are needed

    for m in methods:
        tau_hat = None
        se_hat = None

        if m == "ORF":
            model = fit_orf_econml(Xtr, Ttr, Ytr, n=n, seed=seed, misspec=False)
            tau_hat = np.asarray(model.effect(Xte)).reshape(-1)
            try:
                se_hat = np.asarray(model.effect_inference(Xte).stderr).reshape(-1)
            except Exception:
                se_hat = None


        elif m == "CF":
            model = fit_cf_econml(Xtr, Ttr, Ytr, n=n, seed=seed)
            tau_hat = np.asarray(model.predict(Xte)).reshape(-1)
            se_hat = None
            try:
                if hasattr(model, "predict_inference"):
                    inf = model.predict_inference(Xte)
                    if hasattr(inf, "stderr"):
                        se_hat = np.asarray(inf.stderr).reshape(-1)
                if se_hat is None and hasattr(model, "prediction_stderr"):
                    se_hat = np.asarray(model.prediction_stderr(Xte)).reshape(-1)
            except Exception:
                se_hat = None

        elif m == "DML-CF":
            model = fit_dml_cf_econml(Xtr, Ttr, Ytr, n=n, seed=seed, misspec=False)
            tau_hat = np.asarray(model.effect(Xte)).reshape(-1)
            try:
                se_hat = np.asarray(model.effect_inference(Xte).stderr).reshape(-1)
            except Exception:
                se_hat = None

        else:
            raise ValueError(f"未知方法: {m}")

        assert tau_hat.shape[0] == Xte.shape[0]
        if se_hat is not None:
            assert se_hat.shape[0] == Xte.shape[0]

        # save predictions for calibration
        if pred_pack is not None:
            pred_pack[f"tau_hat_{m}_test"] = tau_hat.astype(float)

        pm = point_metrics(tau_hat, tau_true_te)
        sp = safe_spearman(tau_hat, tau_true_te)
        gates_rmse, gates_hat_means, gates_true_means, _ = gates_stats(tau_hat, tau_true_te, K=5)
        #  Xte_first5
        cm = clan_diffs(Xte_first5, tau_hat, top_q=0.2, bottom_q=0.2, s=5)
        cm_nl = clan_nonlinear_diffs(Xte_first5, tau_hat, top_q=0.2, bottom_q=0.2)
        tb = top_bottom_10(tau_hat, tau_true_te)
        # pointwise SE
        ci = ci_pointwise_coverage(
            tau_hat=tau_hat,
            tau_true=tau_true_te,
            se_hat=se_hat,
            K=5,
            z=1.96,
        )
        # z = (hat - true) / se
        z_sd = np.nan
        z_p975 = np.nan
        if se_hat is not None:
            z = (tau_hat - tau_true_te) / (se_hat + 1e-12)
            z_sd = float(np.nanstd(z))
            z_p975 = float(np.nanpercentile(np.abs(z), 97.5))


        rows.append({
            "dgp": cfg.name, "n": n, "rep": rep, "method": m,
            "spearman": sp, "gates_rmse": gates_rmse,
            **pm, **cm,**cm_nl, **tb, **ci,
            **get_method_hparams(m, n, misspec=False),
            # ✅ 新增：每个rep的5组 GATEs 均值（hat/true）
            "z_sd": z_sd,
            "z_p975": z_p975,
            "gates_hat_q1": float(gates_hat_means[0]),
            "gates_hat_q2": float(gates_hat_means[1]),
            "gates_hat_q3": float(gates_hat_means[2]),
            "gates_hat_q4": float(gates_hat_means[3]),
            "gates_hat_q5": float(gates_hat_means[4]),
            "gates_true_q1": float(gates_true_means[0]),
            "gates_true_q2": float(gates_true_means[1]),
            "gates_true_q3": float(gates_true_means[2]),
            "gates_true_q4": float(gates_true_means[3]),
            "gates_true_q5": float(gates_true_means[4]),
        })

        # diag_misspec
        if diag_misspec and (cfg.name in MISSPEC_DGPS) and (n == MISSPEC_N) and (m in ("ORF", "DML-CF")):
            if rep == 0 and n == 1000 and m in ("ORF", "DML-CF"):
                print("MISSPEC CHECK:",
                      "diag_misspec=", diag_misspec,
                      "cfg.name=", cfg.name,
                      "in_list=", (cfg.name in MISSPEC_DGPS),
                      "n=", n,
                      "n_ok=", (n == MISSPEC_N),
                      "m=", m)

            if m == "ORF":
                model_ms = fit_orf_econml(Xtr, Ttr, Ytr, n=n, seed=seed, misspec=True)
                tau_ms = np.asarray(model_ms.effect(Xte)).reshape(-1)
                try:
                    se_ms = np.asarray(model_ms.effect_inference(Xte).stderr).reshape(-1)
                except Exception:
                    se_ms = None

                pm_ms = point_metrics(tau_ms, tau_true_te)
                cm_ms = clan_diffs(Xte_first5, tau_ms, top_q=0.2, bottom_q=0.2, s=5)
                tb_ms = top_bottom_10(tau_ms, tau_true_te)
                ci_ms = ci_pointwise_coverage(
                    tau_hat=tau_ms,
                    tau_true=tau_true_te,
                    se_hat=se_ms,
                    K=5,
                    z=1.96,
                )
                print("MISSPEC APPEND:", cfg.name, n, rep, m)
                rows.append({
                    "dgp": cfg.name, "n": n, "rep": rep,
                    **get_method_hparams("ORF", n, misspec=True),
                    "method": "ORF_misspec",
                    "spearman": safe_spearman(tau_ms, tau_true_te),
                    "gates_rmse": gates_stats(tau_ms, tau_true_te, K=5)[0],
                    **pm_ms, **cm_ms, **tb_ms, **ci_ms
                })

            if m == "DML-CF":
                model_ms = fit_dml_cf_econml(Xtr, Ttr, Ytr, n=n, seed=seed, misspec=True)
                tau_ms = np.asarray(model_ms.effect(Xte)).reshape(-1)
                try:
                    se_ms = np.asarray(model_ms.effect_inference(Xte).stderr).reshape(-1)
                except Exception:
                    se_ms = None

                pm_ms = point_metrics(tau_ms, tau_true_te)
                cm_ms = clan_diffs(Xte_first5, tau_ms, top_q=0.2, bottom_q=0.2, s=5)
                tb_ms = top_bottom_10(tau_ms, tau_true_te)
                ci_ms = ci_pointwise_coverage(
                    tau_hat=tau_ms,
                    tau_true=tau_true_te,
                    se_hat=se_ms,
                    K=5,
                    z=1.96,
                )

                rows.append({
                    "dgp": cfg.name, "n": n, "rep": rep,
                    **get_method_hparams("DML-CF", n, misspec=True),
                    "method": "DML-CF_misspec",
                    "spearman": safe_spearman(tau_ms, tau_true_te),
                    "gates_rmse": gates_stats(tau_ms, tau_true_te, K=5)[0],
                    **pm_ms, **cm_ms, **tb_ms, **ci_ms
                })

    # write prediction pack once per (dgp,n,rep)
    if pred_pack is not None:
        fn = f"pred_{cfg.name}_n{n}_rep{rep}.npz"
        filepath = os.path.join(save_pred_dir, fn)
        np.savez_compressed(filepath, **pred_pack)
        print(f"[DEBUG] 保存预测到: {filepath}")

    return pd.DataFrame(rows)


def summarize(metrics_long: pd.DataFrame) -> pd.DataFrame:

    def p90(x):
        x_clean = x.dropna()
        return np.quantile(x_clean, 0.90) if len(x_clean) > 0 else np.nan

    def safe_mean(x):
        """安全的均值计算，避免空值"""
        x_clean = x.dropna()
        return np.mean(x_clean) if len(x_clean) > 0 else np.nan

    def safe_std(x):
        x_clean = x.dropna()
        if len(x_clean) <= 1:
            return np.nan
        return np.std(x_clean, ddof=1)

    base_aggs = {
        "bias_mean": ("bias", safe_mean),
        "bias_sd": ("bias", safe_std),
        "rmse_mean": ("rmse", safe_mean),
        "rmse_sd": ("rmse", safe_std),
        "rmse_p90": ("rmse", p90),
        "mae_mean": ("mae", safe_mean),
        "spearman_mean": ("spearman", safe_mean),
        "gates_rmse_mean": ("gates_rmse", safe_mean),
        "clan_diff_x1_mean": ("clan_diff_x1", safe_mean),
        "clan_diff_x2_mean": ("clan_diff_x2", safe_mean),
        "clan_diff_sin_pi_x2_mean": ("clan_diff_sin_pi_x2", safe_mean),
        "clan_diff_abs_x2_m05_mean": ("clan_diff_abs_x2_m05", safe_mean),

    }
    cov_aggs = {
        "pointwise_cover_rate_mean": ("pointwise_cover_rate", safe_mean),
        "pointwise_ci_len_mean": ("pointwise_ci_len", safe_mean),
        "pointwise_se_cv_mean": ("pointwise_se_cv", safe_mean),
        "pointwise_cover_q1_mean": ("pointwise_cover_q1", safe_mean),
        "pointwise_cover_q2_mean": ("pointwise_cover_q2", safe_mean),
        "pointwise_cover_q3_mean": ("pointwise_cover_q3", safe_mean),
        "pointwise_cover_q4_mean": ("pointwise_cover_q4", safe_mean),
        "pointwise_cover_q5_mean": ("pointwise_cover_q5", safe_mean),
    }

    g = metrics_long.groupby(["dgp", "n", "method"])
    out = g.agg(**base_aggs, **cov_aggs).reset_index()
    return out


def run_grid(
        dgps: List[DGPConfig],
        n_list: List[int],
        R: int,
        seed_base: int,
        methods: List[str],
        n_jobs: int,
        diag_misspec: bool,
        N_test: int = 500,
        test_cache_dir: str = "./test_sets"
) -> pd.DataFrame:

    # Correctly create the preds directory
    if test_cache_dir:
        # 获取输出目录
        outdir = os.path.dirname(test_cache_dir)
        if not outdir:  # 如果test_cache_dir就是当前目录
            outdir = "."
        preds_dir = os.path.join(outdir, "preds")
    else:
        preds_dir = "./preds"

    os.makedirs(preds_dir, exist_ok=True)
    print(f"[INFO] 预测保存目录: {preds_dir}")

    # Generate a fixed test set for each DGP.
    print("为每个DGP生成固定的测试集...")
    fixed_test_sets = {}
    for cfg in dgps:
        Xte, tau_true_te = generate_fixed_test_set(
            cfg,
            N_test=N_test,
            seed=seed_base,  # 使用base seed生成测试集
            cache_dir=test_cache_dir
        )
        fixed_test_sets[cfg.name] = (Xte, tau_true_te)
        print(f"DGP: {cfg.name}, 测试集样本量: {Xte.shape[0]}")

    tasks = [(cfg, n, rep) for cfg in dgps for n in n_list for rep in range(R)]

    def _one(task):
        cfg, n, rep = task
        fixed_test_data = fixed_test_sets[cfg.name]
        return run_one_rep(
            cfg=cfg,
            n=n,
            rep=rep,
            seed_base=seed_base,
            fixed_test_data=fixed_test_data,
            methods=methods,
            diag_misspec=diag_misspec,
            save_pred_dir=preds_dir,
            save_pred_rep=1
        )

    if n_jobs == 1:
        dfs = []
        for t in tqdm(tasks, desc="Simulating"):
            dfs.append(_one(t))
        return pd.concat(dfs, ignore_index=True)

    dfs = Parallel(n_jobs=n_jobs)(delayed(_one)(t) for t in tqdm(tasks, desc="Simulating"))
    return pd.concat(dfs, ignore_index=True)


def paired_deltas(metrics_long: pd.DataFrame, base: str = "ORF", rivals=None) -> pd.DataFrame:
    from scipy.stats import ttest_1samp

    df = metrics_long.copy()

    metrics_to_compare = ["rmse", "bias", "mae", "spearman", "gates_rmse", "pointwise_cover_rate"]
    metrics_to_compare = [c for c in metrics_to_compare if c in df.columns]

    if df.empty or len(metrics_to_compare) == 0:
        print("[WARNING] 没有足够的数据进行配对差分析")
        return pd.DataFrame()

    piv = df.pivot_table(
        index=["dgp", "n", "rep"],
        columns="method",
        values=metrics_to_compare,
        aggfunc="mean"
    )

    methods = sorted(df["method"].unique())
    if rivals is None:
        rivals = [m for m in methods if m != base]

    out_rows = []
    for dgp in sorted(df["dgp"].unique()):
        for n in sorted(df["n"].unique()):
            base_col = piv.loc[(dgp, n)]
            if base not in piv.columns.levels[1]:
                continue
            for rv in rivals:
                if rv not in piv.columns.levels[1]:
                    continue
                # metrics to compare
                for metric in metrics_to_compare:
                    b = base_col[(metric, base)]
                    r = base_col[(metric, rv)]
                    d = (b - r).dropna()
                    if d.empty or len(d) <= 1:
                        continue

                    try:
                        tstat, pval = ttest_1samp(d.values, popmean=0.0, nan_policy="omit")
                    except Exception as e:
                        print(f"[WARNING] t检验失败 ({dgp}, n={n}, {base} vs {rv}, {metric}): {e}")
                        tstat, pval = np.nan, np.nan

                    out_rows.append({
                        "dgp": dgp, "n": n, "base": base, "rival": rv, "metric": metric,
                        "delta_mean": float(np.mean(d.values)),
                        "delta_sd": float(np.std(d.values, ddof=1)) if len(d) > 1 else np.nan,
                        "delta_p90": float(np.quantile(d.values, 0.90)),
                        "t_stat": float(tstat) if tstat == tstat else np.nan,
                        "p_value": float(pval) if pval == pval else np.nan,
                        "n_rep_used": int(len(d))
                    })

    if not out_rows:
        print("[WARNING] 配对差分析没有生成任何结果")

    return pd.DataFrame(out_rows)


def diagnose_metrics(metrics_long: pd.DataFrame):
    print("\n=== 诊断报告 ===")

    # Check for outliers in each column.
    for col in metrics_long.columns:
        if metrics_long[col].dtype in [np.float64, np.float32]:
            data = metrics_long[col].dropna()

            if len(data) == 0:
                continue

            zeros = (data == 0).sum()
            neg_inf = np.isneginf(data).sum()
            pos_inf = np.isposinf(data).sum()

            if zeros > 0 or neg_inf > 0 or pos_inf > 0:
                print(f"{col}:")
                print(f"  零值: {zeros}")
                print(f"  负无穷: {neg_inf}")
                print(f"  正无穷: {pos_inf}")

                if zeros > 0:
                    zero_rows = metrics_long[metrics_long[col] == 0]
                    print(f"  零值来自: {zero_rows['method'].unique()}")

    # check spearman and pointwise_se_cv
    for col in ["spearman", "pointwise_se_cv"]:
        if col in metrics_long.columns:
            stats = metrics_long[col].describe()
            print(f"\n{col}统计:")
            print(f"  非空值: {stats['count']}")
            print(f"  均值: {stats['mean']:.4f}")
            print(f"  标准差: {stats['std']:.4f}")
            print(f"  最小值: {stats['min']:.4f}")
            print(f"  最大值: {stats['max']:.4f}")