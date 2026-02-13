# main.py
import os
import sys
import argparse
import pandas as pd

from config import (
    make_dgp_configs,
    DEFAULT_N_LIST,
    DEFAULT_METHODS,
    DEFAULT_R,
    DEFAULT_N_TEST,
    DEFAULT_SEED,
)
from simulation import (
    run_grid,
    summarize,
    paired_deltas,
)
from merge_metrics import main as merge_metrics_main
from merge_summaries import main as merge_summaries_main


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def check_dependencies():
    try:
        import econml  # noqa
        import sklearn  # noqa
        import numpy  # noqa
        import pandas  # noqa
        import matplotlib  # noqa
    except ImportError as e:
        print(f"[ERROR] 缺少依赖包: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--R", type=int, default=DEFAULT_R)
    parser.add_argument("--N_test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--n_list", type=int, nargs="+", default=DEFAULT_N_LIST)
    parser.add_argument("--dgps", type=str, nargs="+", default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS)

    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--diag_misspec", action="store_true")
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args()

    check_dependencies()

    ensure_dir(args.outdir)
    plots_dir = os.path.join(args.outdir, "plots")
    preds_dir = os.path.join(args.outdir, "preds")
    ensure_dir(plots_dir)
    ensure_dir(preds_dir)

    print("[INFO] 构建 DGP 配置...")
    dgp_configs = make_dgp_configs()
    dgp_names = args.dgps if args.dgps else list(dgp_configs.keys())

    print("[INFO] 开始运行 Monte Carlo 实验...")
    dgps_selected = [dgp_configs[name] for name in dgp_names]

    test_cache_dir = os.path.join(args.outdir, "test_sets")

    metrics_long = run_grid(
        dgps=dgps_selected,
        n_list=args.n_list,
        R=args.R,
        seed_base=args.seed,  # 注意：simulation 里参数叫 seed_base
        methods=args.methods,
        n_jobs=args.n_jobs,
        diag_misspec=args.diag_misspec,
        N_test=args.N_test,
        test_cache_dir=test_cache_dir
    )

    metrics_path = os.path.join(args.outdir, "metrics_long.csv")
    metrics_long.to_csv(metrics_path, index=False)
    print(f"[INFO] 已保存 {metrics_path}")

    # summary
    print("[INFO] 生成 summary.csv ...")
    summ = summarize(metrics_long)
    summary_path = os.path.join(args.outdir, "summary.csv")
    summ.to_csv(summary_path, index=False)
    print(f"[INFO] 已保存 {summary_path}")

    # paired_deltas
    print("[INFO] 生成 paired_deltas.csv ...")
    deltas = paired_deltas(metrics_long)
    deltas_path = os.path.join(args.outdir, "paired_deltas.csv")
    deltas.to_csv(deltas_path, index=False)
    print(f"[INFO] 已保存 {deltas_path}")

    if not args.no_plots:

        from visualization import (
            plot_calibration_curve_one,
            plot_fig5_1_accuracy_panels,
            plot_fig5_3_clan_decomposition,
            plot_fig5_5_inference_tradeoffs,
            plot_fig5_6_coverage_diagnostics,
            plot_fig5_7_one_factor_sensitivity,
            plot_misspec_robustness,
        )

        print("[INFO] 生成报告图表（Figure 5 系列）...")

        dgps_order = [
            d for d in
            ["DGP0","DGP1","DGP2","DGP3","DGP4","DGP5","DGP6","DGP7","DGP8","RCT"]
            if d in dgp_names
        ]

        methods_order = [
            m for m in ["ORF", "CF", "DML-CF"]
            if m in args.methods
        ]

        n_ref = 1000 if 1000 in args.n_list else max(args.n_list)

        # Figure 5-1
        plot_fig5_1_accuracy_panels(
            summ, plots_dir,
            n_ref=n_ref,
            dgps_order=dgps_order,
            methods_order=methods_order
        )

        # Figure 5-3
        plot_fig5_3_clan_decomposition(
            summ, plots_dir,
            n_ref=n_ref,
            dgps_order=dgps_order,
            methods_order=methods_order
        )

        # Figure 5-5
        plot_fig5_5_inference_tradeoffs(
            summ, plots_dir,
            n_ref=n_ref,
            dgps_order=dgps_order,
            methods_order=methods_order
        )

        # Figure 5-6
        plot_fig5_6_coverage_diagnostics(
            summ, plots_dir,
            n_ref=n_ref,
            nominal=0.95,
            methods_order=methods_order
        )

        # Figure 5-7（默认基准 DGP0）
        if "DGP0" in dgp_names:
            plot_fig5_7_one_factor_sensitivity(
                summ, plots_dir,
                n_ref=n_ref,
                baseline_dgp="DGP0"
            )

        # Calibration curve
        calib_dgps = [d for d in ["DGP0","DGP3","DGP4","DGP5","DGP6","DGP7"] if d in dgp_names]
        if calib_dgps:
            for d in calib_dgps:
                plot_calibration_curve_one(
                    dgp=d,
                    n=n_ref,
                    pred_dir=preds_dir,  # 注意：pred_dir
                    outdir=plots_dir,  # 注意：outdir
                    K=20,
                    rep=1
                )

        # Misspec
        if args.diag_misspec:
            misspec_dgps = ["DGP0", "DGP3", "DGP4", "DGP7"]
            for dgp_ref in misspec_dgps:
                if dgp_ref in dgp_names:
                    plot_misspec_robustness(
                        metrics_long,
                        plots_dir,
                        dgp_ref=dgp_ref,
                        n_ref=n_ref
                    )

        print("[INFO] 图表生成完成。")

    print("[INFO] 全部流程完成。")


if __name__ == "__main__":
    main()
