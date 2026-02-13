# check_z.py
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def safe_name(s: str) -> str:
    s = str(s)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_box(
    sub: pd.DataFrame,
    value_col: str,
    methods_order: list,
    hline: float,
    title: str,
    ylabel: str,
    save_path: str
) -> None:
    """Generic boxplot saver."""
    data = []
    labels = []
    for m in methods_order:
        v = sub.loc[sub["method"] == m, value_col].dropna().values
        if len(v) == 0:
            continue
        data.append(v)
        labels.append(m)

    if len(data) == 0:
        print(f"[skip] empty plot: {title}")
        return

    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.axhline(hline, linestyle="--")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")


def main():
    csv_path = "results/metrics_long_all.csv"
    outdir = "results/z_diagnostics_all"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到 {csv_path}。请确认 metrics_long.csv 在当前目录，或把 csv_path 改成绝对路径。"
        )

    df = pd.read_csv(csv_path)

    required = {"dgp", "n", "method", "z_sd", "z_p975"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}. 现有列: {list(df.columns)}")

    outdir = ensure_outdir(outdir)

    preferred_methods = ["CF", "DML-CF", "ORF"]
    methods_in_data = list(df["method"].dropna().unique())
    methods_order = [m for m in preferred_methods if m in methods_in_data]

    for m in methods_in_data:
        if m not in methods_order:
            methods_order.append(m)


    dgps = sorted(df["dgp"].dropna().unique())
    ns = sorted(df["n"].dropna().unique())

    for dgp in dgps:
        dgp_safe = safe_name(dgp)
        dgp_dir = ensure_outdir(os.path.join(outdir, dgp_safe))

        for n in ns:
            sub = df[(df["dgp"] == dgp) & (df["n"] == n)].copy()
            if sub.empty:
                continue

            # 图1：z_sd
            save1 = os.path.join(dgp_dir, f"{dgp_safe}_n{int(n)}_z_sd_box.png")
            plot_box(
                sub=sub,
                value_col="z_sd",
                methods_order=methods_order,
                hline=1.0,
                title=f"z_sd by Method ({dgp}, n={int(n)})",
                ylabel="z_sd",
                save_path=save1
            )

            # z_p975
            save2 = os.path.join(dgp_dir, f"{dgp_safe}_n{int(n)}_z_p975_box.png")
            plot_box(
                sub=sub,
                value_col="z_p975",
                methods_order=methods_order,
                hline=1.96,
                title=f"z_p975 by Method ({dgp}, n={int(n)})",
                ylabel="z_p975",
                save_path=save2
            )

    print("\nDone. All plots saved under:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
