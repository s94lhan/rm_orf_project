
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=None,
                    help="Base directory containing results_DGP*/ folders. Default: directory of this script.")
    ap.add_argument("--pattern", type=str, default="results/results_DGP*/metrics_long.csv",
                    help="Glob pattern under base_dir to find metrics_long.csv files.")
    ap.add_argument("--out", type=str, default="results/metrics_long_all.csv",
                    help="Output merged csv filename (saved under base_dir).")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    base = Path(args.base_dir).resolve() if args.base_dir else script_dir

    files = sorted(base.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' under base_dir: {base}\n"
            f"Tip: check you are pointing base_dir to the folder that contains results_DGP0/, results_DGP1/, ...\n"
            f"Tip: you can run: python merge_metrics.py --base_dir <path_to_rm_orf_project>"
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        # infer DGP name from folder (e.g., results_DGP3 -> DGP3) if needed
        if "dgp" not in df.columns:
            parent = f.parent.name  # results_DGP3
            dgp = parent.replace("results_", "")  # DGP3
            df["dgp"] = dgp

        dfs.append(df)

    all_metrics = pd.concat(dfs, ignore_index=True)
    out_path = base / args.out
    all_metrics.to_csv(out_path, index=False)

    print("[DONE] merged:", len(files), "files")
    print(" ->", out_path)

if __name__ == "__main__":
    main()
