import pandas as pd
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).resolve().parent

    print("[INFO] BASE_DIR =", BASE_DIR)

    dgp_dirs = sorted([p for p in BASE_DIR.iterdir() if p.is_dir() and p.name.startswith("results_DGP")])

    print(f"[INFO] Found {len(dgp_dirs)} folders startswith('results_DGP'):")
    for p in dgp_dirs[:20]:
        print("  -", p.name)
    if len(dgp_dirs) > 20:
        print("  ...")

    all_dfs = []
    missing = []

    for dgp_dir in dgp_dirs:
        summary_path = dgp_dir / "summary.csv"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue

        df = pd.read_csv(summary_path)

        if "dgp" not in df.columns:
            df["dgp"] = dgp_dir.name.replace("results_", "")  # results_DGP0 -> DGP0
        df["dgp"] = df["dgp"].astype(str).str.replace("^results_", "", regex=True)

        all_dfs.append(df)
        print(f"[OK] loaded {summary_path}  shape={df.shape}")

    if len(missing) > 0:
        print(f"[WARN] Missing summary.csv in {len(missing)} folders, examples:")
        for x in missing[:10]:
            print("  -", x)
        if len(missing) > 10:
            print("  ...")

    if not all_dfs:
        print("\n[ERROR] No summary.csv files were found.")
        print("Check:")
        print("  1) Folder names start with 'results_DGP' ?")
        print("  2) summary file name is exactly 'summary.csv' ?")
        print("  3) summary.csv is directly under results_DGP*/ (not nested)?")
        raise SystemExit(1)

    summary_all = pd.concat(all_dfs, ignore_index=True)
    out_path = BASE_DIR / "summary_all.csv"
    summary_all.to_csv(out_path, index=False)

    print("\n[DONE] Merged summary saved to:", out_path)
    print("[DONE] shape:", summary_all.shape)
    print("[DONE] dgps:", sorted(summary_all["dgp"].unique()))

if __name__ == "__main__":
    main()
