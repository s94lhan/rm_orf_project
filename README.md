# Orthogonal Random Forest: Monte Carlo Simulation Study

This repository contains the full replication code for the Monte Carlo study comparing ORF, CF, and DML-CF under varying identification challenges.

The project is modularized to ensure clarity, reproducibility, and extensibility.

---

# 1. Computational Environment

Experiments were conducted in:

- Python 3.10.4

Required libraries:

- econml (ORF, CF, DML-CF implementations)
- scikit-learn
- numpy
- scipy
- pandas
- matplotlib
- joblib
- tqdm

Install dependencies using:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas scikit-learn joblib tqdm matplotlib econml
```

# 2. Project Structure
```
rm_orf_project/
│
├── main.py                # Monte Carlo experiment entry point
├── config.py              # DGP configurations and global settings
├── dgp.py                 # Data-generating processes
├── models.py              # ORF, CF, DML-CF implementations
├── metrics.py             # Evaluation metrics
├── simulation.py          # Monte Carlo execution engine
├── merge_metrics.py       # Merge metrics across DGP folders
├── merge_summaries.py     # Merge summary tables
├── visualization.py       # Generates all main figures
├── check_z.py             # Z-statistic calibration diagnostics
├── mcs_test.py            # Model Confidence Set (MCS) analysis
```

# 3. How to Reproduce Results
Step 1: Run Simulations for a Single DGP

Example command:
```bash
python main.py --outdir results_DGP0 --seed 123 --R 30 --dgps DGP0 --methods ORF CF DML-CF --n_list 250 500 1000 --N_test 500 --diag_misspec --n_jobs 10
```
This generates result folders such as:
results_DGP0
Repeat for other DGPs as needed.


Step 2: Merge Results Across DGPs
To obtain cross-DGP result files:
```bash
python merge_metrics.py
python merge_summaries.py
```
This produces:
metrics_long_all.csv
summary_all.csv

Step 3: Generate Figures
(A) Figures Generated per DGP (No Merge Required)
Figure 5-2: CATE calibration plots
Figure 5-9: Misspecification diagnostics
These are generated automatically after running main.py.
Automatically generated after running the corresponding DGP.

(B) Z-Statistic Calibration Diagnostics
Generate boxplots of Z-statistics
Make sure you are in the rm_orf_project directory::
```bash
cd rm_orf_project
python check_z.py
```
Outputs are saved in:
z_diagnostics_all

(C) Figures Based on Merged Results
The following figures use summary_all.csv:
Figure 5-1
Figure 5-3
Figure 5-5
Figure 5-6
Figure 5-7
Figure 5-8
Example reproduction command:
```bash
python -c "from visualization import plot_fig5_1_from_merged_csv; plot_fig5_1_from_merged_csv('summary_all.csv','plots_report',1000)"
python -c "from visualization import plot_fig5_3_from_merged_csv; plot_fig5_3_from_merged_csv('summary_all.csv','plots_report',1000)"
python -c "from visualization import plot_fig5_5_from_merged_csv; plot_fig5_5_from_merged_csv('summary_all.csv','plots_report',1000)"
python -c "from visualization import plot_fig5_6_from_merged_csv; plot_fig5_6_from_merged_csv('summary_all.csv','plots_report',1000)"
python -c "from visualization import plot_fig5_7_from_merged_csv; plot_fig5_7_from_merged_csv('summary_all.csv','plots_report',1000)"
python -c "from visualization import plot_fig5_8_from_merged_csv; plot_fig5_8_from_merged_csv('summary_all.csv','plots_report',1000)"
```
All figures are saved in:
plots_report/

Step 4: Model Confidence Set (MCS) Analysis
(A) RMSE-based MCS
```bash
python mcs_test.py --in_csv metrics_long_all.csv --out_csv mcs_n1000_rmse.csv --n 1000 --metric rmse --B 2000 --alpha 0.05
```

(B) Coverage-deviation-based MCS
```bash
python mcs_test.py --in_csv metrics_long_all.csv --out_csv mcs_n1000_covdev.csv --n 1000 --metric pointwise_cover_rate --loss_transform abs_dev_from_nominal --nominal 0.95 --B 2000 --alpha 0.05
```

# 4. Design Overview
The data-generating processes follow the potential outcomes framework with additive outcome structure and propensity-score-based treatment assignment. Five design dimensions are varied:
Dimensionality
Overlap quality
Alignment between assignment and outcome mechanisms
Heterogeneity strength and structure
Noise level
This allows systematic evaluation of estimation accuracy and inferential robustness under varying identification difficulty.

# 5. Notes
All simulations use paired Monte Carlo design.
All confounders are observed.
Results are fully reproducible under the specified environment.
