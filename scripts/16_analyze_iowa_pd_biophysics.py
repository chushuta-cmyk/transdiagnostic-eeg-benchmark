#!/usr/bin/env python3
"""
Step 16: Statistical Biophysical Profiling & Clinical Scale Association
for the Iowa Parkinson's Disease Cohort (ds004584, N=149).
- Diagnostic contrast: PD (N=100) vs Control (N=49)
- Cognitive correlation: MoCA (N=149)
- Motor symptom correlation: UPDRS Part-III (N=100 PD)
Saves results to reports/iowa_pd_ds004584_biophysical_profiling.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

BENCHMARK_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/transdiagnostic_eeg_benchmark")
DATA_PATH = BENCHMARK_ROOT / "data" / "derivatives" / "cohort_iowa_pd_ds004584_graphs.pt"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

data = torch.load(DATA_PATH)
print(f"Loaded {len(data)} participants from {DATA_PATH.name}")

# Standard 19 channel order:
# 0: Fp1, 1: Fp2, 2: F7, 3: F3, 4: Fz, 5: F4, 6: F8,
# 7: T7, 8: C3, 9: Cz, 10: C4, 11: T8,
# 12: P7, 13: P3, 14: Pz, 15: P4, 16: P8,
# 17: O1, 18: O2

records = []
for d in data:
    x = d["x"].numpy()  # [19, 4]: col 0: Theta, col 1: Delta, col 2: Alpha, col 3: Chi
    theta_global = np.mean(x[:, 0])
    delta_global = np.mean(x[:, 1])
    alpha_global = np.mean(x[:, 2])
    chi_global = np.mean(x[:, 3])

    frontal_theta = np.mean(x[[0, 1, 3, 4, 5], 0])
    frontal_delta = np.mean(x[[0, 1, 3, 4, 5], 1])
    central_beta_surrogate = np.mean(x[[8, 9, 10], 3])  # chi at central channels
    posterior_alpha = np.mean(x[[13, 14, 15, 17, 18], 2])

    records.append({
        "subject_id": d["subject_id"],
        "group": d["group"],
        "age": d["age"],
        "gender": d["gender"],
        "moca": d["moca"],
        "updrs": d["updrs"],
        "delta_global": delta_global,
        "theta_global": theta_global,
        "alpha_global": alpha_global,
        "chi_global": chi_global,
        "frontal_theta": frontal_theta,
        "frontal_delta": frontal_delta,
        "posterior_alpha": posterior_alpha,
        "omst_trees": d["omst_trees"],
        "gce": d["gce"],
        "clean_duration_s": d["clean_duration_s"]
    })

df = pd.DataFrame(records)

print("\n" + "=" * 75)
print("📊 [Iowa PD ds004584] Demographics & Clinical Characteristics:")
print(f"Total: N = {len(df)}")
for grp, grp_df in df.groupby("group"):
    moca_str = f"{grp_df['moca'].mean():.1f} ± {grp_df['moca'].std():.1f}"
    updrs_str = f"{grp_df['updrs'].mean():.1f} ± {grp_df['updrs'].std():.1f}" if grp == "PD" else "N/A"
    print(f"  {grp} (N={len(grp_df)}): Age = {grp_df['age'].mean():.1f} ± {grp_df['age'].std():.1f}, MoCA = {moca_str}, UPDRS = {updrs_str}")

# Diagnostic Contrasts: PD vs Control
metrics = [
    ("Delta Power (Global)", "delta_global"),
    ("Theta Power (Global)", "theta_global"),
    ("Alpha Power (Global)", "alpha_global"),
    ("Aperiodic Exponent Chi (Global)", "chi_global"),
    ("Frontal Delta", "frontal_delta"),
    ("Frontal Theta", "frontal_theta"),
    ("Posterior Alpha", "posterior_alpha"),
    ("OMST Optimal Trees m*", "omst_trees"),
    ("Global Cost Efficiency (GCE)", "gce"),
]

contrast_results = []
print("\n" + "=" * 75)
print("🔬 Group Contrasts: PD (N=100) vs Healthy Control (N=49)")
print(f"{'Metric':<32} | {'PD Mean (SD)':<18} | {'CTL Mean (SD)':<18} | {'Cohen d':<8} | {'t-stat':<8} | {'p-val':<10}")
print("-" * 105)

pd_mask = df["group"] == "PD"
ctl_mask = df["group"] == "Control"

for name, col in metrics:
    v_pd = df.loc[pd_mask, col].dropna().values
    v_ctl = df.loc[ctl_mask, col].dropna().values

    m_pd, s_pd = np.mean(v_pd), np.std(v_pd, ddof=1)
    m_ctl, s_ctl = np.mean(v_ctl), np.std(v_ctl, ddof=1)

    # Pooled Cohen's d
    s_pool = np.sqrt(((len(v_pd) - 1) * s_pd**2 + (len(v_ctl) - 1) * s_ctl**2) / (len(v_pd) + len(v_ctl) - 2))
    d = (m_pd - m_ctl) / s_pool if s_pool > 0 else 0.0

    t_stat, p_val = stats.ttest_ind(v_pd, v_ctl, equal_var=False)

    print(f"{name:<32} | {m_pd:7.4f} ({s_pd:6.4f}) | {m_ctl:7.4f} ({s_ctl:6.4f}) | {d:+7.3f} | {t_stat:+7.3f} | {p_val:.4e}")

    contrast_results.append({
        "contrast": "PD_vs_Control",
        "metric": name,
        "column": col,
        "n_pd": len(v_pd),
        "mean_pd": m_pd,
        "sd_pd": s_pd,
        "n_ctl": len(v_ctl),
        "mean_ctl": m_ctl,
        "sd_ctl": s_ctl,
        "cohens_d": d,
        "t_stat": t_stat,
        "p_val": p_val
    })

# Clinical Correlations: MoCA (All N=149) and UPDRS (PD N=100)
print("\n" + "=" * 75)
print("📈 Clinical Correlations:")
print(f"{'Scale':<12} | {'Metric':<30} | {'Pearson r':<10} | {'p-val':<10} | {'Spearman rho':<12} | {'p-val':<10}")
print("-" * 85)

for scale_name, scale_col, sub_df in [("MoCA (All)", "moca", df), ("UPDRS (PD)", "updrs", df[df["group"] == "PD"])]:
    for name, col in metrics:
        valid = sub_df[[scale_col, col]].dropna()
        if len(valid) < 10:
            continue
        r, pr = stats.pearsonr(valid[scale_col], valid[col])
        rho, prho = stats.spearmanr(valid[scale_col], valid[col])

        print(f"{scale_name:<12} | {name:<30} | {r:+8.4f} | {pr:.4e} | {rho:+10.4f} | {prho:.4e}")

        contrast_results.append({
            "contrast": f"Corr_{scale_name}",
            "metric": name,
            "column": col,
            "n_pd": len(valid),
            "mean_pd": np.nan,
            "sd_pd": np.nan,
            "n_ctl": np.nan,
            "mean_ctl": np.nan,
            "sd_ctl": np.nan,
            "cohens_d": r,  # store r in cohens_d col for unified table
            "t_stat": rho,  # store rho in t_stat col
            "p_val": pr
        })

df_out = pd.DataFrame(contrast_results)
csv_path = REPORTS_DIR / "iowa_pd_ds004584_biophysical_profiling.csv"
df_out.to_csv(csv_path, index=False)
print(f"\nSaved biophysical and clinical report to: {csv_path}")
