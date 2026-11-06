import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

BENCHMARK_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/transdiagnostic_eeg_benchmark")
DATA_PATH = BENCHMARK_ROOT / "data" / "derivatives" / "cohort_dementia_ds004504_graphs.pt"
REPORTS_DIR = BENCHMARK_ROOT / "reports"

data = torch.load(DATA_PATH)
print(f"Loaded {len(data)} subjects from {DATA_PATH.name}")

# Node features: 19 channels x 4 features (0: Delta, 1: Theta, 2: Alpha, 3: Beta)
# Groups: 'A' (AD, n=36), 'F' (FTD, n=23), 'C' (Control, n=29)

records = []
for d in data:
    x = d['x'].numpy() # 19 x 4
    # Compute global mean across 19 channels
    delta_mean = np.mean(x[:, 0])
    theta_mean = np.mean(x[:, 1])
    alpha_mean = np.mean(x[:, 2])
    beta_mean = np.mean(x[:, 3])
    # Also frontal and posterior specific
    # Standard 19 ch indices: Fp1(0), Fp2(1), F7(2), F3(3), Fz(4), F4(5), F8(6), T3(7), C3(8), Cz(9), C4(10), T4(11), T5(12), P3(13), Pz(14), P4(15), T6(16), O1(17), O2(18)
    frontal_theta = np.mean(x[[0, 1, 3, 4, 5], 1])
    posterior_alpha = np.mean(x[[13, 14, 15, 17, 18], 2])
    
    # Slowing ratio: (Delta + Theta) / (Alpha + Beta)
    slowing_ratio = (delta_mean + theta_mean) / (alpha_mean + beta_mean + 1e-8)
    
    records.append({
        'subject_id': d['subject_id'],
        'group_code': d['group_code'],
        'diagnosis': d['diagnosis'],
        'mmse': d['mmse'],
        'age': d['age'],
        'gender': d['gender'],
        'delta_global': delta_mean,
        'theta_global': theta_mean,
        'alpha_global': alpha_mean,
        'beta_global': beta_mean,
        'frontal_theta': frontal_theta,
        'posterior_alpha': posterior_alpha,
        'slowing_ratio': slowing_ratio,
        'omst_trees': d['omst_trees'],
        'gce': d['gce'],
        'clean_duration_s': d['clean_duration_s']
    })

df = pd.DataFrame(records)
print("\n--- Summary by Clinical Group ---")
summary = df.groupby('diagnosis')[['mmse', 'delta_global', 'theta_global', 'alpha_global', 'beta_global', 'slowing_ratio', 'gce']].mean()
print(summary)

# Statistical comparison: Dementia (AD+FTD, n=59) vs Control (n=29)
dem = df[df['diagnosis'].isin(['Alzheimer', 'FTD'])]
ctl = df[df['diagnosis'] == 'Control']

print("\n--- Statistical Comparison: Dementia (N=59) vs Healthy Control (N=29) ---")
stat_rows = []
for feat in ['delta_global', 'theta_global', 'alpha_global', 'beta_global', 'frontal_theta', 'posterior_alpha', 'slowing_ratio', 'gce']:
    v_dem = dem[feat].values
    v_ctl = ctl[feat].values
    
    mean_dem = np.mean(v_dem)
    mean_ctl = np.mean(v_ctl)
    diff = mean_dem - mean_ctl
    
    # Pooled SD for Cohen's d
    s_pool = np.sqrt(((len(v_dem)-1)*np.var(v_dem, ddof=1) + (len(v_ctl)-1)*np.var(v_ctl, ddof=1)) / (len(v_dem) + len(v_ctl) - 2))
    cohen_d = diff / (s_pool + 1e-8)
    
    t_stat, p_val = stats.ttest_ind(v_dem, v_ctl, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(v_dem, v_ctl, alternative='two-sided')
    
    # Correlation with MMSE across all subjects with valid MMSE
    df_valid = df.dropna(subset=['mmse', feat])
    r_mmse, p_mmse = stats.pearsonr(df_valid[feat], df_valid['mmse'])
    
    stat_rows.append({
        'Feature': feat,
        'Mean_Dementia': mean_dem,
        'Mean_Control': mean_ctl,
        'Diff': diff,
        'Cohen_d': cohen_d,
        't_stat': t_stat,
        'p_val': p_val,
        'MannWhitney_p': u_pval,
        'r_MMSE': r_mmse,
        'p_MMSE': p_mmse
    })
    print(f"{feat:16s}: Dem={mean_dem:.4f}, Ctl={mean_ctl:.4f}, d={cohen_d:+.3f}, t={t_stat:.2f}, p={p_val:.2e}, r_MMSE={r_mmse:+.3f} (p={p_mmse:.2e})")

df_stats = pd.DataFrame(stat_rows)
out_csv = REPORTS_DIR / "dementia_ds004504_biophysical_profiling.csv"
df_stats.to_csv(out_csv, index=False)
print(f"\nSaved statistical results to {out_csv}")
