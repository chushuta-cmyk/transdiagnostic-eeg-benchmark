import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

BENCHMARK_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/transdiagnostic_eeg_benchmark")
DATA_PATH = BENCHMARK_ROOT / "data" / "derivatives" / "cohort_unm_pd_ds003490_graphs.pt"
REPORTS_DIR = BENCHMARK_ROOT / "reports"

data = torch.load(DATA_PATH)
print(f"Loaded {len(data)} sessions from {DATA_PATH.name}")

records = []
for d in data:
    x = d['x'].numpy() # 19 x 4
    delta_mean = np.mean(x[:, 0])
    theta_mean = np.mean(x[:, 1])
    alpha_mean = np.mean(x[:, 2])
    beta_mean = np.mean(x[:, 3])
    
    # Frontal vs Central Beta
    # Frontal: Fp1(0), Fp2(1), F7(2), F3(3), Fz(4), F4(5), F8(6)
    # Central/Motor: C3(8), Cz(9), C4(10)
    central_beta = np.mean(x[[8, 9, 10], 3])
    frontal_theta = np.mean(x[[0, 1, 3, 4, 5], 1])
    
    records.append({
        'subject_id': d['subject_id'],
        'group': d['group'],
        'med_state': d['med_state'],
        'session': d['session'],
        'age': d['age'],
        'sex': d['sex'],
        'delta_global': delta_mean,
        'theta_global': theta_mean,
        'alpha_global': alpha_mean,
        'beta_global': beta_mean,
        'central_beta': central_beta,
        'frontal_theta': frontal_theta,
        'omst_trees': d['omst_trees'],
        'gce': d['gce'],
        'clean_duration_s': d['clean_duration_s']
    })

df = pd.DataFrame(records)

print("\n--- Summary by State ---")
print(df.groupby('med_state')[['delta_global', 'theta_global', 'alpha_global', 'beta_global', 'central_beta', 'gce']].mean())

# 1. Within-subject paired comparison: PD-OFF vs PD-ON (N = 25 pairs)
df_pd = df[df['group'] == 'PD']
pd_subs = df_pd['subject_id'].unique()

paired_records = []
for sub in pd_subs:
    sub_df = df_pd[df_pd['subject_id'] == sub]
    on_row = sub_df[sub_df['med_state'] == 'ON']
    off_row = sub_df[sub_df['med_state'] == 'OFF']
    if len(on_row) == 1 and len(off_row) == 1:
        paired_records.append({
            'sub_id': sub,
            'delta_diff': on_row['delta_global'].values[0] - off_row['delta_global'].values[0],
            'theta_diff': on_row['theta_global'].values[0] - off_row['theta_global'].values[0],
            'alpha_diff': on_row['alpha_global'].values[0] - off_row['alpha_global'].values[0],
            'beta_diff': on_row['beta_global'].values[0] - off_row['beta_global'].values[0],
            'central_beta_diff': on_row['central_beta'].values[0] - off_row['central_beta'].values[0],
            'gce_diff': on_row['gce'].values[0] - off_row['gce'].values[0],
            # Absolute values
            'beta_off': off_row['beta_global'].values[0],
            'beta_on': on_row['beta_global'].values[0],
            'cbeta_off': off_row['central_beta'].values[0],
            'cbeta_on': on_row['central_beta'].values[0],
            'theta_off': off_row['theta_global'].values[0],
            'theta_on': on_row['theta_global'].values[0],
            'gce_off': off_row['gce'].values[0],
            'gce_on': on_row['gce'].values[0],
        })

df_paired = pd.DataFrame(paired_records)
print(f"\n--- Within-Subject Paired Test: PD-ON vs PD-OFF (N = {len(df_paired)} pairs) ---")
paired_results = []
for feat in ['delta', 'theta', 'alpha', 'beta', 'central_beta', 'gce']:
    diffs = df_paired[f'{feat}_diff'].values
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    d_z = mean_diff / (sd_diff + 1e-8)
    t_val, p_val = stats.ttest_rel(df_paired[f'{feat}_on'].values if f'{feat}_on' in df_paired else diffs, 
                                   df_paired[f'{feat}_off'].values if f'{feat}_off' in df_paired else np.zeros_like(diffs))
    w_stat, w_pval = stats.wilcoxon(diffs)
    paired_results.append({
        'Feature': feat,
        'Mean_Diff_ON_minus_OFF': mean_diff,
        'Cohen_dz': d_z,
        't_stat': t_val,
        'p_val': p_val,
        'wilcoxon_p': w_pval
    })
    print(f"{feat:16s}: Diff={mean_diff:+.4f}, dz={d_z:+.3f}, t={t_val:+.2f}, paired_p={p_val:.4f}, wilcoxon_p={w_pval:.4f}")

df_paired_stats = pd.DataFrame(paired_results)
out_csv = REPORTS_DIR / "unm_pd_ds003490_on_off_paired_biophysics.csv"
df_paired_stats.to_csv(out_csv, index=False)
print(f"\nSaved paired results to {out_csv}")

# 2. Cross-sectional: PD-OFF (n=25) vs CTL (n=25)
df_off = df[df['med_state'] == 'OFF']
df_ctl = df[df['med_state'] == 'CTL']

print("\n--- Cross-Sectional: PD-OFF (N=25) vs Healthy Controls (N=25) ---")
group_results = []
for feat in ['delta_global', 'theta_global', 'alpha_global', 'beta_global', 'central_beta', 'frontal_theta', 'gce']:
    v1 = df_off[feat].values
    v2 = df_ctl[feat].values
    m1, m2 = np.mean(v1), np.mean(v2)
    s_p = np.sqrt(((len(v1)-1)*np.var(v1, ddof=1) + (len(v2)-1)*np.var(v2, ddof=1)) / (len(v1)+len(v2)-2))
    d = (m1 - m2) / (s_p + 1e-8)
    t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
    group_results.append({
        'Feature': feat,
        'Mean_PD_OFF': m1,
        'Mean_CTL': m2,
        'Cohen_d': d,
        't_stat': t_stat,
        'p_val': p_val
    })
    print(f"{feat:16s}: PD-OFF={m1:.4f}, CTL={m2:.4f}, d={d:+.3f}, t={t_stat:.2f}, p={p_val:.4f}")

out_grp_csv = REPORTS_DIR / "unm_pd_ds003490_group_contrast_biophysics.csv"
pd.DataFrame(group_results).to_csv(out_grp_csv, index=False)
print(f"Saved group contrast results to {out_grp_csv}")
