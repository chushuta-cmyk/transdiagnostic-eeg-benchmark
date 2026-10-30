#!/usr/bin/env python3
"""
Step 02: Multi-Cohort Biophysical & Topological Profiling.
Computes real empirical effect sizes (Cohen's d + 95% CI + Welch t-tests + Paired tests for repeated measures)
across all 4 cohorts:
  1. Sleep Deprivation (ds004902): Acute homeostatic sleep debt (SD vs. NS, paired N=71)
  2. Mumtaz Clinical MDD (4244171): Unmedicated clinical depression (MDD vs. HC, N=30 vs 28)
  3. Cavanagh Depression Rest (ds003478): Large-scale depression cohort (Depressed vs. Control, N=44 vs 75)
  4. UCSD Parkinson's Disease (ds002778): Dopaminergic pathology (PD-OFF vs. HC & PD-ON vs. HC)

Evaluates core biophysical markers:
  - Frontal Periodic Theta (4–8 Hz residual over Fp1, Fp2, F3, F4, Fz)
  - Posterior Periodic Alpha (8–13 Hz residual over P3, P4, Pz, O1, O2)
  - Global Aperiodic Exponent chi (1/f spectral roll-off indexing cortical E/I balance)
  - OMST Global Cost Efficiency (GCE = E_global - Cost)
  - Global Periodic Delta (1–4 Hz residual)
  - OMST Network Edge Density

Calculates Benjamini-Hochberg False Discovery Rate (FDR) q-values.
Outputs:
  reports/multicohort_biophysical_effect_sizes.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch
from statsmodels.stats.multitest import multipletests

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def cohens_d_ci(group1, group0):
    """Computes between-subject Cohen's d and 95% analytical confidence interval."""
    n1, n0 = len(group1), len(group0)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    m1, m0 = np.mean(group1), np.mean(group0)
    v1, v0 = np.var(group1, ddof=1), np.var(group0, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)) + 1e-8
    d = (m1 - m0) / pooled_sd

    # Welch t-test
    t_stat, p_val = stats.ttest_ind(group1, group0, equal_var=False)

    # Analytical SE approximation for Cohen's d
    se_d = np.sqrt((n1 + n0) / (n1 * n0) + (d ** 2) / (2 * (n1 + n0)))
    ci_lo = d - 1.96 * se_d
    ci_hi = d + 1.96 * se_d
    return float(d), float(ci_lo), float(ci_hi), float(t_stat), float(p_val)


def paired_cohens_dz(group1, group0):
    """Computes within-subject paired Cohen's d_z and paired t-test."""
    diff = np.array(group1) - np.array(group0)
    n = len(diff)
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1) + 1e-8
    dz = mean_diff / sd_diff
    t_stat, p_val = stats.ttest_rel(group1, group0)
    se_dz = np.sqrt(1.0 / n + (dz ** 2) / (2.0 * n))
    ci_lo = dz - 1.96 * se_dz
    ci_hi = dz + 1.96 * se_dz
    return float(dz), float(ci_lo), float(ci_hi), float(t_stat), float(p_val)


def extract_features_from_graphs(graph_list):
    """Extracts summary markers from graph instances."""
    frontal_idx = [CHANNELS_19.index(c) for c in ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']]
    posterior_idx = [CHANNELS_19.index(c) for c in ['P3', 'P4', 'Pz', 'O1', 'O2']]

    rows = []
    for g in graph_list:
        x = g['x'].numpy()  # [19, 4] -> 0: theta, 1: delta, 2: alpha, 3: chi
        adj = g['adj'].numpy()

        f_theta = float(np.mean(x[frontal_idx, 0]))
        p_alpha = float(np.mean(x[posterior_idx, 2]))
        g_chi = float(np.mean(x[:, 3]))
        g_delta = float(np.mean(x[:, 1]))
        gce = float(g.get('gce', 0.0))
        density = float(np.mean(adj))
        opt_m = int(g.get('omst_trees', 1))

        rows.append({
            'subject_id': g['subject_id'],
            'sub_name': g.get('sub_name', g.get('bids_id', g['subject_id'])),
            'cohort': g.get('cohort', ''),
            'group': g.get('group', ''),
            'condition': g.get('condition', ''),
            'label': int(g['y']),
            'frontal_theta': f_theta,
            'posterior_alpha': p_alpha,
            'global_chi': g_chi,
            'global_delta': g_delta,
            'omst_gce': gce,
            'omst_density': density,
            'omst_trees': opt_m
        })
    return pd.DataFrame(rows)


def main():
    print("📊 Computing Multi-Cohort Biophysical & Topological Effect Sizes...")

    cohort_files = {
        'Study 1: Sleep Deprivation (ds004902)': (DATA_DIR / "cohort_sleep_ds004902_graphs.pt", 1, 0, 'SD', 'NS', True),
        'Study 2: Mumtaz Clinical MDD (4244171)': (DATA_DIR / "cohort_mumtaz_mdd_graphs.pt", 1, 0, 'MDD', 'HC', False),
        'Study 3: Cavanagh Depression (ds003478)': (DATA_DIR / "cohort_cavanagh_mdd_graphs.pt", 1, 0, 'Depressed', 'Control', False),
        'Study 4a: UCSD Parkinson OFF (ds002778)': (DATA_DIR / "cohort_ucsd_parkinson_graphs.pt", 1, 0, 'PD_OFF', 'HC', False),
        'Study 4b: UCSD Parkinson ON (ds002778)': (DATA_DIR / "cohort_ucsd_parkinson_graphs.pt", 1, 0, 'PD_ON', 'HC', False),
    }

    markers = [
        ('frontal_theta', 'Frontal Periodic Theta (4–8 Hz)'),
        ('posterior_alpha', 'Posterior Periodic Alpha (8–13 Hz)'),
        ('global_chi', 'Global Aperiodic Exponent (χ)'),
        ('global_delta', 'Global Periodic Delta (1–4 Hz)'),
        ('omst_gce', 'OMST Global Cost Efficiency (GCE)'),
        ('omst_density', 'OMST Network Edge Density'),
    ]

    all_results = []

    for cohort_name, (fpath, c_label, r_label, c_grp, r_grp, is_paired) in cohort_files.items():
        if not fpath.exists():
            print(f"  ⚠️ Skipping {cohort_name}: File not found ({fpath.name})")
            continue

        graphs = torch.load(fpath, weights_only=False)
        df_feat = extract_features_from_graphs(graphs)

        # Filter case vs reference
        if 'group' in df_feat.columns and c_grp in df_feat['group'].values and r_grp in df_feat['group'].values:
            df_case = df_feat[df_feat['group'] == c_grp]
            df_ctrl = df_feat[df_feat['group'] == r_grp]
        else:
            df_case = df_feat[df_feat['label'] == c_label]
            df_ctrl = df_feat[df_feat['label'] == r_label]

        n_case, n_ctrl = len(df_case), len(df_ctrl)
        print(f"\n▶ {cohort_name} [Case={c_grp} (N={n_case}) vs. Reference={r_grp} (N={n_ctrl})]:")

        # If paired, align by sub_name
        paired_dict = {}
        if is_paired:
            common_subs = sorted(list(set(df_case['sub_name']).intersection(set(df_ctrl['sub_name']))))
            df_case_p = df_case.set_index('sub_name').loc[common_subs].reset_index()
            df_ctrl_p = df_ctrl.set_index('sub_name').loc[common_subs].reset_index()
        else:
            df_case_p, df_ctrl_p = None, None

        for m_key, m_label in markers:
            vals_case = df_case[m_key].values
            vals_ctrl = df_ctrl[m_key].values

            d, ci_lo, ci_hi, t_val, p_val = cohens_d_ci(vals_case, vals_ctrl)

            # If paired, also compute paired stats
            if is_paired and df_case_p is not None:
                dz, dz_lo, dz_hi, t_rel, p_rel = paired_cohens_dz(df_case_p[m_key].values, df_ctrl_p[m_key].values)
                print(f"  • {m_label:40s}: d = {d:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}], t = {t_val:+.2f}, p = {p_val:.4f} | Paired: dz = {dz:+.3f}, p = {p_rel:.4f}")
            else:
                dz, dz_lo, dz_hi, t_rel, p_rel = np.nan, np.nan, np.nan, np.nan, np.nan
                print(f"  • {m_label:40s}: d = {d:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}], t = {t_val:+.2f}, p = {p_val:.4f}")

            all_results.append({
                'cohort': cohort_name,
                'case_group': c_grp,
                'ref_group': r_grp,
                'n_case': n_case,
                'n_control': n_ctrl,
                'is_paired': is_paired,
                'marker_key': m_key,
                'marker_label': m_label,
                'mean_case': float(np.mean(vals_case)),
                'mean_control': float(np.mean(vals_ctrl)),
                'cohens_d': d,
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
                't_statistic': t_val,
                'p_value': p_val,
                'paired_dz': dz,
                'paired_dz_ci_lower': dz_lo,
                'paired_dz_ci_upper': dz_hi,
                'paired_t_statistic': t_rel,
                'paired_p_value': p_rel
            })

    df_out = pd.DataFrame(all_results)

    # Compute Benjamini-Hochberg FDR q-values across primary 4 focal markers (20 tests)
    focal_keys = ['frontal_theta', 'posterior_alpha', 'global_chi', 'omst_gce']
    focal_mask = df_out['marker_key'].isin(focal_keys)
    _, q_focal, _, _ = multipletests(df_out.loc[focal_mask, 'p_value'], alpha=0.05, method='fdr_bh')
    df_out.loc[focal_mask, 'q_value_focal'] = q_focal
    df_out.loc[focal_mask, 'fdr_sig_focal'] = q_focal < 0.05

    # Compute FDR across all 30 tests
    _, q_all, _, _ = multipletests(df_out['p_value'], alpha=0.05, method='fdr_bh')
    df_out['q_value_all'] = q_all
    df_out['fdr_sig_all'] = q_all < 0.05

    out_csv = REPORTS_DIR / "multicohort_biophysical_effect_sizes.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n✓ Saved multi-cohort effect size benchmark with FDR to {out_csv}")


if __name__ == "__main__":
    main()
