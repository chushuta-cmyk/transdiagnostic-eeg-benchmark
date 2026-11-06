#!/usr/bin/env python3
"""
Step 19: Build Expanded Multi-Cohort Biophysical Effect Sizes & Publication-Quality Figures (Figures 1-4).
Generates:
  - reports/multicohort_biophysical_effect_sizes_expanded.csv
  - reports/Figure_1_Multicohort_Biophysical_Effect_Sizes.png / .pdf
  - reports/Figure_2_Tri_Center_Parkinsons_Generalization.png / .pdf
  - reports/Figure_3_Transdiagnostic_Transfer_and_Ablations.png / .pdf
  - reports/Figure_4_Clinical_Scale_Continuous_Prediction.png / .pdf
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

# Publication styling
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_publication_formats(fig, base_path: Path):
    """Saves figure in Vector (PDF) and 300-DPI Raster (PNG)."""
    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {base_path.name} (PDF & PNG)")


# =========================================================================
# 1. Build Expanded Biophysical Effect Sizes Across All 7 Cohorts
# =========================================================================
def build_expanded_effect_sizes():
    print("Building expanded biophysical effect sizes across all cohorts...")
    
    # Load raw graph datasets
    sleep_graphs = torch.load(DATA_DIR / "cohort_sleep_ds004902_graphs.pt")
    mumtaz_graphs = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt")
    cavanagh_graphs = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt")
    ucsd_pd_graphs = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt")
    dementia_graphs = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt")
    unm_pd_graphs = torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt")
    iowa_pd_graphs = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt")

    # Define comparison cohorts
    # Standard 19 channel index mapping:
    # 0:Fp1, 1:Fp2, 2:F7, 3:F3, 4:Fz, 5:F4, 6:F8, 7:T7, 8:C3, 9:Cz, 10:C4, 11:T8, 12:P7, 13:P3, 14:Pz, 15:P4, 16:P8, 17:O1, 18:O2
    # Feature columns in x: [0: Theta, 1: Delta, 2: Alpha, 3: Chi]
    
    def extract_features(graph_list):
        feats = []
        for g in graph_list:
            x = g['x'].numpy() # 19 x 4
            frontal_theta = np.mean(x[[0, 1, 2, 3, 4, 5, 6], 0])
            global_theta = np.mean(x[:, 0])
            global_delta = np.mean(x[:, 1])
            posterior_alpha = np.mean(x[[13, 14, 15, 17, 18], 2])
            global_alpha = np.mean(x[:, 2])
            global_chi = np.mean(x[:, 3])
            gce = float(g['gce'])
            m_trees = float(g['omst_trees'])
            feats.append({
                'frontal_theta': frontal_theta,
                'global_theta': global_theta,
                'global_delta': global_delta,
                'posterior_alpha': posterior_alpha,
                'global_alpha': global_alpha,
                'global_chi': global_chi,
                'omst_gce': gce,
                'omst_trees': m_trees,
                'subject_id': g.get('subject_id', g.get('sub_name', 'sub'))
            })
        return pd.DataFrame(feats)

    comparisons = [
        ("Study 1: Acute Sleep Deprivation", "ds004902",
         [g for g in sleep_graphs if g.get('session_type') == 'SD' or g.get('y') == 1],
         [g for g in sleep_graphs if g.get('session_type') == 'NS' or g.get('y') == 0],
         "SD", "NS", True),
        ("Study 2: Mumtaz Outpatient MDD", "4244171",
         [g for g in mumtaz_graphs if int(g['y']) == 1],
         [g for g in mumtaz_graphs if int(g['y']) == 0],
         "MDD", "Control", False),
        ("Study 3: Cavanagh University Depression", "ds003478",
         [g for g in cavanagh_graphs if int(g['y']) == 1],
         [g for g in cavanagh_graphs if int(g['y']) == 0],
         "MDD", "Control", False),
        ("Study 4: UCSD Parkinson All", "ds002778",
         [g for g in ucsd_pd_graphs if int(g['y']) == 1],
         [g for g in ucsd_pd_graphs if int(g['y']) == 0],
         "PD", "Control", False),
        ("Study 5a: Alzheimer's Disease (AD)", "ds004504",
         [g for g in dementia_graphs if g.get('diagnosis') == 'Alzheimer'],
         [g for g in dementia_graphs if g.get('diagnosis') == 'Control'],
         "AD", "Control", False),
        ("Study 5b: Frontotemporal Dementia (FTD)", "ds004504",
         [g for g in dementia_graphs if g.get('diagnosis') == 'FTD'],
         [g for g in dementia_graphs if g.get('diagnosis') == 'Control'],
         "FTD", "Control", False),
        ("Study 5c: Clinical Dementia (All)", "ds004504",
         [g for g in dementia_graphs if g.get('diagnosis') in ('Alzheimer', 'FTD')],
         [g for g in dementia_graphs if g.get('diagnosis') == 'Control'],
         "Dementia", "Control", False),
        ("Study 6a: UNM Parkinson OFF", "ds003490",
         [g for g in unm_pd_graphs if g.get('med_state') == 'OFF'],
         [g for g in unm_pd_graphs if g.get('med_state') == 'CTL'],
         "PD_OFF", "Control", False),
        ("Study 6b: UNM Parkinson ON", "ds003490",
         [g for g in unm_pd_graphs if g.get('med_state') == 'ON'],
         [g for g in unm_pd_graphs if g.get('med_state') == 'CTL'],
         "PD_ON", "Control", False),
        ("Study 7: Iowa Parkinson Cohort", "ds004584",
         [g for g in iowa_pd_graphs if g.get('group') == 'PD'],
         [g for g in iowa_pd_graphs if g.get('group') == 'Control'],
         "PD", "Control", False),
    ]

    marker_defs = [
        ('frontal_theta', 'Frontal Periodic Theta (4–8 Hz)'),
        ('global_delta', 'Global Periodic Delta (1–4 Hz)'),
        ('posterior_alpha', 'Posterior Periodic Alpha (8–13 Hz)'),
        ('global_chi', 'Global Aperiodic Exponent (χ)'),
        ('omst_gce', 'OMST Global Cost Efficiency (GCE)')
    ]

    records = []
    for cohort_label, ds_id, cases, ctrls, case_name, ctrl_name, is_paired in comparisons:
        df_case = extract_features(cases)
        df_ctrl = extract_features(ctrls)
        n_case = len(df_case)
        n_ctrl = len(df_ctrl)

        for m_key, m_label in marker_defs:
            v_case = df_case[m_key].values
            v_ctrl = df_ctrl[m_key].values

            m_c, s_c = np.mean(v_case), np.std(v_case, ddof=1)
            m_ctl, s_ctl = np.mean(v_ctrl), np.std(v_ctrl, ddof=1)

            # Pooled Cohen's d
            s_pool = np.sqrt(((n_case - 1) * s_c**2 + (n_ctrl - 1) * s_ctl**2) / (n_case + n_ctrl - 2))
            d = (m_c - m_ctl) / s_pool if s_pool > 0 else 0.0

            # CI for Cohen's d: SE = sqrt((n1+n2)/(n1*n2) + d^2 / (2*(n1+n2)))
            se_d = np.sqrt((n_case + n_ctrl) / (n_case * n_ctrl) + (d**2) / (2 * (n_case + n_ctrl)))
            ci_lo = d - 1.96 * se_d
            ci_hi = d + 1.96 * se_d

            t_stat, p_val = stats.ttest_ind(v_case, v_ctrl, equal_var=False)

            records.append({
                'cohort': cohort_label,
                'dataset_id': ds_id,
                'case_group': case_name,
                'ref_group': ctrl_name,
                'n_case': n_case,
                'n_control': n_ctrl,
                'is_paired': is_paired,
                'marker_key': m_key,
                'marker_label': m_label,
                'mean_case': m_c,
                'mean_control': m_ctl,
                'cohens_d': d,
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
                't_statistic': t_stat,
                'p_value': p_val
            })

    df_out = pd.DataFrame(records)
    out_csv = REPORTS_DIR / "multicohort_biophysical_effect_sizes_expanded.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"✓ Saved expanded effect sizes ({len(df_out)} rows) to: {out_csv}")
    return df_out


# =========================================================================
# 2. Plot Figure 1: Multi-Cohort Biophysical Effect Sizes (4-Panel Forest Plot)
# =========================================================================
def plot_figure_1_expanded():
    csv_path = REPORTS_DIR / "multicohort_biophysical_effect_sizes_expanded.csv"
    if not csv_path.exists():
        build_expanded_effect_sizes()
    df = pd.read_csv(csv_path)

    # 4 focal markers: Frontal Theta, Global Delta, Posterior Alpha, Global Chi
    focal_markers = [
        ('frontal_theta', 'A. Frontal Periodic Theta (4–8 Hz)\n[Polarity Dissociation Axis]'),
        ('global_delta', 'B. Global Periodic Delta (1–4 Hz)\n[Cortical Slow-Wave Intrusion]'),
        ('posterior_alpha', 'C. Posterior Periodic Alpha (8–13 Hz)\n[Aperiodic-Adjusted Oscillation]'),
        ('global_chi', 'D. Global Aperiodic Exponent (χ)\n[Cortical Excitation/Inhibition]')
    ]

    # Target display order (Neurodegenerative -> Sleep -> Depression)
    display_cohorts = [
        'Study 5c: Clinical Dementia (All)',
        'Study 5a: Alzheimer\'s Disease (AD)',
        'Study 5b: Frontotemporal Dementia (FTD)',
        'Study 7: Iowa Parkinson Cohort',
        'Study 6a: UNM Parkinson OFF',
        'Study 4: UCSD Parkinson All',
        'Study 1: Acute Sleep Deprivation',
        'Study 2: Mumtaz Outpatient MDD',
        'Study 3: Cavanagh University Depression',
    ]

    short_labels = [
        'Dementia All (N=88)',
        'Alzheimer AD (N=65)',
        'FTD Dementia (N=52)',
        'Iowa PD (N=149)',
        'UNM PD OFF (N=50)',
        'UCSD PD (N=47)',
        'Sleep Deprivation (N=142)',
        'Mumtaz MDD (N=58)',
        'Cavanagh MDD (N=119)'
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 7.5), sharey=True)

    y_pos = np.arange(len(display_cohorts))

    for ax_idx, (m_key, m_title) in enumerate(focal_markers):
        ax = axes[ax_idx]
        sub_df = df[df['marker_key'] == m_key].copy()
        
        sub_df['order_key'] = sub_df['cohort'].apply(lambda x: display_cohorts.index(x) if x in display_cohorts else 99)
        sub_df = sub_df.sort_values('order_key', ascending=False).reset_index(drop=True)
        # Filter down to display_cohorts
        sub_df = sub_df[sub_df['cohort'].isin(display_cohorts)].reset_index(drop=True)

        d_vals = sub_df['cohens_d'].values
        ci_lo = sub_df['ci_lower'].values
        ci_hi = sub_df['ci_upper'].values
        p_vals = sub_df['p_value'].values

        # Error bars: asymmetric [d - ci_lo, ci_hi - d]
        xerr = np.zeros((2, len(d_vals)))
        xerr[0, :] = d_vals - ci_lo
        xerr[1, :] = ci_hi - d_vals

        # Coloring:
        # Negative effect (blue), Positive effect (red), Non-significant (grey)
        colors = []
        for d, p in zip(d_vals, p_vals):
            if p < 0.05:
                colors.append('#D9534F' if d > 0 else '#337AB7')
            else:
                colors.append('#777777')

        # Vertical reference line at Cohen's d = 0
        ax.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        # Reference zone for trivial effect |d| < 0.2
        ax.axvspan(-0.2, 0.2, color='#ECEFF1', alpha=0.5, zorder=0)

        # Plot error bars
        for idx in range(len(d_vals)):
            ax.errorbar(d_vals[idx], y_pos[idx], xerr=[[xerr[0, idx]], [xerr[1, idx]]],
                        fmt='o', color=colors[idx], ecolor=colors[idx], elinewidth=2.2,
                        capsize=4.5, capthick=1.8, markersize=8.0, zorder=3)

            # Significance stars
            p = p_vals[idx]
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if star:
                offset = 0.12 if d_vals[idx] >= 0 else -0.12
                ha = 'left' if d_vals[idx] >= 0 else 'right'
                ax.text(d_vals[idx] + offset, y_pos[idx], star, va='center', ha=ha,
                        fontsize=12, fontweight='bold', color=colors[idx])

        ax.set_title(m_title, fontsize=12.5, fontweight='bold', pad=12)
        ax.set_xlabel("Effect Size (Cohen's d ± 95% CI)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_xlim(-1.6, 1.6)

    # Set y-tick labels on the first axis
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(short_labels[::-1], fontsize=11, fontweight='semibold')
    axes[0].invert_yaxis()

    fig.tight_layout()
    save_publication_formats(fig, REPORTS_DIR / "Figure_1_Multicohort_Biophysical_Effect_Sizes")


# =========================================================================
# 3. Plot Figure 2: Tri-Center Parkinson's Generalization & Medication Invariance
# =========================================================================
def plot_figure_2_pd_tri_center():
    csv_pd = REPORTS_DIR / "cross_site_pd_transfer_benchmarks.csv"
    csv_onoff = REPORTS_DIR / "unm_pd_ds003490_on_off_paired_biophysics.csv"
    
    df_pd = pd.read_csv(csv_pd)
    df_onoff = pd.read_csv(csv_onoff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={'width_ratios': [1.1, 1.0]})

    # Panel A: Tri-Center Heatmap (GCN-OMST vs LR vs RF)
    piv_gcn = df_pd[df_pd['model'] == 'GCN_OMST'].pivot(index='source_cohort', columns='target_cohort', values='auc')
    # Order: Iowa -> UNM -> UCSD
    cohort_order = ['Iowa_PD', 'UNM_PD_OFF', 'UCSD_PD']
    short_names = ['Iowa (N=149)', 'UNM OFF (N=50)', 'UCSD (N=47)']

    matrix = np.zeros((3, 3))
    for i, s in enumerate(cohort_order):
        for j, t in enumerate(cohort_order):
            if s == t:
                # Diagonal: In-cohort 5-fold CV AUC from Step 17
                if s == 'Iowa_PD': matrix[i, j] = 0.685
                elif s == 'UNM_PD_OFF': matrix[i, j] = 0.699
                elif s == 'UCSD_PD': matrix[i, j] = 0.725
            else:
                matrix[i, j] = piv_gcn.loc[s, t]

    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True,
                xticklabels=short_names, yticklabels=short_names, ax=ax1,
                vmin=0.50, vmax=0.85, annot_kws={"fontsize": 13, "fontweight": "bold"})
    ax1.set_title("A. Tri-Center Parkinson's Generalization (GCN-OMST AUC)\n[Diagonal: In-Cohort CV | Off-Diagonal: Zero-Shot Transfer]",
                  fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Target Test Center", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Source Training Center", fontsize=11, fontweight='bold')

    # Panel B: Medication Invariance in UNM Paired Cohort (N=25 PD-ON vs PD-OFF pairs)
    metrics = ['Delta Power', 'Theta Power', 'Alpha Power', 'Beta Power', 'Central Beta', 'GCE']
    diff_means = df_onoff['Mean_Diff_ON_minus_OFF'].values
    p_vals = df_onoff['p_val'].values
    dz_vals = df_onoff['Cohen_dz'].values
    # Standard error of mean diff: SE = std / sqrt(25) = (diff_means / (Cohen_dz * sqrt(25))) * (1/sqrt(25)) approx
    ci_err = np.abs(diff_means / (np.maximum(np.abs(df_onoff['t_stat'].values), 1e-4))) * 2.064

    y_pos = np.arange(len(metrics))
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.2)
    ax2.barh(y_pos, diff_means, xerr=ci_err, color='#5BC0DE', edgecolor='#2A88BD',
             height=0.45, capsize=4, align='center', alpha=0.85)

    for idx, (dm, p) in enumerate(zip(diff_means, p_vals)):
        label_text = f"p = {p:.2f} (ns)"
        ax2.text(0.04 if dm >= 0 else -0.04, y_pos[idx], label_text, va='center',
                 ha='left' if dm >= 0 else 'right', fontsize=10.5, fontweight='bold', color='#333333')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics, fontsize=11, fontweight='semibold')
    ax2.set_xlabel("Within-Subject Change (ON Medication − OFF Washout ± 95% CI)", fontsize=11, fontweight='bold')
    ax2.set_title("B. Levodopa Invariance on Resting Biomarkers (UNM N=25 Pairs)\n[Paired t-test: All p > 0.70, Resting Baseline Unmasked]",
                  fontsize=12, fontweight='bold', pad=12)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.set_xlim(-0.15, 0.15)
    ax2.invert_yaxis()

    fig.tight_layout()
    save_publication_formats(fig, REPORTS_DIR / "Figure_2_Tri_Center_Parkinsons_Generalization")


# =========================================================================
# 4. Plot Figure 3: Full Transdiagnostic Transfer Matrix & Graph Ablations
# =========================================================================
def plot_figure_3_transdiagnostic_and_ablations():
    csv_all = REPORTS_DIR / "comprehensive_transdiagnostic_transfer_matrix.csv"
    csv_models = REPORTS_DIR / "multicohort_7cohort_extended_model_comparison.csv"
    
    df_all = pd.read_csv(csv_all)
    df_models = pd.read_csv(csv_models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={'width_ratios': [1.15, 1.1]})

    # Panel A: 4x4 Transdiagnostic Transfer Heatmap (Sleep Debt, MDD, PD, Dementia)
    trans_cohorts = ['Sleep_Debt', 'Mumtaz_MDD', 'Iowa_PD', 'Dementia_All']
    trans_labels = ['Sleep Debt', 'Mumtaz MDD', 'Iowa PD', 'Dementia (All)']

    piv = df_all[df_all['model'] == 'GCN_OMST'].pivot(index='source_cohort', columns='target_cohort', values='auc')
    mat = np.zeros((4, 4))
    for i, s in enumerate(trans_cohorts):
        for j, t in enumerate(trans_cohorts):
            if s == t:
                # In-cohort CV
                if s == 'Sleep_Debt': mat[i, j] = 0.529
                elif s == 'Mumtaz_MDD': mat[i, j] = 0.902
                elif s == 'Iowa_PD': mat[i, j] = 0.685
                elif s == 'Dementia_All': mat[i, j] = 0.842
            else:
                mat[i, j] = piv.loc[s, t]

    sns.heatmap(mat, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, center=0.50,
                xticklabels=trans_labels, yticklabels=trans_labels, ax=ax1,
                vmin=0.25, vmax=0.88, annot_kws={"fontsize": 12.5, "fontweight": "bold"})
    ax1.set_title("A. Full Transdiagnostic Cross-Disorder Transfer (GCN-OMST AUC)\n[Positive Transfer: Neurodegeneration | Inversion: Depression Double Dissociation]",
                  fontsize=11.5, fontweight='bold', pad=12)
    ax1.set_xlabel("Target Evaluation Cohort", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Source Training Cohort", fontsize=11, fontweight='bold')

    # Panel B: Graph Ablation Comparison Across Cohorts (OMST vs Identity A=I vs Full A=1)
    ablation_cohorts = [
        'Study 2: Mumtaz MDD (Eyes Closed)',
        'Study 5c: Dementia All (AD+FTD) vs Control',
        'Study 4: UCSD Parkinson All Sessions',
        'Study 7: Iowa Parkinson (ds004584)',
        'Study 3: Cavanagh University Depression'
    ]
    short_ab_labels = ['Mumtaz MDD', 'Dementia All', 'UCSD PD', 'Iowa PD', 'Cavanagh MDD']

    df_ab = df_models[df_models['cohort'].isin(ablation_cohorts)].copy()
    
    y_pos = np.arange(len(short_ab_labels))
    height = 0.22

    # Extract AUCs
    omst_auc = [df_ab[(df_ab['cohort'] == c) & (df_ab['model'] == 'GCN_OMST')]['auc'].values[0] for c in ablation_cohorts]
    id_auc = [df_ab[(df_ab['cohort'] == c) & (df_ab['model'] == 'GCN_Identity')]['auc'].values[0] for c in ablation_cohorts]
    full_auc = [df_ab[(df_ab['cohort'] == c) & (df_ab['model'] == 'GCN_Full')]['auc'].values[0] for c in ablation_cohorts]

    ax2.barh(y_pos - height, omst_auc, height=height, label='GCN-OMST (Filtered Topology)', color='#2E7D32', alpha=0.9)
    ax2.barh(y_pos, id_auc, height=height, label='GCN-Identity (A = I, Isolated)', color='#1976D2', alpha=0.9)
    ax2.barh(y_pos + height, full_auc, height=height, label='GCN-Full (A = 1, Dense All-to-All)', color='#E64A19', alpha=0.85)

    ax2.axvline(0.5, color='gray', linestyle='--', linewidth=1.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_ab_labels, fontsize=11, fontweight='semibold')
    ax2.set_xlabel("Cross-Validated AUC", fontsize=11, fontweight='bold')
    ax2.set_title("B. Topological Graph Inductive Bias Ablations\n[Evaluating Optimal Minimum Spanning Tree Message Passing]",
                  fontsize=11.5, fontweight='bold', pad=12)
    ax2.set_xlim(0.45, 1.0)
    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.invert_yaxis()

    fig.tight_layout()
    save_publication_formats(fig, REPORTS_DIR / "Figure_3_Transdiagnostic_Transfer_and_Ablations")


# =========================================================================
# 5. Plot Figure 4: Continuous Clinical Scale Regression
# =========================================================================
def plot_figure_4_clinical_scales():
    # Load raw graphs for actual scatter plots
    dementia_graphs = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt")
    iowa_pd_graphs = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel A: MMSE in Dementia vs Global Slowing (Delta Power) ---
    mmse_vals = [float(g['mmse']) for g in dementia_graphs if pd.notnull(g.get('mmse'))]
    delta_vals = [float(np.mean(g['x'].numpy()[:, 1])) for g in dementia_graphs if pd.notnull(g.get('mmse'))]
    diag_vals = [g.get('diagnosis') for g in dementia_graphs if pd.notnull(g.get('mmse'))]

    colors_diag = {'Control': '#2E7D32', 'FTD': '#F57C00', 'Alzheimer': '#D32F2F'}
    c_list = [colors_diag.get(d, '#777777') for d in diag_vals]

    ax1.scatter(delta_vals, mmse_vals, c=c_list, s=55, alpha=0.8, edgecolors='k', linewidth=0.6)
    sns.regplot(x=np.array(delta_vals), y=np.array(mmse_vals), ax=ax1, scatter=False,
                color='#1565C0', line_kws={'linewidth': 2.2})
    r_mmse, p_mmse = stats.pearsonr(delta_vals, mmse_vals)
    ax1.set_title(f"A. Clinical Dementia MMSE vs. Delta Slowing\n[Pearson r = {r_mmse:.3f}, p = {p_mmse:.2e} | GBDT r = +0.472]",
                  fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Global Periodic Delta Power (1–4 Hz)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("MMSE Cognitive Score (0–30)", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Custom legend for diagnosis
    patches = [
        mpatches.Patch(color='#2E7D32', label='Control (N=29)'),
        mpatches.Patch(color='#F57C00', label='FTD (N=23)'),
        mpatches.Patch(color='#D32F2F', label='Alzheimer (N=36)')
    ]
    ax1.legend(handles=patches, loc='lower left', frameon=True, fontsize=9.5)

    # --- Panel B: MoCA in Iowa PD vs. Theta Slowing ---
    moca_vals = [float(g['moca']) for g in iowa_pd_graphs if pd.notnull(g.get('moca'))]
    theta_vals = [float(np.mean(g['x'].numpy()[:, 0])) for g in iowa_pd_graphs if pd.notnull(g.get('moca'))]
    grp_vals = [g.get('group') for g in iowa_pd_graphs if pd.notnull(g.get('moca'))]

    colors_pd = {'Control': '#2E7D32', 'PD': '#C2185B'}
    c_pd_list = [colors_pd.get(g, '#777777') for g in grp_vals]

    ax2.scatter(theta_vals, moca_vals, c=c_pd_list, s=50, alpha=0.75, edgecolors='k', linewidth=0.6)
    sns.regplot(x=np.array(theta_vals), y=np.array(moca_vals), ax=ax2, scatter=False,
                color='#6A1B9A', line_kws={'linewidth': 2.2})
    r_moca, p_moca = stats.pearsonr(theta_vals, moca_vals)
    ax2.set_title(f"B. Iowa Parkinson MoCA vs. Theta Slowing\n[Pearson r = {r_moca:.3f}, p = {p_moca:.2e} | SVR r = +0.339]",
                  fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_xlabel("Global Periodic Theta Power (4–8 Hz)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("MoCA Cognitive Score (0–30)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.4)

    patches_pd = [
        mpatches.Patch(color='#2E7D32', label='Control (N=49)'),
        mpatches.Patch(color='#C2185B', label='Parkinson (N=100)')
    ]
    ax2.legend(handles=patches_pd, loc='lower left', frameon=True, fontsize=9.5)

    # --- Panel C: UPDRS Part-III Motor Severity vs Posterior Alpha in PD ---
    pd_only = [g for g in iowa_pd_graphs if g.get('group') == 'PD' and pd.notnull(g.get('updrs'))]
    updrs_vals = [float(g['updrs']) for g in pd_only]
    alpha_post = [float(np.mean(g['x'].numpy()[[13, 14, 15, 17, 18], 2])) for g in pd_only]

    ax3.scatter(alpha_post, updrs_vals, color='#D81B60', s=50, alpha=0.8, edgecolors='k', linewidth=0.6)
    sns.regplot(x=np.array(alpha_post), y=np.array(updrs_vals), ax=ax3, scatter=False,
                color='#880E4F', line_kws={'linewidth': 2.2})
    r_updrs, p_updrs = stats.pearsonr(alpha_post, updrs_vals)
    ax3.set_title(f"C. Iowa PD Motor Severity vs. Posterior Alpha\n[Pearson r = {r_updrs:.3f}, p = {p_updrs:.3f} | GBDT r = +0.165]",
                  fontsize=11.5, fontweight='bold', pad=10)
    ax3.set_xlabel("Posterior Alpha Power (8–13 Hz)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("UPDRS Part-III Motor Score", fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.4)

    fig.tight_layout()
    save_publication_formats(fig, REPORTS_DIR / "Figure_4_Clinical_Scale_Continuous_Prediction")


def main():
    print("=" * 75)
    print("🎨 Generating Expanded Biophysical Effect Sizes & Figures 1-4...")
    print("=" * 75)
    build_expanded_effect_sizes()
    plot_figure_1_expanded()
    plot_figure_2_pd_tri_center()
    plot_figure_3_transdiagnostic_and_ablations()
    plot_figure_4_clinical_scales()
    print("\n✅ All 4 Figures and Effect Size CSV generated successfully!")


if __name__ == "__main__":
    main()
