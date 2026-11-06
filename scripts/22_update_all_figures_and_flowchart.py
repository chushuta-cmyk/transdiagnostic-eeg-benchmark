#!/usr/bin/env python3
"""
Step 22: High-Precision Synchronization of Figures & Tables across 7 Cohorts.
1. Re-generates Figure 1 (Biophysical Effect Sizes) with canonical 10-20 channel indexing.
2. Re-generates Figure 4 (Clinical Scales) with:
   - Panel A: Clinical Dementia MMSE vs Posterior Alpha Collapse (r = +0.374, p = 3.36e-4)
   - Panel B: Clinical Dementia MMSE vs Global Theta Slowing (r = -0.373, p = 3.43e-4)
   - Panel C: Iowa PD MoCA vs Global Theta Slowing (r = -0.343, p = 1.88e-5)
   - Panel D: Within-Patient Sensitivity Analysis (Pure PD N=100 MoCA prediction: SVR r = +0.334, p = 6.74e-4)
3. Re-generates Figure S5 (Enrollment Flowchart) covering all 7 cohorts:
   - Enrolled: 588 participants (699 sessions)
   - Excluded: 22 participants (noise, borderline BDI, incomplete pairs)
   - Analyzed: 566 unique participants (677 recording sessions)
4. Exports updated reports/multicohort_biophysical_effect_sizes_expanded.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import seaborn as sns

BENCHMARK_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/transdiagnostic_eeg_benchmark")
DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard 19 channel list in harmonizer_and_graph.py
CHANNELS_19 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
    'Fz', 'Cz', 'Pz'
]
frontal_idx = [CHANNELS_19.index(c) for c in ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']]     # [0, 1, 2, 3, 16]
posterior_idx = [CHANNELS_19.index(c) for c in ['P3', 'P4', 'Pz', 'O1', 'O2']]   # [6, 7, 18, 8, 9]
central_idx = [CHANNELS_19.index(c) for c in ['C3', 'Cz', 'C4']]                 # [4, 17, 5]

sns.set_theme(style="ticks", font_scale=1.05)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def save_publication_formats(fig, base_path: Path):
    fig.savefig(base_path.with_suffix('.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base_path.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {base_path.name} (.pdf & .png)")


# =========================================================================
# 1. Build Canonical Biophysical Effect Sizes & Re-generate Figure 1
# =========================================================================
def build_expanded_effect_sizes_and_figure_1():
    print("Building canonical biophysical effect sizes across all 7 cohorts...")
    
    sleep_graphs = torch.load(DATA_DIR / "cohort_sleep_ds004902_graphs.pt")
    mumtaz_graphs = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt")
    cavanagh_graphs = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt")
    ucsd_pd_graphs = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt")
    dementia_graphs = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt")
    unm_pd_graphs = torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt")
    iowa_pd_graphs = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt")

    def extract_features(graph_list):
        feats = []
        for g in graph_list:
            x = g['x'].numpy()  # [19, 4]: col 0: Theta, col 1: Delta, col 2: Alpha, col 3: Chi
            frontal_theta = np.mean(x[frontal_idx, 0])
            global_theta = np.mean(x[:, 0])
            global_delta = np.mean(x[:, 1])
            posterior_alpha = np.mean(x[posterior_idx, 2])
            global_alpha = np.mean(x[:, 2])
            global_chi = np.mean(x[:, 3])
            gce = float(g.get('gce', 0.0))
            m_trees = float(g.get('omst_trees', 1.0))
            feats.append({
                'frontal_theta': frontal_theta,
                'global_theta': global_theta,
                'global_delta': global_delta,
                'posterior_alpha': posterior_alpha,
                'global_alpha': global_alpha,
                'global_chi': global_chi,
                'omst_gce': gce,
                'omst_trees': m_trees,
                'subject_id': g.get('subject_id', 'sub')
            })
        return pd.DataFrame(feats)

    # Paired sleep graphs
    sd_sleep = sorted([g for g in sleep_graphs if g.get('condition') == 'SD'], key=lambda g: g['subject_id'])
    ns_sleep = sorted([g for g in sleep_graphs if g.get('condition') == 'NS'], key=lambda g: g['subject_id'])

    comparisons = [
        ("Study 1: Acute Sleep Deprivation", "ds004902", sd_sleep, ns_sleep, "SD", "NS", True),
        ("Study 2: Mumtaz Outpatient MDD", "4244171",
         [g for g in mumtaz_graphs if g.get('group') == 'MDD'],
         [g for g in mumtaz_graphs if g.get('group') == 'HC'],
         "MDD", "Control", False),
        ("Study 3: Cavanagh University Depression", "ds003478",
         [g for g in cavanagh_graphs if g.get('group') == 'Depressed'],
         [g for g in cavanagh_graphs if g.get('group') == 'Control'],
         "MDD", "Control", False),
        ("Study 4: UCSD Parkinson All", "ds002778",
         [g for g in ucsd_pd_graphs if 'PD' in g.get('group', '')],
         [g for g in ucsd_pd_graphs if g.get('group') == 'HC'],
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
         [g for g in unm_pd_graphs if g.get('group') == 'PD' and g.get('med_state') == 'OFF'],
         [g for g in unm_pd_graphs if g.get('group') == 'CTL'],
         "PD_OFF", "Control", False),
        ("Study 6b: UNM Parkinson ON", "ds003490",
         [g for g in unm_pd_graphs if g.get('group') == 'PD' and g.get('med_state') == 'ON'],
         [g for g in unm_pd_graphs if g.get('group') == 'CTL'],
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

            if is_paired:
                diff = v_case - v_ctrl
                d = np.mean(diff) / np.std(diff, ddof=1)
                se_d = np.sqrt(1 / len(diff) + (d**2) / (2 * len(diff)))
                ci_lo = d - 1.96 * se_d
                ci_hi = d + 1.96 * se_d
                t_stat, p_val = stats.ttest_rel(v_case, v_ctrl)
                m_c, m_ctl = np.mean(v_case), np.mean(v_ctrl)
            else:
                m_c, s_c = np.mean(v_case), np.std(v_case, ddof=1)
                m_ctl, s_ctl = np.mean(v_ctrl), np.std(v_ctrl, ddof=1)
                s_pool = np.sqrt(((n_case - 1) * s_c**2 + (n_ctrl - 1) * s_ctl**2) / (n_case + n_ctrl - 2))
                d = (m_c - m_ctl) / s_pool if s_pool > 0 else 0.0
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
    print(f"✓ Saved updated effect sizes ({len(df_out)} rows) to: {out_csv}")

    # Plot Figure 1: 4-Panel Forest Plot
    focal_markers = [
        ('frontal_theta', 'A. Frontal Periodic Theta (4–8 Hz)\n[Polarity Dissociation Axis]'),
        ('global_delta', 'B. Global Periodic Delta (1–4 Hz)\n[Cortical Slow-Wave Intrusion]'),
        ('posterior_alpha', 'C. Posterior Periodic Alpha (8–13 Hz)\n[Aperiodic-Adjusted Oscillation]'),
        ('global_chi', 'D. Global Aperiodic Exponent (χ)\n[Cortical Excitation/Inhibition]')
    ]

    cohort_order = [
        "Study 1: Acute Sleep Deprivation",
        "Study 2: Mumtaz Outpatient MDD",
        "Study 3: Cavanagh University Depression",
        "Study 4: UCSD Parkinson All",
        "Study 6a: UNM Parkinson OFF",
        "Study 6b: UNM Parkinson ON",
        "Study 7: Iowa Parkinson Cohort",
        "Study 5a: Alzheimer's Disease (AD)",
        "Study 5b: Frontotemporal Dementia (FTD)",
        "Study 5c: Clinical Dementia (All)",
    ]

    cohort_clean_names = {
        "Study 1: Acute Sleep Deprivation": "1. Sleep Debt (ds004902, N=142)",
        "Study 2: Mumtaz Outpatient MDD": "2. Clinical MDD (Mumtaz, N=58)",
        "Study 3: Cavanagh University Depression": "3. Univ. Depression (ds003478, N=119)",
        "Study 4: UCSD Parkinson All": "4. UCSD Parkinson (ds002778, N=46)",
        "Study 6a: UNM Parkinson OFF": "5a. UNM PD-OFF (ds003490, N=50)",
        "Study 6b: UNM Parkinson ON": "5b. UNM PD-ON (ds003490, N=50)",
        "Study 7: Iowa Parkinson Cohort": "6. Iowa Parkinson (ds004584, N=149)",
        "Study 5a: Alzheimer's Disease (AD)": "7a. Dementia AD (ds004504, N=65)",
        "Study 5b: Frontotemporal Dementia (FTD)": "7b. Dementia FTD (ds004504, N=52)",
        "Study 5c: Clinical Dementia (All)": "7c. Dementia All (ds004504, N=88)",
    }

    fig, axes = plt.subplots(1, 4, figsize=(22, 7.8), sharey=True)

    domain_colors = {
        "1. Sleep Debt": "#3b82f6",
        "2. Clinical MDD": "#8b5cf6",
        "3. Univ. Depression": "#a855f7",
        "4. UCSD Parkinson": "#ec4899",
        "5a. UNM PD-OFF": "#f43f5e",
        "5b. UNM PD-ON": "#f43f5e",
        "6. Iowa Parkinson": "#e11d48",
        "7a. Dementia AD": "#f97316",
        "7b. Dementia FTD": "#ea580c",
        "7c. Dementia All": "#c2410c",
    }

    y_pos = np.arange(len(cohort_order))[::-1]

    for ax_idx, (m_key, title) in enumerate(focal_markers):
        ax = axes[ax_idx]
        df_sub = df_out[df_out['marker_key'] == m_key].set_index('cohort').loc[cohort_order].reset_index()

        ax.axvline(0, color='#94a3b8', linestyle='--', linewidth=1.2, zorder=1)
        ax.axvspan(-0.2, 0.2, color='#f1f5f9', alpha=0.5, zorder=0)

        for i, row in df_sub.iterrows():
            d_val = row['cohens_d']
            ci_l = row['ci_lower']
            ci_u = row['ci_upper']
            p_val = row['p_value']
            c_name = cohort_clean_names[row['cohort']]

            color = '#64748b'
            for prefix, c in domain_colors.items():
                if c_name.startswith(prefix):
                    color = c
                    break

            is_sig = (p_val < 0.05)
            marker = 's' if is_sig else 'o'
            msize = 8 if is_sig else 6

            ax.errorbar(d_val, y_pos[i], xerr=[[d_val - ci_l], [ci_u - d_val]],
                        fmt=marker, color=color, ecolor=color, elinewidth=1.8,
                        capsize=4, capthick=1.5, markersize=msize, zorder=3)

            p_str = f"p={p_val:.3f}" if p_val >= 0.001 else f"p={p_val:.1e}"
            star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            ann_text = f"{d_val:+.2f} {star}"
            offset = 0.08 if d_val >= 0 else -0.08
            ha = 'left' if d_val >= 0 else 'right'
            ax.text(d_val + offset, y_pos[i], ann_text, va='center', ha=ha,
                    fontsize=8.5, fontweight='bold' if is_sig else 'normal',
                    color='#0f172a' if is_sig else '#64748b')

        ax.set_title(title, fontsize=11.0, fontweight='bold', pad=12)
        ax.set_xlabel("Cohen's d (Case vs. Control)", fontsize=10.5, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5, axis='x')
        ax.set_xlim(-1.8, 1.8)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([cohort_clean_names[c] for c in cohort_order], fontsize=10.0, fontweight='bold')

    fig.tight_layout()
    save_publication_formats(fig, REPORTS_DIR / "Figure_1_Multicohort_Biophysical_Effect_Sizes")


# =========================================================================
# 2. Re-generate Figure 4: Clinical Scale Continuous Prediction
# =========================================================================
def plot_figure_4_clinical_scales():
    print("Generating Figure 4 (Clinical Scale Continuous Prediction)...")
    dementia_graphs = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt")
    iowa_pd_graphs = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt")

    fig, axes = plt.subplots(1, 4, figsize=(23, 5.5))
    ax1, ax2, ax3, ax4 = axes

    # --- Panel A: Dementia MMSE vs Posterior Periodic Alpha Collapse ---
    mmse_vals = [float(g['mmse']) for g in dementia_graphs if pd.notnull(g.get('mmse'))]
    alpha_post_dem = [float(np.mean(g['x'].numpy()[posterior_idx, 2])) for g in dementia_graphs if pd.notnull(g.get('mmse'))]
    diag_vals = [g.get('diagnosis') for g in dementia_graphs if pd.notnull(g.get('mmse'))]

    colors_diag = {'Control': '#2E7D32', 'FTD': '#F57C00', 'Alzheimer': '#D32F2F'}
    c_list = [colors_diag.get(d, '#777777') for d in diag_vals]

    ax1.scatter(alpha_post_dem, mmse_vals, c=c_list, s=55, alpha=0.85, edgecolors='k', linewidth=0.6)
    sns.regplot(x=np.array(alpha_post_dem), y=np.array(mmse_vals), ax=ax1, scatter=False,
                color='#1565C0', line_kws={'linewidth': 2.2})
    r_alpha, p_alpha = stats.pearsonr(alpha_post_dem, mmse_vals)
    ax1.set_title(f"A. Dementia MMSE vs. Posterior Alpha Collapse\n[Pearson r = {r_alpha:+.3f}, p = {p_alpha:.2e} | GBDT r = +0.472]",
                  fontsize=11.0, fontweight='bold', pad=10)
    ax1.set_xlabel("Posterior Periodic Alpha (8–13 Hz)", fontsize=10.5, fontweight='bold')
    ax1.set_ylabel("MMSE Cognitive Score (0–30)", fontsize=10.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.4)

    patches_dem = [
        mpatches.Patch(color='#2E7D32', label='Control (N=29)'),
        mpatches.Patch(color='#F57C00', label='FTD (N=23)'),
        mpatches.Patch(color='#D32F2F', label='Alzheimer (N=36)')
    ]
    ax1.legend(handles=patches_dem, loc='lower right', frameon=True, fontsize=9.0)

    # --- Panel B: Dementia MMSE vs Global Theta Slowing ---
    theta_dem = [float(np.mean(g['x'].numpy()[:, 0])) for g in dementia_graphs if pd.notnull(g.get('mmse'))]
    ax2.scatter(theta_dem, mmse_vals, c=c_list, s=55, alpha=0.85, edgecolors='k', linewidth=0.6)
    sns.regplot(x=np.array(theta_dem), y=np.array(mmse_vals), ax=ax2, scatter=False,
                color='#C2185B', line_kws={'linewidth': 2.2})
    r_theta_dem, p_theta_dem = stats.pearsonr(theta_dem, mmse_vals)
    ax2.set_title(f"B. Dementia MMSE vs. Theta Slowing\n[Pearson r = {r_theta_dem:+.3f}, p = {p_theta_dem:.2e}]",
                  fontsize=11.0, fontweight='bold', pad=10)
    ax2.set_xlabel("Global Periodic Theta Power (4–8 Hz)", fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("MMSE Cognitive Score (0–30)", fontsize=10.5, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.4)

    # --- Panel C: Iowa PD MoCA vs Global Theta Slowing ---
    moca_vals = [float(g['moca']) for g in iowa_pd_graphs if pd.notnull(g.get('moca'))]
    theta_pd = [float(np.mean(g['x'].numpy()[:, 0])) for g in iowa_pd_graphs if pd.notnull(g.get('moca'))]
    grp_vals = [g.get('group') for g in iowa_pd_graphs if pd.notnull(g.get('moca'))]

    colors_pd = {'Control': '#2E7D32', 'PD': '#8E24AA'}
    c_pd_list = [colors_pd.get(g, '#777777') for g in grp_vals]

    ax3.scatter(theta_pd, moca_vals, c=c_pd_list, s=50, alpha=0.75, edgecolors='k', linewidth=0.6)
    sns.regplot(x=np.array(theta_pd), y=np.array(moca_vals), ax=ax3, scatter=False,
                color='#4A148C', line_kws={'linewidth': 2.2})
    r_moca, p_moca = stats.pearsonr(theta_pd, moca_vals)
    ax3.set_title(f"C. Iowa Parkinson MoCA vs. Theta Slowing\n[Pearson r = {r_moca:+.3f}, p = {p_moca:.2e} | All N=149]",
                  fontsize=11.0, fontweight='bold', pad=10)
    ax3.set_xlabel("Global Periodic Theta Power (4–8 Hz)", fontsize=10.5, fontweight='bold')
    ax3.set_ylabel("MoCA Cognitive Score (0–30)", fontsize=10.5, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.4)

    patches_pd = [
        mpatches.Patch(color='#2E7D32', label='Control (N=49)'),
        mpatches.Patch(color='#8E24AA', label='Parkinson (N=100)')
    ]
    ax3.legend(handles=patches_pd, loc='lower left', frameon=True, fontsize=9.0)

    # --- Panel D: Sensitivity Analysis - Within PD Patients Only (MoCA) ---
    pd_only_graphs = [g for g in iowa_pd_graphs if g.get('group') == 'PD' and pd.notnull(g.get('moca'))]
    moca_pd_only = np.array([float(g['moca']) for g in pd_only_graphs])
    theta_pd_only = np.array([float(np.mean(g['x'].numpy()[:, 0])) for g in pd_only_graphs])

    # Out-of-fold SVR predictions within pure PD patients
    from sklearn.model_selection import KFold
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_pd = np.array([g['x'].numpy().reshape(-1) for g in pd_only_graphs])
    y_pred_pd = np.zeros(len(moca_pd_only))

    for tr_idx, te_idx in kf.split(X_pd):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_pd[tr_idx])
        X_te = scaler.transform(X_pd[te_idx])
        svr = SVR(C=1.0, epsilon=0.1)
        svr.fit(X_tr, moca_pd_only[tr_idx])
        y_pred_pd[te_idx] = svr.predict(X_te)

    r_pure_pd, p_pure_pd = stats.pearsonr(moca_pd_only, y_pred_pd)

    ax4.scatter(y_pred_pd, moca_pd_only, color='#E91E63', s=55, alpha=0.8, edgecolors='k', linewidth=0.6)
    sns.regplot(x=y_pred_pd, y=moca_pd_only, ax=ax4, scatter=False,
                color='#880E4F', line_kws={'linewidth': 2.2})
    ax4.set_title(f"D. Within-Patient SVR Prediction (Pure PD N=100)\n[Cross-Val r = {r_pure_pd:+.3f}, p = {p_pure_pd:.2e} | No HC Leak]",
                  fontsize=11.0, fontweight='bold', pad=10)
    ax4.set_xlabel("SVR-Predicted MoCA Score", fontsize=10.5, fontweight='bold')
    ax4.set_ylabel("Actual MoCA Score (9–30)", fontsize=10.5, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.4)

    fig.tight_layout()
    save_publication_formats(fig, REPORTS_DIR / "Figure_4_Clinical_Scale_Continuous_Prediction")


# =========================================================================
# 3. Re-generate Figure S5: 7-Cohort Master Participant Flowchart
# =========================================================================
def plot_figure_s5_seven_cohorts():
    print("Generating Figure S5 (7-Cohort Master Participant Flowchart)...")
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.axis('off')

    def draw_box(x, y, w, h, text, bg='#f8fafc', border='#1e293b', fontsize=8.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                      facecolor=bg, edgecolor=border, lw=1.3, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
                color='#0f172a', zorder=3)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#475569", lw=1.3), zorder=1)

    # Top: Total Master Enrollment
    draw_box(0.25, 0.88, 0.50, 0.09,
             "Total Multi-Cohort Enrolled Sample Across Seven Independent Centers\n"
             "N = 574 Unique Participants (677 Primary / 737 Total Sessions)\n"
             "6 Public OpenNeuro Datasets + 1 Clinical Repository (ds004902, 4244171, ds003478, ds002778, ds004504, ds003490, ds004584)",
             bg='#e2e8f0', border='#0f172a', fontsize=9.2)

    # Level 2: Seven Cohorts Enrollment
    # 7 columns
    n_c = 7
    w = 0.125
    gap = (1.0 - n_c * w) / (n_c + 1)
    xs = [gap + i * (w + gap) for i in range(n_c)]
    h = 0.12

    cohort_boxes = [
        ("Study 1: Sleep Debt\n(ds004902)\nEnrolled: 71 pairs\n(142 sessions)", xs[0]),
        ("Study 2: Clinical MDD\n(Figshare 4244171)\nEnrolled: N = 64\n(34 MDD, 30 HC)", xs[1]),
        ("Study 3: Univ. Dep.\n(ds003478)\nEnrolled: N = 121\n(Student Screening)", xs[2]),
        ("Study 4: UCSD PD\n(ds002778)\nEnrolled: N = 31\n(15 PD, 16 HC)", xs[3]),
        ("Study 5: Dementia\n(ds004504)\nEnrolled: N = 88\n(36 AD, 23 FTD, 29 C)", xs[4]),
        ("Study 6: UNM PD\n(ds003490)\nEnrolled: N = 50\n(25 PD, 25 CTL)", xs[5]),
        ("Study 7: Iowa PD\n(ds004584)\nEnrolled: N = 149\n(100 PD, 49 CTL)", xs[6]),
    ]

    for i, (title, x) in enumerate(cohort_boxes):
        draw_arrow(0.50, 0.88, x + w / 2, 0.72 + h)
        draw_box(x, 0.72, w, h, title, bg='#eff6ff', border='#2563eb', fontsize=7.8)

    # Level 3: Exclusions / QC
    exclusions = [
        ("QC Criteria Met:\n0 Excluded\n(71 complete pairs)", xs[0], '#f0fdf4', '#16a34a'),
        ("Excluded: N = 6\n(Severe channel noise\nor motion >20%)", xs[1], '#fef2f2', '#dc2626'),
        ("Excluded: N = 2\n(Borderline BDI\n10 ≤ BDI ≤ 13)", xs[2], '#fef2f2', '#dc2626'),
        ("QC Criteria Met:\n0 Excluded\n(31 valid subjects\n46 sessions)", xs[3], '#f0fdf4', '#16a34a'),
        ("QC Criteria Met:\n0 Excluded\n(88 valid .set runs)", xs[4], '#f0fdf4', '#16a34a'),
        ("QC Criteria Met:\n0 Excluded\n(50 valid pairs/ctl)", xs[5], '#f0fdf4', '#16a34a'),
        ("QC Criteria Met:\n0 Excluded\n(149 valid BIDS)", xs[6], '#f0fdf4', '#16a34a'),
    ]

    for text, x, bg_c, bdr_c in exclusions:
        draw_arrow(x + w / 2, 0.72, x + w / 2, 0.46 + h)
        draw_box(x, 0.46, w, h, text, bg=bg_c, border=bdr_c, fontsize=7.5)

    # Level 4: Final Analyzed Cohorts
    analyzed = [
        ("Analyzed: N = 71\n(142 sessions)\n71 TSD (24–30h wake)\n71 Normal Baseline", xs[0]),
        ("Analyzed: N = 58\n(58 EC sessions)\n30 Outpatient MDD\n28 Healthy Controls", xs[1]),
        ("Analyzed: N = 119\n(119 sessions)\n44 Depressed (BDI≥14)\n75 Controls (BDI≤9)", xs[2]),
        ("Analyzed: N = 31\n(46 sessions)\n16 Healthy Controls\n15 PD (ON + OFF sess)", xs[3]),
        ("Analyzed: N = 88\n(88 sessions)\n36 AD, 23 FTD\n29 Healthy Controls", xs[4]),
        ("Analyzed: N = 50\n(75 sessions)\n25 PD (ON + OFF sess)\n25 Healthy Controls", xs[5]),
        ("Analyzed: N = 149\n(149 sessions)\n100 Parkinson\n49 Healthy Controls", xs[6]),
    ]

    for text, x in analyzed:
        draw_arrow(x + w / 2, 0.46, x + w / 2, 0.20 + h)
        draw_box(x, 0.20, w, h, text, bg='#f0fdf4', border='#15803d', fontsize=7.6)

    # Level 5: Bottom Summary Box
    for x in xs:
        draw_arrow(x + w / 2, 0.20, 0.50, 0.08)

    draw_box(0.20, -0.01, 0.60, 0.09,
             "Final Harmonized Multi-Cohort Benchmark Dataset\n"
             "Total Analyzed: N = 566 Unique Participants | 677 Recording Sessions (Primary EC)\n"
             "Universal 19-Channel 10–20 Coordinate Mapping · Specparam Gain Invariance · Strict Fold-Nested HC Calibration",
             bg='#dcfce7', border='#166534', fontsize=9.0)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.04, 1.0)

    out_pdf = REPORTS_DIR / "fig_S5_enrollment_flowchart.pdf"
    out_png = REPORTS_DIR / "fig_S5_enrollment_flowchart.png"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✓ Generated: {out_pdf.name} & {out_png.name}")


def main():
    print("=" * 75)
    print("🚀 Step 22: Synchronizing Figures 1, 4, S5 and Effect Sizes...")
    print("=" * 75)
    build_expanded_effect_sizes_and_figure_1()
    plot_figure_4_clinical_scales()
    plot_figure_s5_seven_cohorts()
    print("\n🎉 Step 22 complete! All figures and flowcharts synchronized with 7 cohorts!")


if __name__ == "__main__":
    main()
