#!/usr/bin/env python3
"""
Step 04: Publication-Quality Visualization for Multi-Cohort Transdiagnostic EEG Benchmark.
Generates publication-standard vector graphics (.pdf, .svg) and 300-DPI raster images (.png, .tiff):
  - Figure 1: Multi-Cohort Biophysical Effect Sizes Across Pathologies (4-Panel Forest Plot with FDR q < 0.05 and paired dz)
  - Figure 2: NormativeGCN vs. Logistic Regression Benchmark (Within-Cohort CV & Zero-Shot Cross-Cohort Transfer)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

# Publication styling
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_publication_formats(fig, base_path: Path):
    """Saves figure in Vector (PDF, SVG) and 300-DPI Raster (PNG, LZW-TIFF) formats."""
    pdf_path = base_path.with_suffix(".pdf")
    svg_path = base_path.with_suffix(".svg")
    png_path = base_path.with_suffix(".png")
    tiff_path = base_path.with_suffix(".tiff")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with Image.open(png_path) as img:
        img.convert("RGB").save(tiff_path, format="TIFF", dpi=(300, 300), compression="tiff_lzw")

    print(f"✓ Saved publication formats for {base_path.name}: PDF, SVG, PNG, and 300-DPI TIFF")


def plot_figure_1_forest():
    csv_path = REPORTS_DIR / "multicohort_biophysical_effect_sizes.csv"
    if not csv_path.exists():
        print(f"⚠️ {csv_path.name} not found. Skipping Figure 1.")
        return

    df = pd.read_csv(csv_path)

    # 4 focal markers
    focal_markers = [
        ('frontal_theta', 'A. Frontal Periodic Theta (4–8 Hz)\n[Polarity Dissociation Axis]'),
        ('posterior_alpha', 'B. Posterior Periodic Alpha (8–13 Hz)\n[Aperiodic-Adjusted Oscillation]'),
        ('global_chi', 'C. Global Aperiodic Exponent (χ)\n[Cortical Excitation/Inhibition]'),
        ('omst_gce', 'D. OMST Global Cost Efficiency\n[Topological Network Wiring]')
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.8), sharey=True)

    cohorts = [
        'Study 1: Sleep Deprivation (ds004902)',
        'Study 2: Mumtaz Clinical MDD (4244171)',
        'Study 3: Cavanagh Depression (ds003478)',
        'Study 4a: UCSD Parkinson OFF (ds002778)',
        'Study 4b: UCSD Parkinson ON (ds002778)'
    ]
    y_pos = np.arange(len(cohorts))
    colors = ['#1f77b4', '#d62728', '#9467bd', '#ff7f0e', '#2ca02c']

    cohort_labels = [
        'Study 1: Sleep Debt (N=71 pairs)',
        'Study 2: Inpatient MDD (N=58)',
        'Study 3: Univ Depression (N=119)',
        'Study 4a: Parkinson OFF (N=31)',
        'Study 4b: Parkinson ON (N=31)'
    ]

    for ax_idx, (m_key, m_title) in enumerate(focal_markers):
        ax = axes[ax_idx]
        sub = df[df['marker_key'] == m_key].set_index('cohort').reindex(cohorts).reset_index()

        # Zero reference line
        ax.axvline(0, color='#555555', linestyle='--', linewidth=1.2, alpha=0.8)

        for i in range(len(cohorts)):
            c = colors[i]
            # For Study 1, use paired dz and paired CI for within-subject design
            if i == 0 and sub.loc[i, 'is_paired']:
                d = sub.loc[i, 'paired_dz']
                ci_lo = sub.loc[i, 'paired_dz_ci_lower']
                ci_hi = sub.loc[i, 'paired_dz_ci_upper']
                p_val = sub.loc[i, 'paired_p_value']
            else:
                d = sub.loc[i, 'cohens_d']
                ci_lo = sub.loc[i, 'ci_lower']
                ci_hi = sub.loc[i, 'ci_upper']
                p_val = sub.loc[i, 'p_value']

            if np.isnan(d):
                continue

            xerr = [[d - ci_lo], [ci_hi - d]]
            ax.errorbar(d, y_pos[i], xerr=xerr, fmt='o',
                        color=c, ecolor=c, elinewidth=2.2, capsize=5, capthick=2, markersize=8)

            q_val = sub.loc[i, 'q_value_focal'] if 'q_value_focal' in sub.columns else np.nan

            # Significance annotation
            if pd.notnull(q_val) and q_val < 0.01:
                ax.text(d, y_pos[i] - 0.22, "**", ha='center', va='bottom', color=c, fontweight='bold', fontsize=12)
            elif pd.notnull(q_val) and q_val < 0.05:
                ax.text(d, y_pos[i] - 0.22, "*", ha='center', va='bottom', color=c, fontweight='bold', fontsize=12)
            elif pd.notnull(p_val) and p_val < 0.05:
                ax.text(d, y_pos[i] - 0.22, "†", ha='center', va='bottom', color=c, fontweight='bold', fontsize=11)

        ax.set_title(m_title, fontweight='bold', fontsize=11, pad=10)
        ax.set_xlabel("Effect Size [95% CI]\n(Cohen's d, or paired dz for Study 1)", fontweight='bold', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

        if ax_idx == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cohort_labels, fontweight='bold', fontsize=10)

        # Invert y-axis so Study 1 is at top (y=0) and Study 4b is at bottom (y=4)
        ax.invert_yaxis()

    # Global legend / annotation
    fig.text(0.5, -0.05, "* FDR q < 0.05; ** FDR q < 0.01 (Benjamini-Hochberg across 20 focal hypothesis tests); † Uncorrected p < 0.05 (exploratory). dz indicates within-subject paired effect size for Study 1.",
             ha='center', fontsize=10, fontstyle='italic', color='#333333')

    plt.suptitle("Multi-Cohort Biophysical Effect Sizes Across Acute Sleep Debt, Major Depression, and Parkinson's Disease",
                 fontweight='bold', fontsize=14, y=1.04)
    plt.tight_layout()
    save_publication_formats(fig, REPORTS_DIR / "Figure_1_Multicohort_Biophysical_Effect_Sizes")


def plot_figure_2_benchmarks():
    csv_path = REPORTS_DIR / "multicohort_baseline_comparison.csv"
    if not csv_path.exists():
        csv_path = REPORTS_DIR / "multicohort_classification_benchmark.csv"

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.8), gridspec_kw={'width_ratios': [1.15, 1.25]})

    # --- Panel A: Within-Cohort Benchmark ---
    df_within = df[df['evaluation_type'].str.contains('Within-Cohort')].copy()
    
    # Order: Study 1 -> Study 2 -> Study 3 -> Study 4 All -> Study 4a ON -> Study 4b OFF
    target_within_order = [
        'Sleep_ds004902_SingleSess',
        'Mumtaz_MDD_4244171',
        'Cavanagh_ds003478',
        'UCSD_Parkinson_AllSessions',
        'UCSD_Parkinson_ON_vs_HC',
        'UCSD_Parkinson_OFF_vs_HC'
    ]
    df_within['order_key'] = df_within['cohort'].apply(lambda x: target_within_order.index(x) if x in target_within_order else 99)
    df_within = df_within.sort_values('order_key').reset_index(drop=True)

    clean_within_names = [
        'Study 1: Sleep Debt (142 sessions)',
        'Study 2: Mumtaz Inpatient MDD (N=58)',
        'Study 3: Cavanagh Depression (N=119)',
        'Study 4: UCSD Parkinson All (46 sessions)',
        'Study 4a: UCSD Parkinson ON vs HC (N=31)',
        'Study 4b: UCSD Parkinson OFF vs HC (N=31)'
    ]

    y_pos = np.arange(len(df_within))
    height = 0.36

    gcn_aucs = df_within['gcn_auc'].values
    gcn_lo = df_within['gcn_ci_lower'].values
    gcn_hi = df_within['gcn_ci_upper'].values
    gcn_err = [gcn_aucs - gcn_lo, gcn_hi - gcn_aucs]

    lr_aucs = df_within['lr_auc'].values
    lr_lo = df_within['lr_ci_lower'].values
    lr_hi = df_within['lr_ci_upper'].values
    lr_err = [lr_aucs - lr_lo, lr_hi - lr_aucs]

    ax1.barh(y_pos - height/2, gcn_aucs, height=height, xerr=gcn_err, color='#2ca02c', alpha=0.88,
             edgecolor='black', capsize=3.5, label='NormativeGCN (Graph Topology + Z-Map)', error_kw=dict(elinewidth=1.6))
    ax1.barh(y_pos + height/2, lr_aucs, height=height, xerr=lr_err, color='#1f77b4', alpha=0.75,
             edgecolor='black', capsize=3.5, label='Logistic Regression Baseline (Flat 76-D Z-Map)', error_kw=dict(elinewidth=1.6))

    for i in range(len(df_within)):
        # GCN label placed cleanly outside error bar
        x_gcn = max(gcn_hi[i], gcn_aucs[i]) + 0.02
        ax1.text(x_gcn, y_pos[i] - height/2, f"{gcn_aucs[i]:.3f}", va='center', fontweight='bold', fontsize=9, color='#1b5e20')
        # LR label placed cleanly outside error bar
        x_lr = max(lr_hi[i], lr_aucs[i]) + 0.02
        ax1.text(x_lr, y_pos[i] + height/2, f"{lr_aucs[i]:.3f}", va='center', fontsize=8.5, color='#0d47a1')

    ax1.axvline(0.5, color='#d62728', linestyle='--', linewidth=1.3, label='Chance Baseline (0.50)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(clean_within_names, fontweight='bold', fontsize=9.5)
    ax1.set_xlabel("ROC-AUC [95% Bootstrap CI]", fontweight='bold', fontsize=10)
    ax1.set_xlim(0.35, 1.15)
    ax1.set_title("A. Within-Cohort Cross-Validation Benchmarks\n(Strict Leak-Free Fold-Nested Normative Calibration)", fontweight='bold', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5, axis='x')
    ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.92)
    ax1.invert_yaxis()

    # --- Panel B: Cross-Cohort Zero-Shot Transfer ---
    df_trans = df[df['evaluation_type'].str.contains('Zero-Shot')].copy().reset_index(drop=True)
    y_pos2 = np.arange(len(df_trans))

    clean_trans_names = [
        'Cavanagh MDD → Mumtaz MDD (Cross-Center)',
        'Mumtaz MDD → Cavanagh MDD (Severe-to-Mild)',
        'Sleep Debt → Parkinson (Shared Slowing)',
        'Sleep Debt → Mumtaz MDD (Theta Dissociation)*',
        'Cavanagh MDD → Parkinson (Bidirectional Theta)*'
    ]

    tgcn_aucs = df_trans['gcn_auc'].values
    tgcn_lo = df_trans['gcn_ci_lower'].values
    tgcn_hi = df_trans['gcn_ci_upper'].values
    tgcn_err = [tgcn_aucs - tgcn_lo, tgcn_hi - tgcn_aucs]

    tlr_aucs = df_trans['lr_auc'].values
    tlr_lo = df_trans['lr_ci_lower'].values
    tlr_hi = df_trans['lr_ci_upper'].values
    tlr_err = [tlr_aucs - tlr_lo, tlr_hi - tlr_aucs]

    ax2.barh(y_pos2 - height/2, tgcn_aucs, height=height, xerr=tgcn_err, color='#ff7f0e', alpha=0.88,
             edgecolor='black', capsize=3.5, label='NormativeGCN Zero-Shot Transfer', error_kw=dict(elinewidth=1.6))
    ax2.barh(y_pos2 + height/2, tlr_aucs, height=height, xerr=tlr_err, color='#1f77b4', alpha=0.75,
             edgecolor='black', capsize=3.5, label='Logistic Regression Baseline', error_kw=dict(elinewidth=1.6))

    for i in range(len(df_trans)):
        val = tgcn_aucs[i]
        hi_val = tgcn_hi[i]
        x_gcn = max(hi_val, val) + 0.02
        if val < 0.35:
            eff = 1.0 - val
            ax2.text(x_gcn, y_pos2[i] - height/2, f"AUC = {val:.3f} (Eff: {eff:.3f})*",
                     va='center', fontweight='bold', fontsize=8.5, color='#b71c1c')
        else:
            ax2.text(x_gcn, y_pos2[i] - height/2, f"AUC = {val:.3f}",
                     va='center', fontweight='bold', fontsize=8.5, color='#e65100')
        
        x_lr = max(tlr_hi[i], tlr_aucs[i]) + 0.02
        ax2.text(x_lr, y_pos2[i] + height/2, f"{tlr_aucs[i]:.3f}",
                 va='center', fontsize=8.5, color='#0d47a1')

    ax2.axvline(0.5, color='#d62728', linestyle='--', linewidth=1.3, label='Chance Baseline (0.50)')
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(clean_trans_names, fontweight='bold', fontsize=9.5)
    ax2.set_xlabel("ROC-AUC [95% Bootstrap CI]", fontweight='bold', fontsize=10)
    ax2.set_xlim(0.05, 1.25)
    ax2.set_title("B. Zero-Shot Cross-Cohort Model Transfer\n(Testing Generalization & Polarity Dissociations)", fontweight='bold', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.5, axis='x')
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.92)
    ax2.invert_yaxis()

    fig.text(0.5, -0.04, "* Inverted AUC indicates diametrically reversed electrophysiological polarity between conditions (effective discrimination AUC = 1.0 − AUC). All error bars represent 1,000-sample bootstrap 95% confidence intervals.",
             ha='center', fontsize=9.5, fontstyle='italic', color='#333333')

    plt.suptitle("NormativeGCN Classification Benchmarks: Within-Cohort CV vs. Zero-Shot Cross-Cohort Transfer Against Linear Baselines",
                 fontweight='bold', fontsize=13, y=1.03)
    plt.tight_layout()
    save_publication_formats(fig, REPORTS_DIR / "Figure_2_Within_vs_Cross_Cohort_Performance")


def main():
    print("🎨 Generating Publication-Standard Figures 1 & 2 (Vector PDF/SVG + 300-DPI PNG/TIFF)...")
    plot_figure_1_forest()
    plot_figure_2_benchmarks()


if __name__ == "__main__":
    main()
