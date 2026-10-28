#!/usr/bin/env python3
"""
Step 04: Publication-Quality Visualization for Multi-Cohort Transdiagnostic EEG Benchmark.
Generates 300-DPI publication figures in reports/:
  - Figure_1_Multicohort_Biophysical_Effect_Sizes.png & .tiff (Forest plot comparing d values)
  - Figure_2_OMST_Topology_and_Efficiency.png & .tiff (GCE and network density distributions)
  - Figure_3_Within_vs_Cross_Cohort_Performance.png & .tiff (Within-cohort CV vs. Zero-shot transfer AUC)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_png_and_tiff(fig, base_path: Path):
    png_path = base_path.with_suffix(".png")
    tiff_path = base_path.with_suffix(".tiff")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    with Image.open(png_path) as img:
        img.convert("RGB").save(tiff_path, format="TIFF", dpi=(300, 300), compression="tiff_lzw")
    print(f"✓ Saved {png_path.name} & 300-DPI {tiff_path.name}")


def plot_figure_1_forest():
    csv_path = REPORTS_DIR / "multicohort_biophysical_effect_sizes.csv"
    if not csv_path.exists():
        print(f"⚠️ {csv_path.name} not found. Skipping Figure 1.")
        return

    df = pd.read_csv(csv_path)

    # Key markers of interest
    focal_markers = [
        ('frontal_theta', 'Frontal Periodic Theta (4–8 Hz)'),
        ('posterior_alpha', 'Posterior Periodic Alpha (8–13 Hz)'),
        ('global_chi', 'Global Aperiodic Exponent (χ)'),
        ('omst_gce', 'OMST Global Cost Efficiency (GCE)')
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)

    cohorts = df['cohort'].unique()
    y_pos = np.arange(len(cohorts))
    colors = ['#2980b9', '#c0392b', '#8e44ad', '#d35400', '#16a085']

    for ax_idx, (m_key, m_title) in enumerate(focal_markers):
        ax = axes[ax_idx]
        sub = df[df['marker_key'] == m_key].set_index('cohort').reindex(cohorts).reset_index()

        d_vals = sub['cohens_d'].values
        ci_lo = sub['ci_lower'].values
        ci_hi = sub['ci_upper'].values

        xerr = [d_vals - ci_lo, ci_hi - d_vals]

        ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

        for i, (d, c) in enumerate(zip(d_vals, colors[:len(cohorts)])):
            ax.errorbar(d, y_pos[i], xerr=[[xerr[0][i]], [xerr[1][i]]], fmt='o',
                        color=c, ecolor=c, elinewidth=2.2, capsize=5, capthick=2, markersize=8)

        ax.set_title(m_title, fontweight='bold', fontsize=11)
        ax.set_xlabel("Cohen's d [95% CI]", fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.4)
        if ax_idx == 0:
            clean_labels = [c.replace('Study ', 'S').split(' (')[0] for c in cohorts]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(clean_labels, fontweight='bold', fontsize=9.5)

    plt.suptitle("Multi-Cohort Biophysical Effect Sizes Across Sleep Debt, Depression, and Parkinson's Disease",
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_png_and_tiff(fig, REPORTS_DIR / "Figure_1_Multicohort_Biophysical_Effect_Sizes")


def plot_figure_2_benchmarks():
    csv_path = REPORTS_DIR / "multicohort_classification_benchmark.csv"
    if not csv_path.exists():
        print(f"⚠️ {csv_path.name} not found. Skipping Figure 2.")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    names = df['cohort'].values
    aucs = df['roc_auc'].values
    lo = df['auc_ci_lower'].values
    hi = df['auc_ci_upper'].values
    types = df['evaluation_type'].values

    y_pos = np.arange(len(names))
    xerr = [aucs - lo, hi - aucs]

    colors = ['#27ae60' if 'Within' in t else '#e67e22' for t in types]

    bars = ax.barh(y_pos, aucs, xerr=xerr, color=colors, alpha=0.8, edgecolor='black', height=0.6,
                   capsize=4, error_kw=dict(elinewidth=1.8, ecolor='black'))

    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2, label='Chance Baseline (AUC = 0.50)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontweight='bold', fontsize=10)
    ax.set_xlabel("ROC-AUC [95% Bootstrap CI]", fontweight='bold')
    ax.set_xlim(0.15, 1.02)
    ax.set_title("Fold-Nested NormativeGCN Classification: Within-Cohort CV vs. Cross-Cohort Transfer",
                 fontweight='bold', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4, axis='x')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='Within-Cohort CV (Valid Model)'),
        Patch(facecolor='#e67e22', edgecolor='black', label='Cross-Cohort Transfer (Zero-Shot)'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Chance Level (0.50)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5)

    plt.tight_layout()
    save_png_and_tiff(fig, REPORTS_DIR / "Figure_2_Within_vs_Cross_Cohort_Performance")


def main():
    print("🎨 Generating Multi-Cohort Publication Figures...")
    plot_figure_1_forest()
    plot_figure_2_benchmarks()


if __name__ == "__main__":
    main()
