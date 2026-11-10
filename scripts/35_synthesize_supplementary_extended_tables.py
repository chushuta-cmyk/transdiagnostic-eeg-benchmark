#!/usr/bin/env python3
"""
Supplementary Extended Tables Synthesis (Tables S19 - S23).

Compiles markdown statistical tables from raw benchmark CSV outputs:
- Table S19: Empirical quantification of validation data leakage
- Table S20: Cross-site PD calibration ablation & gain perturbation
- Table S21: Multi-seed repeated cross-validation benchmark
- Table S22: PLI eigenvalue spectrum audit & Riemannian tangent space comparison
- Table S23: Independent random-effects meta-analysis of frontal theta polarity
"""

from pathlib import Path
import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = BENCHMARK_ROOT / "reports"


def fmt_p_sci(p: float, bold_sig: bool = False) -> str:
    """Format float p-value into clean scientific notation or decimal string."""
    if pd.isna(p):
        return "—"
    if p < 0.001:
        mant, exp = f"{p:.2e}".split("e")
        s = f"{mant} × 10^{int(exp)}"
    else:
        s = f"{p:.4f}"
    if bold_sig and p < 0.05:
        s = f"**{s}**"
    return s


def build_table_s19_leakage() -> str:
    """Synthesize Markdown Table S19 (quantification of validation data leakage)."""
    csv_path = REPORTS_DIR / "benchmark_leakage_ablation.csv"
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path)

    lines = [
        "### Table S19. Empirical Quantification of Data Leakage Across Repeated-Session Cohorts",
        "| Cohort (Label Structure) | Evaluation Regime | Model | N (Sess / Subj) | Fold Subject Overlap | Out-of-Fold ROC-AUC | Session Bootstrap 95% CI | Participant-Cluster 95% CI | Bal. Acc. | ΔAUC vs. Leak-Free |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    design_map = {
        "Paired_Crossover_Discordant": "Paired Crossover (y in {0,1} per subj)",
        "Repeated_Sessions_Concordant": "Repeated Case Sessions (y = 1 concordant)",
    }

    prev_cohort = None
    for _, row in df.iterrows():
        c_disp = f"**{row['cohort']}** ({design_map.get(row['design_type'], row['design_type'])})" if row["cohort"] != prev_cohort else ""
        prev_cohort = row["cohort"]

        if row["regime"] == "Strict_Leak_Free":
            reg_fmt = "**Strict Leak-Free (`StratifiedGroupKFold`)**"
            delta_str = "0.000 (Ref)"
        elif row["regime"] == "Leaky_Pre_Standardization":
            reg_fmt = "Leaky Pre-Standardization (Grouped Split)"
            d_val = float(row["delta_auc_vs_strict"])
            delta_str = f"**{d_val:+.3f}**" if abs(d_val) >= 0.020 else f"{d_val:+.3f}"
        else:
            reg_fmt = "*Leaky Subject Mixing (`StratifiedKFold`)*"
            d_val = float(row["delta_auc_vs_strict"])
            delta_str = f"**{d_val:+.3f}**" if abs(d_val) >= 0.020 else f"{d_val:+.3f}"

        n_str = f"{int(row['n_sessions'])} / {int(row['n_participants'])}"
        ov_str = f"{row['subject_overlap_pct']:.1f}%"
        auc_str = f"{row['auc']:.3f}"
        sess_ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        clus_ci = f"[{row['cluster_ci_lower']:.3f}, {row['cluster_ci_upper']:.3f}]"
        bacc_str = f"{row['balanced_accuracy']:.3f}"

        lines.append(
            f"| {c_disp} | {reg_fmt} | `{row['model']}` | {n_str} | {ov_str} | {auc_str} | {sess_ci} | {clus_ci} | {bacc_str} | {delta_str} |"
        )

    lines.append(
        "\n*Note: Evaluates the empirical impact of validation shortcuts across all three cohorts containing repeated recordings per participant. **Strict Leak-Free** enforces `StratifiedGroupKFold(n_splits=5)` grouped by participant ID in the outer loop ($0.0\\%$ subject overlap between training and test folds), `StratifiedGroupKFold(n_splits=3)` grouped by participant ID in the inner hyperparameter selection loop, and channel-wise healthy-control standardization ($\\mu_{\\text{HC}}, \\sigma_{\\text{HC}}$ across the $19 \\times 4$ features) strictly within each outer training fold (a **Leave-One-Control-Out [LOCO] jackknife standardization sensitivity check** inside each training fold—where each training HC recording is standardized using the mean and SD of the remaining $N_{\\text{HC,train}} - 1$ training controls to eliminate finite-sample self-inclusion variance shrinkage—yielded consistent out-of-fold ROC-AUC: `Tuned_LogisticRegression` $0.608$ vs. $0.612$ in Study 1, $0.613$ vs. $0.656$ in Study 4, and $0.715$ vs. $0.723$ in Study 6c). **Leaky Pre-Standardization** computes channel-wise $z$-scores across the pooled cohort prior to participant-grouped splitting, introducing minor perturbation ($|\\Delta\\text{AUC}| \\le 0.050$). **Leaky Subject Mixing** replaces participant-grouped partitioning with standard session-level `StratifiedKFold`, causing $52.2\\%\\text{--}70.4\\%$ of test sessions to share a participant with the training fold. Notably, the **direction** of subject-mixing bias is governed by within-subject label concordance: in Study 4 (UCSD PD) and Study 6c (UNM PD All), repeated ON and OFF sessions from the same patient share a **concordant positive label** ($y=1$), so memorizing participant-specific spectral traits leaks the positive class label across folds and inflates ROC-AUC in a model-dependent manner by up to $+0.102$ (`GCN_OMST` $0.725 \\to 0.827$ [$+0.102$], `RandomForest` $0.673 \\to 0.731$ [$+0.058$], and `Tuned_LogisticRegression` $0.656 \\to 0.708$ [$+0.052$] in UCSD; `RandomForest` $0.716 \\to 0.807$ [$+0.091$] and `GCN_OMST` $0.681 \\to 0.762$ [$+0.081$] in UNM, whereas `Tuned_LogisticRegression` in UNM was $0.723 \\to 0.710$ [$\\Delta = -0.014$] under strict participant-grouped inner L2 tuning vs. $0.704 \\to 0.710$ [$+0.006$] under ungrouped inner tuning). Conversely, Study 1 (Sleep Deprivation) is a **paired crossover design with discordant labels** where each participant contributes one normal-sleep session ($y=0$) and one sleep-deprived session ($y=1$); when a participant's $y=0$ session is placed in the training fold and their paired $y=1$ session in the test fold, the negative shift observed in non-linear classifiers (`RandomForest` $\\Delta\\text{AUC} = -0.069$; `GCN_OMST` $\\Delta\\text{AUC} = -0.050$) is consistent with negative within-subject residual correlation across discordant sessions (fitting individual baseline traits on a participant's training session pulls predictions on their paired test session toward the opposite training-session label), depressing out-of-fold discrimination. Both 1,000-iteration session-level bootstrap 95% CIs and participant-cluster bootstrap 95% CIs (resampling unique participant IDs with replacement) are reported.*"
    )
    return "\n".join(lines)


def build_table_s20_gain_calibration() -> str:
    """Synthesize Markdown Table S20 (spectral representation, calibration & gain perturbation)."""
    csv_path = REPORTS_DIR / "benchmark_gain_and_calibration_ablation.csv"
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path)

    df_calib = df[df["experiment"] == "Calibration_Strategy"].copy()

    lines = [
        "### Table S20. Cross-Site Parkinson's Disease Calibration Ablation and Empirical Amplifier Gain Perturbation",
        "#### Panel A. Six-Pair Cross-Site Parkinson's Transfer Across Three Spectral Representations and Three Calibration Regimes",
        "| Source → Target | Spectral Representation | Calibration Regime | Tuned LR AUC [95% CI] | LR Bal. Acc. (p = 0.5) | LR Brier | Random Forest AUC [95% CI] | RF Bal. Acc. | GCN-OMST AUC [95% CI] |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    feat_map = {
        "specparam_periodic_residual": "**`specparam` Periodic Residual + χ**",
        "relative_band_power": "Relative Band Power + χ",
        "raw_log_band_power": "Unparameterized Raw log₁₀ Power + χ",
    }
    calib_map = {
        "Uncalibrated_Raw": "Uncalibrated Raw Transfer",
        "Global_Cohort_ZScore": "Global Cohort Z-Score",
        "Target_HC_Normative": "**Target-HC Normative**",
    }

    prev_pair = None
    prev_feat = None
    for _, row in df_calib.iterrows():
        pair_str = f"**{row['source_cohort']} → {row['target_cohort']}**"
        p_disp = pair_str if pair_str != prev_pair else ""
        f_disp = feat_map.get(row["feature_type"], row["feature_type"]) if (row["feature_type"] != prev_feat or pair_str != prev_pair) else ""
        prev_pair = pair_str
        prev_feat = row["feature_type"]

        c_disp = calib_map.get(row["calibration"], row["calibration"])
        lr_s = f"{row['auc_logistic_regression']:.3f} [{row['ci_lower_lr']:.3f}, {row['ci_upper_lr']:.3f}]"
        rf_s = f"{row['auc_random_forest']:.3f} [{row['ci_lower_rf']:.3f}, {row['ci_upper_rf']:.3f}]"
        gcn_s = f"{row['auc_gcn_omst']:.3f} [{row['ci_lower_gcn']:.3f}, {row['ci_upper_gcn']:.3f}]"

        lines.append(
            f"| {p_disp} | {f_disp} | {c_disp} | {lr_s} | {row['bacc_lr']:.3f} | {row['brier_lr']:.3f} | {rf_s} | {row['bacc_rf']:.3f} | {gcn_s} |"
        )

    # Summary means across the 6 directed pairs by Feature x Calibration
    lines.extend([
        "",
        "#### Panel B. Mean Cross-Site Transfer Performance Across All Six Directed PD Pairs by Representation and Calibration",
        "| Spectral Representation | Calibration Regime | Mean LR AUC | Mean LR Bal. Acc. (p = 0.5) | Mean LR Brier | Mean RF AUC | Mean RF Bal. Acc. | Mean GCN-OMST AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for f_type in ["specparam_periodic_residual", "relative_band_power", "raw_log_band_power"]:
        for idx_c, c_type in enumerate(["Uncalibrated_Raw", "Global_Cohort_ZScore", "Target_HC_Normative"]):
            sub = df_calib[(df_calib["feature_type"] == f_type) & (df_calib["calibration"] == c_type)]
            f_lbl = feat_map[f_type] if idx_c == 0 else ""
            c_lbl = calib_map[c_type]
            lines.append(
                f"| {f_lbl} | {c_lbl} | {sub['auc_logistic_regression'].mean():.3f} | {sub['bacc_lr'].mean():.3f} | {sub['brier_lr'].mean():.3f} | {sub['auc_random_forest'].mean():.3f} | {sub['bacc_rf'].mean():.3f} | {sub['auc_gcn_omst'].mean():.3f} |"
            )

    # Panel C: Amplifier Gain Multiplier Perturbation
    df_gain = df[df["experiment"] == "Amplifier_Gain_Ablation"].copy()
    lines.extend([
        "",
        "#### Panel C. Empirical Target-Site Amplifier Gain Perturbation (PSD_k(f) = k² PSD(f), Uncalibrated Cross-Site Transfer Averaged Across All Six PD Pairs)",
        "| Spectral Representation | Target Gain Regime (k) | Feature Drift |ΔX| | Prob. Drift |Δp_LR| | Linear LR AUC | LR Bal. Acc. (p = 0.5) | LR Brier | Non-Linear RF AUC | RF Bal. Acc. (p = 0.5) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    k_order = [
        ("0.5", "Uniform k = 0.5× (-6.0 dB)"),
        ("1.0", "Baseline k = 1.0× (0.0 dB)"),
        ("2.0", "Uniform k = 2.0× (+6.0 dB)"),
        ("5.0", "Uniform k = 5.0× (+14.0 dB)"),
        ("Heterogeneous_0.5x_5.0x", "**Heterogeneous Multi-Amplifier (50% 0.5×, 50% 5.0×)**"),
    ]

    for f_type in ["specparam_periodic_residual", "relative_band_power", "raw_log_band_power"]:
        sub_f = df_gain[df_gain["feature_type"] == f_type].copy()
        sub_f["k_str"] = sub_f["gain_multiplier_k"].astype(str)
        for idx_k, (k_key, k_lbl) in enumerate(k_order):
            rows_k = sub_f[sub_f["k_str"] == k_key]
            f_lbl = feat_map[f_type] if idx_k == 0 else ""
            lines.append(
                f"| {f_lbl} | {k_lbl} | {rows_k['feature_abs_drift'].mean():.4f} | {rows_k['prob_abs_drift_lr'].mean():.4f} | {rows_k['auc_logistic_regression'].mean():.3f} | {rows_k['bacc_lr'].mean():.3f} | {rows_k['brier_lr'].mean():.3f} | {rows_k['auc_random_forest'].mean():.3f} | {rows_k['bacc_rf'].mean():.3f} |"
            )

    lines.append(
        "\n*Note: All three spectral representations were extracted directly from raw BIDS Welch power spectral densities (`data/derivatives/pd_tricenter_welch_psds.pt`) across UCSD (`ds002778`, $N=46$), UNM OFF (`ds003490`, $N=50$), and Iowa (`ds004584`, $N=149$) using a unified spectral extraction pipeline so that multiplicative amplifier gain perturbations $\\mathrm{PSD}_k(f) = k^2 \\mathrm{PSD}(f)$ could be applied identically to the underlying power spectra (this direct PSD residual pipeline differs from the pre-computed `specparam` node tensors evaluated in Main Text Table 3 and Supplementary Table S16, accounting for the $\\sim 0.05\\text{--}0.07$ single-pair variation between Table 3 and Panel A such as `UNM -> UCSD` GCN $0.590$ vs. $0.662$ and `UCSD -> UNM` GCN $0.621$ vs. $0.565$). Four methodological findings emerge: (1) **`specparam` vs. Relative Band Power:** Both `specparam` periodic residuals and relative band power are algebraically invariant to multiplicative gain ($|\\Delta \\mathbf{X}| = 0.0000$). Across the 6 directed PD transfer pairs (Panel B), `specparam` periodic residuals achieve higher mean transfer ROC-AUC than relative band power for `RandomForest` across all three calibration regimes (`Target-HC Normative` $0.712$ vs. $0.693$, `Global Cohort Z-Score` $0.718$ vs. $0.702$, `Uncalibrated Raw` $0.707$ vs. $0.689$) and for `Tuned Logistic Regression` across all three regimes (`Target-HC Normative` $0.683$ vs. $0.653$, `Global Cohort Z-Score` $0.700$ vs. $0.656$, `Uncalibrated Raw` $0.708$ vs. $0.633$). For `GCN-OMST`, `specparam` outperformed relative band power under `Uncalibrated Raw` transfer ($0.735$ vs. $0.640$), whereas under `Global Cohort Z-Score` ($0.707$ vs. $0.723$) and `Target-HC Normative` ($0.698$ vs. $0.713$) relative band power yielded higher 6-pair mean AUC due to single-pair variance in `UNM_OFF -> UCSD` ($0.662$ vs. $0.812$, $N=16$ target HCs). (2) **Unperturbed Accuracy vs. Gain-Perturbation Robustness:** On unperturbed empirical recordings ($k=1.0$), unparameterized $\\log_{10}$ band power actually achieved slightly higher uncalibrated transfer ROC-AUC (`Tuned LR` $0.754$ vs. $0.708$; `RandomForest` $0.730$ vs. $0.707$) than `specparam` periodic residuals, demonstrating that spectral parameterization provides **algebraic robustness against cross-hardware gain mismatches** rather than higher baseline accuracy when recordings already share similar amplifier scaling. (3) **Trade-Offs of Unsupervised Target-HC Normative Calibration:** Standardizing target-site recordings against unlabeled local healthy controls (*unsupervised target-reference / transductive HC calibration*, requiring zero target patient data, diagnostic labels, or model retraining) improved fixed-threshold ($p=0.5$) Balanced Accuracy for `Tuned LR` on `specparam` ($0.582 \\to 0.635$) and relative power ($0.548 \\to 0.623$; `RandomForest` $0.655 \\to 0.670$; unparameterized `LR` $0.622 \\to 0.621$) and rescued `GCN-OMST` transfer on unparameterized raw $\\log_{10}$ power ($0.471 \\to 0.685$). However, on already gain-invariant `specparam` features, `Target-HC Normative` calibration slightly reduced 6-pair mean rank-order ROC-AUC for `Tuned LR` ($0.708 \\to 0.683$) and `GCN-OMST` ($0.735 \\to 0.698$) and increased mean `Tuned LR` Brier score across all representations (`specparam` $0.226 \\to 0.256$, relative power $0.238 \\to 0.258$, raw log power $0.211 \\to 0.262$) because estimating $76$ reference $(\\hat{\\mu}_{\\text{HC}}, \\hat{\\sigma}_{\\text{HC}})$ parameters from small control groups ($N=16$ HCs in UCSD, $N=25$ in UNM) injects finite-sample estimation variance. A **Split-Target-HC Holdout sensitivity check** (20 random 50/50 splits of target-site HCs where $50\\%$ of target HCs estimate $(\\hat{\\mu}_{\\text{HC}}, \\hat{\\sigma}_{\\text{HC}})$ and the disjoint held-out $50\\%$ target HCs plus all target PD patients form the evaluation set) yielded nearly identical 6-pair performance (`Tuned LR` mean AUC $0.680$ vs. $0.683$, Balanced Accuracy $0.633$ vs. $0.635$; `RandomForest` mean AUC $0.705$ vs. $0.712$). (4) **Uniform vs. Heterogeneous Amplifier Gain Scaling (Panel C):** Multiplying raw voltage by $k$ scales $\\mathrm{PSD}_k(f) = k^2 \\mathrm{PSD}(f)$ and shifts raw $\\log_{10}$ band power by $+2\\log_{10}k$. Under **uniform** target-cohort gain scaling, linear `LogisticRegression` ROC-AUC on raw log power remains numerically unchanged ($0.754$) solely because adding a constant scalar $c = 2\\log_{10}k$ to every test sample shifts linear logits by a constant $c \\sum_j w_j$, preserving within-cohort rank order while distorting predicted probabilities ($|\\Delta p_{\\text{LR}}| = 0.205\\text{--}0.339$), collapsing fixed-threshold ($p=0.5$) Balanced Accuracy to chance ($0.500$ at $k=5.0$, Brier score $0.211 \\to 0.342$), and collapsing non-linear `RandomForest` ROC-AUC from $0.730$ ($k=1.0$) to $0.446$ ($k=2.0$) and $0.474$ ($k=5.0$) as shifted features cross axis-aligned tree split thresholds. Under **heterogeneous multi-amplifier gain** ($50\\%$ of target sessions at $0.5\\times$ and $50\\%$ at $5.0\\times$, simulating a multi-amplifier clinical archive), even linear `LogisticRegression` ROC-AUC on raw log power degrades from $0.754$ to $0.636$ ($\\Delta\\text{AUC} = -0.118$) and `RandomForest` drops to $0.530$ ($\\Delta\\text{AUC} = -0.200$), whereas `specparam` periodic residuals maintain exact zero drift across all metrics and classifiers.*"
    )
    return "\n".join(lines)


def build_table_s21_multiseed() -> str:
    """Synthesize Markdown Table S21 (multi-seed repeated cross-validation stability)."""
    csv_path = REPORTS_DIR / "benchmark_multiseed_summary.csv"
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path)

    lines = [
        "### Table S21. Multi-Seed Repeated 5-Fold Cross-Validation Benchmark Across Five Independent Random Seeds",
        "| Cohort Contrast (N Sess / Subj) | Model | Ref. Seed (seed=42, Table 2) | 5-Seed Mean AUC ± SD | 5-Seed [Min, Max] | Paired ΔAUC vs. GCN_OMST (Mean ± SD) | 5-Seed Mean Bal. Acc. |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    prev_c = None
    for _, row in df.iterrows():
        n_s, n_p = int(row["n_sessions"]), int(row["n_participants"])
        n_lbl = f"{n_s}" if n_s == n_p else f"{n_s} / {n_p}"
        c_disp = f"**{row['cohort']} (N = {n_lbl})**" if row["cohort"] != prev_c else ""
        prev_c = row["cohort"]

        m_lbl = f"`{row['model']}`"
        if row["model"] == "GCN_OMST":
            m_lbl = "**`GCN_OMST`**"
            delta_s = "0.000 ± 0.000 (Ref)"
        else:
            if row["model"] == "GCN_Random":
                m_lbl = "*`GCN_Random`*"
            delta_s = f"{row['mean_delta_vs_omst']:+.3f} ± {row['std_delta_vs_omst']:.3f}"

        ref_s = f"{row['seed42_auc']:.3f}"
        mean_s = f"{row['mean_auc']:.3f} ± {row['std_auc']:.3f}"
        rng_s = f"[{row['min_auc']:.3f}, {row['max_auc']:.3f}]"
        bacc_s = f"{row['mean_bacc']:.3f}"

        lines.append(
            f"| {c_disp} | {m_lbl} | {ref_s} | {mean_s} | {rng_s} | {delta_s} | {bacc_s} |"
        )

    # Add 9-Primary-Contrast Grand Summary rows at bottom
    primary_9 = [
        "Study 1: Sleep Deprivation",
        "Study 2: Mumtaz MDD (EC)",
        "Study 3: Cavanagh Depression",
        "Study 4: UCSD Parkinson All",
        "Study 5a: Dementia AD vs CTL",
        "Study 5b: Dementia FTD vs CTL",
        "Study 5c: Dementia All vs CTL",
        "Study 6a: UNM PD-OFF vs CTL",
        "Study 7: Iowa Parkinson",
    ]
    df_p9 = df[df["cohort"].isin(primary_9)]
    for idx_m, m_name in enumerate(["GCN_OMST", "GCN_Random", "GCN_Identity", "Tuned_LogisticRegression", "Tuned_Ridge", "RandomForest", "Single_Theta"]):
        sub_m = df_p9[df_p9["model"] == m_name]
        c_title = "**Grand Mean Across 9 Primary Contrasts**" if idx_m == 0 else ""
        m_lbl = f"**`{m_name}`**" if m_name == "GCN_OMST" else f"`{m_name}`"
        ref_m = sub_m["seed42_auc"].mean()
        mean_m = sub_m["mean_auc"].mean()
        std_m = sub_m["std_auc"].mean()
        min_m = sub_m["min_auc"].mean()
        max_m = sub_m["max_auc"].mean()
        d_m = sub_m["mean_delta_vs_omst"].mean()
        ds_m = sub_m["std_delta_vs_omst"].mean()
        b_m = sub_m["mean_bacc"].mean()
        d_str = "0.000 ± 0.000 (Ref)" if m_name == "GCN_OMST" else f"{d_m:+.3f} ± {ds_m:.3f}"
        lines.append(
            f"| {c_title} | {m_lbl} | **{ref_m:.3f}** | **{mean_m:.3f} ± {std_m:.3f}** | [{min_m:.3f}, {max_m:.3f}] | {d_str} | {b_m:.3f} |"
        )

    lines.append(
        "\n*Note: Evaluates 5-fold cross-validation (`StratifiedGroupKFold` in outer and inner loops for repeated-session cohorts) with fold-nested healthy-control standardization across five independent random seeds (`42, 123, 456, 789, 2026`). The column **Ref. Seed (`seed=42`, Table 2)** reproduces the exact primary benchmark reported in Main Text Table 2 and Supplementary Table S11 (using the identical 7-value $C$-grid for `Tuned_LogisticRegression` and participant-grouped inner 3-fold CV in repeated-session cohorts), allowing direct side-by-side comparison between the reference split and multi-seed distributional stability. Across the 9 primary contrasts, 5-seed grand-mean ROC-AUC for `GCN_OMST` ($0.731$) trails `RandomForest` ($0.752$), `Tuned_Ridge` ($0.741$), `Tuned_LogisticRegression` ($0.738$), `GCN_Random` ($0.736$), and `GCN_Identity` ($0.736$). While seed standard deviation is modest ($\\text{SD} \\approx 0.004\\text{--}0.043$) in medium-to-large cohorts ($N \\ge 58$), partition sensitivity is substantial in small repeated-session cohorts—most notably Study 4 UCSD Parkinson's ($N=46$ sessions from $31$ unique participants, $\\sim 6$ participants per validation fold), where `GCN_OMST` spans $[0.463, 0.725]$ ($0.624 \\pm 0.105$) and the reference `seed=42` split ($0.725$) lay at the upper extreme of the 5-seed distribution, whereas the 1-parameter `Single_Theta` model remained stable ($0.719 \\pm 0.018$, $[0.694, 0.738]$). These multi-seed results indicate that no stable, reproducible classification advantage of OMST graph message passing over topology-free or classical regularized models is detectable in 19-channel resting EEG.*"
    )
    return "\n".join(lines)


def build_table_s22_riemannian() -> str:
    """Synthesize Markdown Table S22 (PLI eigenvalue audit and Riemannian vs Euclidean connectivity)."""
    eig_path = REPORTS_DIR / "pli_spd_eigenvalue_audit.csv"
    bench_path = REPORTS_DIR / "benchmark_riemannian_tangent_space.csv"
    if not eig_path.exists() or not bench_path.exists():
        return ""
    df_eig = pd.read_csv(eig_path)
    df_b = pd.read_csv(bench_path)

    lines = [
        "### Table S22. Mathematical Audit of 19 × 19 PLI Matrix Definiteness and Euclidean vs. SPD-Conditioned Riemannian Connectivity Benchmarks",
        "#### Panel A. Eigenvalue Spectrum Audit of Symmetrized 19 × 19 Phase Lag Index Matrices (I₁₉ + PLI) Across All 677 Primary Sessions",
        "| Dataset / Cohort | Primary Sessions (N) | Minimum Eigenvalue λ_min | Median λ_min | Max λ_min | Indefinite Matrices (λ_min ≤ 0) | % Indefinite (Raw & After +10⁻⁴I) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, r in df_eig.iterrows():
        c_lbl = f"**{r['cohort']}**" if "Total" in r["cohort"] else r["cohort"]
        ind_str = f"{int(r['n_indefinite_raw'])} / {int(r['n_sessions'])}"
        pct_str = f"**{r['pct_indefinite_raw']:.1f}%**" if r["pct_indefinite_raw"] > 0 else "0.0%"
        lines.append(
            f"| {c_lbl} | {int(r['n_sessions'])} | {r['min_eigenvalue']:+.4f} | {r['median_min_eigenvalue']:+.4f} | {r['max_min_eigenvalue']:+.4f} | {ind_str} | {pct_str} |"
        )

    lines.extend([
        "",
        "#### Panel B. 5-Fold Cross-Validated ROC-AUC [95% CI] Across Direct Euclidean PLI (171-D) and SPD-Conditioned Riemannian Tangent Space (190-D) Baselines",
        "| Cohort Contrast (N) | Classifier | Direct Euclidean Upper-Tri PLI (171-D, No SPD Assumption) | Eigenvalue-Adaptive Diagonal-Loaded SPD Tangent (190-D, λ_min ≥ 0.01) | Higham Spectral-Floor SPD Tangent (190-D, λ_i ≥ 10⁻⁴) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    cohorts_order = df_b["cohort"].unique()
    for c_name in cohorts_order:
        sub_c = df_b[df_b["cohort"] == c_name]
        n_samp = int(sub_c["n_samples"].iloc[0])
        for idx_clf, clf in enumerate(["LR", "Ridge", "RF"]):
            c_disp = f"**{c_name} (N = {n_samp})**" if idx_clf == 0 else ""
            r_euc = sub_c[(sub_c["representation"] == "Euclidean_UpperTri_171D") & (sub_c["classifier"] == clf)].iloc[0]
            r_ger = sub_c[(sub_c["representation"] == "Riemannian_Gershgorin_SPD_190D") & (sub_c["classifier"] == clf)].iloc[0]
            r_hig = sub_c[(sub_c["representation"] == "Riemannian_Higham_SPD_190D") & (sub_c["classifier"] == clf)].iloc[0]

            s_euc = f"{r_euc['auc']:.3f} [{r_euc['ci_lower']:.3f}, {r_euc['ci_upper']:.3f}]"
            s_ger = f"{r_ger['auc']:.3f} [{r_ger['ci_lower']:.3f}, {r_ger['ci_upper']:.3f}]"
            s_hig = f"{r_hig['auc']:.3f} [{r_hig['ci_lower']:.3f}, {r_hig['ci_upper']:.3f}]"

            lines.append(f"| {c_disp} | `{clf}` | {s_euc} | {s_ger} | {s_hig} |")

    lines.append(
        "\n*Note: **Panel A** audits the exact eigenvalue spectrum of the symmetrized $19 \\times 19$ connectivity matrix $\\mathbf{C} = \\mathbf{I}_{19} + \\mathrm{PLI}$ (unit diagonal, off-diagonal $\\mathrm{PLI}_{ij} \\in [0, 1]$) across all $677$ primary sessions. Unlike sample covariance matrices—which are Gram matrices and therefore positive semi-definite by construction—the Phase Lag Index is a non-linear sign statistic ($|\\langle \\operatorname{sign}(\\sin \\Delta\\phi_{ij}) \\rangle|$) that does not satisfy Mercer's kernel condition. Consequently, $85 / 677$ ($12.56\\%$) primary sessions possess negative minimum eigenvalues ($\\lambda_{\\min}$ down to $-7.3697$ in Study 2), and a naive $+10^{-4}\\mathbf{I}$ shift leaves all $85$ matrices indefinite, rendering direct real-valued matrix logarithms undefined without SPD conditioning. **Panel B** therefore benchmarks three mathematically well-defined functional-connectivity representations under identical leak-free 5-fold cross-validation splits and nested hyperparameter grids (`LR`, `Ridge`, `RF`): (1) **Direct Euclidean Upper-Triangular PLI (`171-D`)**, which vectorizes the $19 \\times 18 / 2 = 171$ unique off-diagonal phase-coupling weights directly without any SPD assumption or matrix logarithm; (2) **Eigenvalue-Adaptive Diagonal-Loaded SPD Tangent Space (`190-D`)**, which applies an exact eigenvalue-adaptive diagonal shift $\\mathbf{C}_{\\text{SPD}} = \\mathbf{C} + (\\max(0, -\\lambda_{\\min}(\\mathbf{C})) + 0.01)\\mathbf{I}_{19}$ to guarantee strict positive definiteness ($\\lambda_i \\ge 0.01$) while preserving eigenvectors and exact pairwise eigenvalue spacing ($\\lambda_i - \\lambda_j$) prior to Log-Euclidean tangent projection $\\mathbf{S} = \\operatorname{logm}(\\mathbf{C}_{\\text{SPD}})$; and (3) **Higham Spectral-Floor SPD Tangent Space (`190-D`)**, which projects $\\mathbf{C}$ onto the Frobenius-nearest SPD cone via eigendecomposition $\\mathbf{V} \\operatorname{diag}(\\max(\\lambda_i, 10^{-4})) \\mathbf{V}^T$ followed by $\\operatorname{logm}$. Across all nine primary contrasts (and nine of the ten evaluated contrasts overall, with the sole exception of the secondary contrast Study 6b UNM PD-ON vs. CTL where `Direct Euclidean Upper-Triangular PLI + Ridge` reached $0.773$ vs. $0.746$ for `Single_Theta` and $0.721$ for `RandomForest`), direct Euclidean PLI and both SPD-conditioned tangent-space projections achieve comparable moderate discrimination (mean best AUC $\\approx 0.65\\text{--}0.67$) and trail the best-performing node-feature classifiers (`0.629`–`0.933` in Table 2; note that when compared specifically against the univariate `Single_Theta` baseline, connectivity classifiers scored higher in Studies 1, 2, and 5b, but remained below the top multivariate node-feature classifiers in all nine primary contrasts), demonstrating that the limited diagnostic performance of PLI connectivity on 19-channel resting EEG reflects the lower signal-to-noise ratio of sensor-level phase synchronization rather than matrix-logarithm conditioning.*"
    )
    return "\n".join(lines)


def build_table_s23_meta_analysis() -> str:
    """Synthesize Markdown Table S23 (independent random-effects meta-analysis of frontal theta polarity)."""
    csv_path = REPORTS_DIR / "meta_analysis_frontal_theta_polarity.csv"
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path)

    lines = [
        "### Table S23. Random-Effects Meta-Analysis and Independence Sensitivity Suite for Resting Frontal Periodic Theta Polarity",
        "| Analysis Specification & Cohort Set | Clinical / Physiological Domain | k Independent Studies | Pooled Cohen's d | DL 95% CI | Ad-Hoc Variance-Floored HKSJ 95% CI | z-Score | DL p-Value | Cochran's Q (df, p_Q) | τ² (τ) | I² | Subgroup Polarity Test Q_between (p_between) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    domain_map = {
        "Depressive_HypoOscillatory_Domain": "**Depressive / Hypo-Oscillatory Domain**",
        "Neurodegenerative_Slowing_Domain": "**Neurodegenerative Slowing Domain**",
        "Pathological_and_SleepPressure_Slowing_Domain": "**Slowing + Acute Sleep-Pressure Domain**",
        "Overall_Pooled": "Unstratified Pooled Total",
    }

    prev_set = None
    for _, row in df.iterrows():
        a_set = row["analysis_set"]
        set_disp = f"**{a_set}**" if a_set != prev_set else ""
        prev_set = a_set

        dom_disp = domain_map.get(row["domain_group"], row["domain_group"])
        k_val = int(row["k_studies"])
        d_str = f"**{row['pooled_d']:+.3f}**" if row["domain_group"] != "Overall_Pooled" else f"{row['pooled_d']:+.3f}"
        dl_ci = f"[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]"
        hksj_ci = f"[{row['hksj_ci_lower']:+.3f}, {row['hksj_ci_upper']:+.3f}]"
        z_str = f"{row['z_score']:+.2f}"
        p_str = fmt_p_sci(float(row["p_value"]), bold_sig=(row["domain_group"] != "Overall_Pooled"))
        q_str = f"{row['Q_stat']:.2f} ({int(row['Q_df'])}, {row['Q_p_value']:.2f})" if row["Q_p_value"] >= 0.001 else f"{row['Q_stat']:.2f} ({int(row['Q_df'])}, {fmt_p_sci(float(row['Q_p_value']))})"
        tau_str = f"{row['tau2']:.3f} ({row['tau']:.3f})"
        i2_str = f"{row['I2_percent']:.1f}%"

        if row["domain_group"] == "Depressive_HypoOscillatory_Domain":
            qb_str = f"**Q = {row['Q_between']:.2f}** ({fmt_p_sci(float(row['p_between']), bold_sig=True)})"
        else:
            qb_str = "''"

        lines.append(
            f"| {set_disp} | {dom_disp} | {k_val} | {d_str} | {dl_ci} | {hksj_ci} | {z_str} | {p_str} | {q_str} | {tau_str} | {i2_str} | {qb_str} |"
        )

    lines.append(
        "\n*Note: All individual study effect sizes and analytical standard errors are read directly from `reports/master_results.json` (identical to Main Text Table 1 and Supplementary Table S4). To guarantee strict statistical independence—eliminating shared healthy control groups (e.g., Study 5 AD vs. HC and FTD vs. HC share $N=29$ controls) and repeated within-patient medication sessions (Study 4 UCSD ON/OFF and Study 6 UNM ON/OFF)—the **Primary Independent Meta-Analysis (`Primary_Independent_K6`)** is restricted to $K = 6$ mutually exclusive between-subject cohorts ($N = 495$ unique participants): the **Depressive / Hypo-Oscillatory Domain** ($k = 2$: Study 2 Outpatient MDD EC [$d = -0.826 \\pm 0.273$] and Study 3 Univ. Depression Screening [$d = -0.676 \\pm 0.195$]) versus the **Neurodegenerative Slowing Domain** ($k = 4$: Study 4a UCSD PD-OFF [$d = +0.559 \\pm 0.366$], Study 5c Combined Dementia [$d = +0.903 \\pm 0.236$], Study 6a UNM PD-OFF [$d = +0.865 \\pm 0.296$], and Study 7 Iowa PD [$d = +0.899 \\pm 0.184$]). Both standard DerSimonian–Laird (DL) normal-theory 95% CIs and small-sample **Ad-Hoc Variance-Floored (Modified/Truncated) Hartung–Knapp–Sidik–Jonkman (HKSJ)** $t$-distribution 95% CIs ($df = k - 1$, using $\\operatorname{SE}_{\\text{HKSJ-floored}} = \\sqrt{\\max(1, Q/(k-1))} \\cdot \\operatorname{SE}_{\\text{DL}}$ so that intervals do not artificially narrow below the DL standard error when $Q < k - 1$; unfloored classical HKSJ intervals using $\\sqrt{Q/(k-1)} \\cdot \\operatorname{SE}_{\\text{DL}}$ are $[-1.622, +0.170]$ for the $k=2$ depressive domain and $[+0.662, +1.051]$ for the $k=4$ neurodegenerative domain) are reported alongside Cochran's $Q$, between-study variance $\\tau^2$ ($\\tau$), and $I^2$. Note two small-$k$ statistical properties: (i) at $k = 2\\text{--}4$, point estimates $\\tau^2 = 0.000$ and $I^2 = 0.0\\%$ have wide uncertainty and indicate only that observed dispersion did not exceed within-study sampling variance; and (ii) for the $k = 2$ depressive domain, the variance-floored HKSJ interval ($[-2.744, +1.291]$, $p_{\\text{HKSJ}} = 0.1370$) is wide by mathematical construction because $df = k - 1 = 1$ scales the standard error by $t_{0.975, 1} = 12.706$, so inference for the depressive domain relies on the convergent significance of both individual cohorts ($p = 0.0027$ and $p = 4.96 \\times 10^{-4}$) alongside the DL pooled estimate ($d_{\\text{RE}} = -0.726$, 95% CI $[-1.037, -0.415]$). Across all five strictly participant-independent sensitivity checks—replacing UCSD PD-OFF with participant-averaged UCSD ($N=15$ unique PD patients [mean of ON & OFF] vs. $16$ HCs, $d = +0.918 \\pm 0.378$, `Sens1`), replacing PD-OFF with PD-ON states (`Sens2`), replacing Combined Dementia with AD-only (`Sens3`) or FTD-only (`Sens4`) without control overlap, or adding the Study 1 Acute Sleep Deprivation crossover cohort converted to the Morris–DeShon independent-groups equivalent scale ($d_{\\text{IG}} = +0.267 \\pm 0.112$ accounting for within-subject correlation $r = 0.5709$ across $N=71$ pairs; paired $d_z = +0.284 \\pm 0.121$, $K = 7$, `Sens5`, which introduces expected heterogeneity $\\tau^2 = 0.100, I^2 = 70.0\\%$ between acute homeostatic sleep debt and chronic neurodegeneration)—and across all four Leave-One-Cohort-Out deletions, the between-domain polarity contrast remains robust ($Q_{\\text{between}} = 45.67\\text{--}66.82$, $p = 1.40 \\times 10^{-11}$ to $3.33 \\times 10^{-16}$).*"
    )
    return "\n".join(lines)


def main():
    """Synthesize Tables S19–S23 and write markdown output file."""
    print("🚀 Synthesizing Audited Supplementary Extended Tables S19 - S23...")
    t19 = build_table_s19_leakage()
    t20 = build_table_s20_gain_calibration()
    t21 = build_table_s21_multiseed()
    t22 = build_table_s22_riemannian()
    t23 = build_table_s23_meta_analysis()

    out_file = REPORTS_DIR / "supplementary_extended_tables_s19_s23.md"
    content = "\n\n---\n\n".join([t for t in [t19, t20, t21, t22, t23] if t])
    out_file.write_text(content, encoding="utf-8")
    print(f"✅ Saved extended tables to: {out_file}")


if __name__ == "__main__":
    main()
