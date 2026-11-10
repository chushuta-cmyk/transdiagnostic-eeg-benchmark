#!/usr/bin/env python3
"""
Supplementary Extended Tables Synthesis (Tables S19 - S23).

Compiles markdown statistical tables from raw benchmark CSV outputs:
- Table S19: Empirical quantification of validation data leakage
- Table S20: Cross-site PD calibration ablation & gain perturbation
- Table S21: Multi-seed repeated cross-validation benchmark
- Table S22: PLI eigenvalue spectrum audit & Riemannian tangent space comparison
- Table S23: Independent random-effects meta-analysis of frontal theta polarity

Formatted strictly as clean Markdown with pure Unicode text (ZERO math dollar signs)
so all numbers and symbols are completely preserved across Word and docx text extractors.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = BENCHMARK_ROOT / "reports"


def fmt_p_sci(p: float, bold_sig: bool = False) -> str:
    """Format float p-value into clean scientific notation or decimal string with Unicode exponents."""
    if pd.isna(p):
        return "—"
    if p < 0.001:
        mant, exp = f"{p:.2e}".split("e")
        exp_i = int(exp)
        s = f"{mant} × 10^{exp_i}"
    else:
        s = f"{p:.4f}"
    if bold_sig and p < 0.05:
        s = f"**{s}**"
    return s


def build_table_s19_leakage() -> str:
    """Synthesize Markdown Table S19 (quantification of validation data leakage)."""
    csv_path = REPORTS_DIR / "benchmark_leakage_ablation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required input: {csv_path}")
    df = pd.read_csv(csv_path)

    lines = [
        "### Table S19. Empirical Quantification of Data Leakage Across Repeated-Session Cohorts",
        "| Cohort (Label Structure) | Evaluation Regime | Model | N (Sess / Subj) | Fold Subject Overlap | Out-of-Fold ROC-AUC | Session Bootstrap 95% CI | Participant-Cluster 95% CI | Bal. Acc. | ΔAUC vs. Participant-Grouped Reference |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    design_map = {
        "Paired_Crossover_Discordant": "Paired Crossover (y in {0,1} per subj)",
        "Repeated_Sessions_Concordant": "Repeated Case Sessions (y = 1 concordant)",
    }

    for _, row in df.iterrows():
        c_disp = f"**{row['cohort']}** ({design_map.get(row['design_type'], row['design_type'])})"

        if row["regime"] == "Strict_Leak_Free":
            reg_fmt = "**Participant-Grouped Reference (`StratifiedGroupKFold`)**"
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
        "\n*Note: Evaluates the empirical impact of validation shortcuts across all three cohorts containing repeated recordings per participant. **Participant-Grouped Reference** enforces `StratifiedGroupKFold(n_splits=5)` grouped by participant ID in the outer loop (0.0% subject overlap between training and test folds), `StratifiedGroupKFold(n_splits=3)` grouped by participant ID in the inner hyperparameter selection loop, and channel-wise healthy-control standardization (mean and SD across the 19 × 4 features) strictly within each outer training fold (a **Leave-One-Control-Out [LOCO] jackknife standardization sensitivity check** inside each training fold—where each training HC recording is standardized using the mean and SD of the remaining N_HC,train - 1 training controls to eliminate finite-sample self-inclusion variance shrinkage—yielded consistent out-of-fold ROC-AUC: `Tuned_LogisticRegression` 0.608 vs. 0.612 in Study 1, 0.613 vs. 0.656 in Study 4, and 0.715 vs. 0.723 in Study 6c). **Leaky Pre-Standardization** computes channel-wise z-scores across the pooled cohort prior to participant-grouped splitting, changed AUC only slightly (ΔAUC from -0.050 to +0.002; mean -0.010), so global pre-standardization is not the main source of optimistic bias in this benchmark. **Leaky Subject Mixing** replaces participant-grouped partitioning with standard session-level `StratifiedKFold`, causing 52.2%–70.4% of test sessions to share a participant with the training fold. The substantive source of inflation is session mixing: its mean ΔAUC was +0.029 (range -0.069 to +0.102). The **direction** of subject-mixing bias is governed by within-subject label concordance: in Study 4 (UCSD PD) and Study 6c (UNM PD All), repeated ON and OFF sessions from the same patient share a **concordant positive label** (y = 1), so memorizing participant-specific spectral traits leaks the positive class label across folds and inflates ROC-AUC in a model-dependent manner by up to +0.102 (`GCN_OMST` 0.725 → 0.827 [+0.102], `RandomForest` 0.673 → 0.731 [+0.058], and `Tuned_LogisticRegression` 0.656 → 0.708 [+0.052] in UCSD; `RandomForest` 0.716 → 0.807 [+0.091] and `GCN_OMST` 0.681 → 0.762 [+0.081] in UNM, whereas `Tuned_LogisticRegression` in UNM was 0.723 → 0.710 [Δ = -0.014] under strict participant-grouped inner L2 tuning vs. 0.704 → 0.710 [+0.006] under ungrouped inner tuning). Conversely, Study 1 (Sleep Deprivation) is a **paired crossover design with discordant labels** where each participant contributes one normal-sleep session (y = 0) and one sleep-deprived session (y = 1); when a participant's y = 0 session is placed in the training fold and their paired y = 1 session in the test fold, the negative shift observed in non-linear classifiers (`RandomForest` ΔAUC = -0.069; `GCN_OMST` ΔAUC = -0.050) is consistent with negative within-subject residual correlation across discordant sessions (fitting individual baseline traits on a participant's training session pulls predictions on their paired test session toward the opposite training-session label), depressing out-of-fold discrimination. Both 1,000-iteration session-level bootstrap 95% CIs and participant-cluster bootstrap 95% CIs (resampling unique participant IDs with replacement) are reported.*"
    )
    return "\n".join(lines)


def build_table_s20_gain_calibration() -> str:
    """Synthesize Markdown Table S20 (spectral representation, calibration & gain perturbation)."""
    csv_path = REPORTS_DIR / "benchmark_gain_and_calibration_ablation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required input: {csv_path}")
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
        "raw_log_band_power": "Unparameterized Raw log10 Power + χ",
    }
    calib_map = {
        "Uncalibrated_Raw": "Uncalibrated Raw Transfer",
        "Global_Cohort_ZScore": "Global Cohort Z-Score",
        "Target_HC_Normative": "**Target-HC Normative**",
    }

    for _, row in df_calib.iterrows():
        pair_str = f"**{row['source_cohort']} → {row['target_cohort']}**"
        f_disp = feat_map.get(row["feature_type"], row["feature_type"])
        c_disp = calib_map.get(row["calibration"], row["calibration"])
        lr_s = f"{row['auc_logistic_regression']:.3f} [{row['ci_lower_lr']:.3f}, {row['ci_upper_lr']:.3f}]"
        rf_s = f"{row['auc_random_forest']:.3f} [{row['ci_lower_rf']:.3f}, {row['ci_upper_rf']:.3f}]"
        gcn_s = f"{row['auc_gcn_omst']:.3f} [{row['ci_lower_gcn']:.3f}, {row['ci_upper_gcn']:.3f}]"

        lines.append(
            f"| {pair_str} | {f_disp} | {c_disp} | {lr_s} | {row['bacc_lr']:.3f} | {row['brier_lr']:.3f} | {rf_s} | {row['bacc_rf']:.3f} | {gcn_s} |"
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
            f_lbl = feat_map[f_type]
            c_lbl = calib_map[c_type]
            lines.append(
                f"| {f_lbl} | {c_lbl} | {sub['auc_logistic_regression'].mean():.3f} | {sub['bacc_lr'].mean():.3f} | {sub['brier_lr'].mean():.3f} | {sub['auc_random_forest'].mean():.3f} | {sub['bacc_rf'].mean():.3f} | {sub['auc_gcn_omst'].mean():.3f} |"
            )

    # Panel C: Amplifier Gain Multiplier Perturbation
    df_gain = df[df["experiment"] == "Amplifier_Gain_Ablation"].copy()
    lines.extend([
        "",
        "#### Panel C. Empirical Target-Site Amplifier Gain Perturbation (PSD_k(f) = k² · PSD(f), Uncalibrated Cross-Site Transfer Averaged Across All Six PD Pairs)",
        "| Spectral Representation | Target Gain Regime (k) | Feature Drift \\|ΔX\\| | Prob. Drift \\|Δp_LR\\| | Linear LR AUC | LR Bal. Acc. (p = 0.5) | LR Brier | Non-Linear RF AUC | RF Bal. Acc. (p = 0.5) |",
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
            f_lbl = feat_map[f_type]
            lines.append(
                f"| {f_lbl} | {k_lbl} | {rows_k['feature_abs_drift'].mean():.4f} | {rows_k['prob_abs_drift_lr'].mean():.4f} | {rows_k['auc_logistic_regression'].mean():.3f} | {rows_k['bacc_lr'].mean():.3f} | {rows_k['brier_lr'].mean():.3f} | {rows_k['auc_random_forest'].mean():.3f} | {rows_k['bacc_rf'].mean():.3f} |"
            )

    lines.append(
        "\n*Note: All three spectral representations were extracted directly from raw BIDS Welch power spectral densities (`data/derivatives/pd_tricenter_welch_psds.pt`) across UCSD (`ds002778`, N = 46), UNM OFF (`ds003490`, N = 50), and Iowa (`ds004584`, N = 149) using a unified spectral extraction pipeline so that multiplicative amplifier gain perturbations PSD_k(f) = k² · PSD(f) could be applied identically to the underlying power spectra (this direct PSD residual pipeline differs from the pre-computed `specparam` node tensors evaluated in Main Text Table 3 and Supplementary Table S16, accounting for the ~0.05–0.07 single-pair variation between Table 3 and Panel A such as `UNM -> UCSD` GCN 0.590 vs. 0.662 and `UCSD -> UNM` GCN 0.621 vs. 0.565). Note that the `Uncalibrated Raw Transfer` regime is a reference condition that is trivially exposed to gain mismatch, because any target-site z-scoring cancels a uniform gain; the informative comparison for uniform gain is therefore between representations, not between calibration regimes. Four methodological findings emerge: (1) **`specparam` vs. Relative Band Power:** Both `specparam` periodic residuals and relative band power are algebraically invariant to multiplicative gain (|ΔX| = 0.0000). Across the 6 directed PD transfer pairs (Panel B), `specparam` periodic residuals achieve higher mean transfer ROC-AUC than relative band power for `RandomForest` across all three calibration regimes (`Target-HC Normative` 0.712 vs. 0.693, `Global Cohort Z-Score` 0.718 vs. 0.702, `Uncalibrated Raw` 0.707 vs. 0.689) and for `Tuned Logistic Regression` across all three regimes (`Target-HC Normative` 0.683 vs. 0.653, `Global Cohort Z-Score` 0.700 vs. 0.656, `Uncalibrated Raw` 0.708 vs. 0.633). For `GCN-OMST`, `specparam` outperformed relative band power under `Uncalibrated Raw` transfer (0.735 vs. 0.640), whereas under `Global Cohort Z-Score` (0.707 vs. 0.723) and `Target-HC Normative` (0.698 vs. 0.713) relative band power yielded higher 6-pair mean AUC due to single-pair variance in `UNM_OFF -> UCSD` (0.662 vs. 0.812, N = 16 target HCs). (2) **Unperturbed Accuracy vs. Gain-Perturbation Robustness:** On unperturbed empirical recordings (k = 1.0), unparameterized log10 band power actually achieved slightly higher uncalibrated transfer ROC-AUC (`Tuned LR` 0.754 vs. 0.708; `RandomForest` 0.730 vs. 0.707) than `specparam` periodic residuals, demonstrating that spectral parameterization provides **algebraic robustness against cross-hardware gain mismatches** rather than higher baseline accuracy when recordings already share similar amplifier scaling. (3) **Trade-Offs of Unsupervised Target-HC Normative Calibration:** Standardizing target-site recordings against local healthy controls (*target-reference / transductive HC calibration*; it requires no target patient data and no model retraining, but it does require knowing which target-site participants are healthy controls, and those controls are also part of the evaluation set except in the Split-Target-HC check) improved fixed-threshold (p = 0.5) Balanced Accuracy for `Tuned LR` on `specparam` (0.582 → 0.635) and relative power (0.548 → 0.623; `RandomForest` 0.655 → 0.670; unparameterized `LR` 0.622 → 0.621) and rescued `GCN-OMST` transfer on unparameterized raw log10 power (0.471 → 0.685). However, on `specparam` features, which are already invariant to multiplicative gain, `Target-HC Normative` calibration slightly reduced 6-pair mean rank-order ROC-AUC for `Tuned LR` (0.708 → 0.683) and `GCN-OMST` (0.735 → 0.698) and increased mean `Tuned LR` Brier score across all representations (`specparam` 0.226 → 0.256, relative power 0.238 → 0.258, raw log power 0.211 → 0.262) because estimating 76 reference (μ_HC, σ_HC) parameters from small control groups (N = 16 HCs in UCSD, N = 25 in UNM) injects finite-sample estimation variance. A **Split-Target-HC Holdout sensitivity check** (20 random 50/50 splits of target-site HCs where 50% of target HCs estimate (μ_HC, σ_HC) and the disjoint held-out 50% are evaluated as test controls alongside target patients) yielded consistent ROC-AUC (mean across 20 splits for `specparam` `Tuned LR`: `Iowa -> UNM` 0.736 ± 0.021 vs. 0.737 under full HC; `UNM -> UCSD` 0.708 ± 0.038 vs. 0.719), indicating that small target-HC samples introduce modest finite-sample variance; because the held-out controls are disjoint from the reference controls, this check is the only analysis here in which the calibration sample does not overlap the evaluation set. (4) **Gain-Perturbation Sensitivity of Unparameterized Power:** Under uniform target gain scaling (Panel C), rank-order ROC-AUC of linear classifiers on unparameterized log10 power is preserved (because a uniform constant offset shifts linear decision logits by a constant scalar), but probability calibration degrades (|Δp_LR| = 0.205–0.339) and fixed-threshold (p = 0.5) Balanced Accuracy collapses from 0.622 to 0.444 (k = 0.5) and 0.535 (k = 5.0). Under **heterogeneous multi-amplifier scaling** (Panel C row 5), unparameterized log10 power exhibits steep performance degradation for both linear models (`Tuned LR` ROC-AUC drops from 0.754 to 0.636, Bal. Acc. drops to 0.505) and non-linear models (`RandomForest` ROC-AUC drops from 0.730 to 0.446, Bal. Acc. drops to 0.474), whereas `specparam` periodic residuals remain perfectly invariant (|ΔX| = 0.0000, |Δp| = 0.0000, ROC-AUC = 0.708, Bal. Acc. = 0.582, RF AUC = 0.707).*",
    )
    return "\n".join(lines)


def build_table_s21_multiseed() -> str:
    """Synthesize Markdown Table S21 (five-seed repeated cross-validation stability)."""
    csv_path = REPORTS_DIR / "benchmark_multiseed_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required input for Table S21: {csv_path}")
    df = pd.read_csv(csv_path)

    model_order = ["GCN_OMST", "GCN_Random", "GCN_Identity", "Tuned_LogisticRegression",
                   "Tuned_Ridge", "RandomForest", "Single_Theta"]
    lines = [
        "### Table S21. Five-Seed Repeated Cross-Validation Stability for Seven Model Configurations Across Ten Contrasts (350 Evaluations)",
        "| Cohort Contrast | Model Identifier | Ref. Seed (seed = 42, Table 2) | 5-Seed Mean AUC ± SD | [Min, Max] | ΔAUC vs. GCN-OMST (Mean ± SD) | Mean Bal. Acc. |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cohort in df["cohort"].unique():
        sub = df[df["cohort"] == cohort].set_index("model")
        n_s, n_p = int(sub["n_sessions"].iloc[0]), int(sub["n_participants"].iloc[0])
        n_lbl = f"{n_s}" if n_s == n_p else f"{n_s} / {n_p}"
        c_disp = f"**{cohort} (N = {n_lbl})**"
        for m in model_order:
            row = sub.loc[m]
            if m == "GCN_OMST":
                m_lbl, delta_s = "**`GCN_OMST`**", "0.000 ± 0.000 (Ref)"
            else:
                m_lbl = f"`{m}`"
                delta_s = f"{row['mean_delta_vs_omst']:+.3f} ± {row['std_delta_vs_omst']:.3f}"
            lines.append(
                f"| {c_disp} | {m_lbl} | {row['seed42_auc']:.3f} | {row['mean_auc']:.3f} ± {row['std_auc']:.3f} | "
                f"[{row['min_auc']:.3f}, {row['max_auc']:.3f}] | {delta_s} | {row['mean_bacc']:.3f} |"
            )

    primary_9 = [c for c in df["cohort"].unique() if not c.startswith("Study 6b")]
    d9 = df[df["cohort"].isin(primary_9)]
    pv = d9.pivot(index="cohort", columns="model", values="mean_auc")
    for m in model_order:
        sub_m = d9[d9["model"] == m]
        m_lbl = f"**`{m}`**" if m == "GCN_OMST" else f"`{m}`"
        if m == "GCN_OMST":
            d_str = "0.000 ± 0.000 (Ref)"
        else:
            d_str = f"{sub_m['mean_delta_vs_omst'].mean():+.3f} ± {sub_m['std_delta_vs_omst'].mean():.3f}"
        lines.append(
            f"| **Grand mean across 9 primary contrasts** | {m_lbl} | **{sub_m['seed42_auc'].mean():.3f}** | "
            f"**{sub_m['mean_auc'].mean():.3f} ± {sub_m['std_auc'].mean():.3f}** | per-contrast ranges above | {d_str} | {sub_m['mean_bacc'].mean():.3f} |"
        )

    gm = {m: d9[d9["model"] == m]["mean_auc"].mean() for m in model_order}
    wil = {}
    for m in model_order[1:]:
        r = wilcoxon(pv["GCN_OMST"], pv[m], method="exact")
        wil[m] = float(r.pvalue)
    s4 = df[df["cohort"].str.startswith("Study 4")].set_index("model")
    nz_sd = df[df["std_auc"] > 0]["std_auc"]
    n_zero = int((df["std_auc"] == 0).sum())

    note = (
        "\n*Note: Seven configurations (`GCN_OMST`, `GCN_Random`, `GCN_Identity`, `Tuned_LogisticRegression`, `Tuned_Ridge`, "
        "`RandomForest`, `Single_Theta`) were re-evaluated on ten contrasts (the nine primary contrasts plus Study 6b UNM PD-ON) "
        "across five seeds (`42, 123, 456, 789, 2026`), giving 7 × 10 × 5 = 350 evaluations; the other five configurations of the "
        "12-model benchmark (`GCN_Full`, `Linear_SVM`, `RBF_SVM`, `GBDT`, `Single_Chi`) were not repeated across seeds. Each seed re-draws the "
        "outer and inner cross-validation partitions (`StratifiedGroupKFold` for repeated-session cohorts, `StratifiedKFold` otherwise), "
        "the GCN weight initialization, the random-graph adjacency of `GCN_Random`, and the `RandomForest` seed. Fold-nested "
        "healthy-control standardization was applied inside every outer training fold. The column **Ref. Seed** reproduces Main Text Table 2 "
        "and Supplementary Table S11 exactly (all 70 cohort-by-model values match). "
        f"Across the nine primary contrasts the 5-seed grand-mean ROC-AUC was {gm['GCN_OMST']:.3f} for `GCN_OMST`, compared with "
        f"{gm['RandomForest']:.3f} (`RandomForest`), {gm['Tuned_Ridge']:.3f} (`Tuned_Ridge`), {gm['Tuned_LogisticRegression']:.3f} "
        f"(`Tuned_LogisticRegression`), {gm['GCN_Random']:.3f} (`GCN_Random`), {gm['GCN_Identity']:.3f} (`GCN_Identity`), and "
        f"{gm['Single_Theta']:.3f} (`Single_Theta`). Exact two-sided Wilcoxon signed-rank tests on the nine per-contrast 5-seed means "
        f"did not detect a difference between `GCN_OMST` and any comparator (p = {min(wil.values()):.3f}–{max(wil.values()):.3f}; "
        f"`RandomForest` p = {wil['RandomForest']:.3f}), although with n = 9 non-independent contrasts these tests have low power. "
        f"Non-zero across-seed SDs ranged from {nz_sd.min():.3f} to {nz_sd.max():.3f}. In Study 1 every participant contributes exactly one "
        "session per class, `StratifiedGroupKFold` returned the same partition for all five seeds, and the deterministic estimators "
        f"(`Tuned_LogisticRegression`, `Tuned_Ridge`, `Single_Theta`; {n_zero} model-by-cohort cells) therefore have zero SD; the SD of the "
        "remaining models in that cohort reflects stochastic initialization only. Partition sensitivity was largest in Study 4 UCSD Parkinson's "
        "disease (46 sessions from 31 participants, about six participants per validation fold): "
        f"`GCN_OMST` spanned [{s4.loc['GCN_OMST','min_auc']:.3f}, {s4.loc['GCN_OMST','max_auc']:.3f}] "
        f"({s4.loc['GCN_OMST','mean_auc']:.3f} ± {s4.loc['GCN_OMST','std_auc']:.3f}) and the reference seed lay at the upper end of that range, whereas the "
        f"one-parameter `Single_Theta` model was more stable ({s4.loc['Single_Theta','mean_auc']:.3f} ± {s4.loc['Single_Theta','std_auc']:.3f}, "
        f"[{s4.loc['Single_Theta','min_auc']:.3f}, {s4.loc['Single_Theta','max_auc']:.3f}]). These results show no reproducible advantage of "
        "the tested GCN over topology-free or classical regularized models in this design; they do not exclude small effects.*"
    )
    lines.append(note)
    return "\n".join(lines)


def build_table_s22_riemannian() -> str:
    """Synthesize Markdown Table S22 (PLI eigenvalue audit and Riemannian vs Euclidean connectivity)."""
    eig_path = REPORTS_DIR / "pli_spd_eigenvalue_audit.csv"
    bench_path = REPORTS_DIR / "benchmark_riemannian_tangent_space.csv"
    if not eig_path.exists() or not bench_path.exists():
        raise FileNotFoundError(f"Missing required input: {eig_path} or {bench_path}")
    df_eig = pd.read_csv(eig_path)
    df_b = pd.read_csv(bench_path)

    lines = [
        "### Table S22. Mathematical Audit of 19 × 19 PLI Matrix Definiteness and Euclidean vs. SPD-Conditioned Riemannian Connectivity Benchmarks",
        "#### Panel A. Eigenvalue Spectrum Audit of Symmetrized 19 × 19 Phase Lag Index Matrices (I_19 + PLI) Across All 677 Primary Sessions",
        "| Dataset / Cohort | Primary Sessions (N) | Minimum Eigenvalue λ_min | Median λ_min | Max λ_min | Indefinite Matrices (λ_min ≤ 0) | % Indefinite (Raw & After +10^-4I) |",
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
        "| Cohort Contrast (N) | Classifier | Direct Euclidean Upper-Tri PLI (171-D, No SPD Assumption) | Eigenvalue-Adaptive Diagonal-Loaded SPD Tangent (190-D, λ_min ≥ 0.01) | Higham Spectral-Floor SPD Tangent (190-D, λ_i ≥ 10^-4) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    cohorts_order = df_b["cohort"].unique()
    for c_name in cohorts_order:
        sub_c = df_b[df_b["cohort"] == c_name]
        n_samp = int(sub_c["n_samples"].iloc[0])
        for idx_clf, clf in enumerate(["LR", "Ridge", "RF"]):
            c_disp = f"**{c_name} (N = {n_samp})**"
            r_euc = sub_c[(sub_c["representation"] == "Euclidean_UpperTri_171D") & (sub_c["classifier"] == clf)].iloc[0]
            r_ger = sub_c[(sub_c["representation"] == "Riemannian_Gershgorin_SPD_190D") & (sub_c["classifier"] == clf)].iloc[0]
            r_hig = sub_c[(sub_c["representation"] == "Riemannian_Higham_SPD_190D") & (sub_c["classifier"] == clf)].iloc[0]

            s_euc = f"{r_euc['auc']:.3f} [{r_euc['ci_lower']:.3f}, {r_euc['ci_upper']:.3f}]"
            s_ger = f"{r_ger['auc']:.3f} [{r_ger['ci_lower']:.3f}, {r_ger['ci_upper']:.3f}]"
            s_hig = f"{r_hig['auc']:.3f} [{r_hig['ci_lower']:.3f}, {r_hig['ci_upper']:.3f}]"

            lines.append(f"| {c_disp} | `{clf}` | {s_euc} | {s_ger} | {s_hig} |")

    lines.append(
        "\n*Note: **Panel A** audits the exact eigenvalue spectrum of the symmetrized 19 × 19 connectivity matrix C = I_19 + PLI (unit diagonal, off-diagonal PLI_ij ∈ [0, 1]) across all 677 primary sessions. Unlike sample covariance matrices—which are Gram matrices and therefore positive semi-definite by construction—the Phase Lag Index is a non-linear sign statistic (|⟨sign(sin Δφ_ij)⟩|) that does not satisfy Mercer's kernel condition. Consequently, 85 / 677 (12.56%) primary sessions possess negative minimum eigenvalues (λ_min down to -7.3697 in Study 2), and a naive +10^-4I shift leaves all 85 matrices indefinite, rendering direct real-valued matrix logarithms undefined without SPD conditioning. **Panel B** therefore benchmarks three mathematically well-defined functional-connectivity representations under identical leak-free 5-fold cross-validation splits and nested hyperparameter grids (`LR`, `Ridge`, `RF`): (1) **Direct Euclidean Upper-Triangular PLI (`171-D`)**, which vectorizes the 19 × 18 / 2 = 171 unique off-diagonal phase-coupling weights directly without any SPD assumption or matrix logarithm; (2) **Eigenvalue-Adaptive Diagonal-Loaded SPD Tangent Space (`190-D`)**, which applies an exact eigenvalue-adaptive diagonal shift C_SPD = C + (max(0, -λ_min(C)) + 0.01)·I_19 to guarantee strict positive definiteness (λ_i ≥ 0.01) while preserving eigenvectors and exact pairwise eigenvalue spacing (λ_i - λ_j) prior to Log-Euclidean tangent projection S = logm(C_SPD); and (3) **Higham Spectral-Floor SPD Tangent Space (`190-D`)**, which projects C onto the Frobenius-nearest SPD matrix by clamping non-positive eigenvalues to a floor of λ_i ≥ 10^-4 before tangent projection. Across all nine primary contrasts (and nine of ten evaluated contrasts overall), both direct 171-D Euclidean and 190-D SPD tangent-space classifiers trail the best-performing node-feature models (e.g., in Study 2 Outpatient MDD, node-feature models achieve ROC-AUC up to 0.933 [Table 2], whereas 171-D Euclidean PLI achieves 0.655 and Riemannian tangent achieves 0.640–0.669; in Study 5c Combined Dementia, node models reach 0.880 vs. 0.739 for Euclidean and 0.743 for Adaptive SPD). These findings explain why graph neural networks show no topological superiority over tuned node-level tabular baselines: the underlying sensor-level phase synchronization matrices are noisy, frequently indefinite, and carry substantially less diagnostic signal than localized parameterized oscillatory power.*"
    )
    return "\n".join(lines)


def build_table_s23_meta_analysis() -> str:
    """Synthesize Markdown Table S23 (random-effects meta-analysis of frontal periodic theta direction)."""
    csv_path = REPORTS_DIR / "meta_analysis_frontal_theta_polarity.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required input for Table S23: {csv_path}")
    df = pd.read_csv(csv_path)

    lines = [
        "### Table S23. Random-Effects Meta-Analysis of Resting Frontal Periodic Theta Across Independent Cohorts and Sensitivity Specifications",
        "| Meta-Analytic Scope | Cohort Group | k | Pooled Cohen's d_RE (DL) | DL 95% CI | Variance-Floored HKSJ 95% CI | z | p (DL) | p (HKSJ) | Cochran Q (df) | τ² (τ) | I² | Between-Group Q (p) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    def scope_label(name: str) -> str:
        if name.startswith("Primary_Independent_K6"):
            return "**Primary independent set (K = 6 cohorts, N = 495)**"
        if name.startswith("Sens1"):
            return "Sensitivity 1: participant-averaged UCSD PD (15 vs. 16)"
        if name.startswith("Sens2"):
            return "Sensitivity 2: PD-ON medication states"
        if name.startswith("Sens3"):
            return "Sensitivity 3: AD-only dementia contrast"
        if name.startswith("Sens4"):
            return "Sensitivity 4: FTD-only dementia contrast"
        if name.startswith("Sens5"):
            return "Sensitivity 5: adding Study 1 sleep deprivation (d_IG, K = 7)"
        loco = {
            "UCSD": "Leave-one-cohort-out: omit Study 4a UCSD PD-OFF",
            "Dementia": "Leave-one-cohort-out: omit Study 5c combined dementia",
            "UNM": "Leave-one-cohort-out: omit Study 6a UNM PD-OFF",
            "Iowa": "Leave-one-cohort-out: omit Study 7 Iowa PD",
        }
        for key, lbl in loco.items():
            if name.startswith("LeaveOneOut") and key in name:
                return lbl
        return name

    group_map = {
        "Depressive_HypoOscillatory_Domain": "Depression cohorts (Studies 2, 3; lower theta)",
        "Neurodegenerative_Slowing_Domain": "PD and dementia cohorts (higher theta)",
        "Pathological_and_SleepPressure_Slowing_Domain": "PD, dementia, and sleep deprivation (higher theta)",
        "Overall_Pooled": "Overall pooled (opposite-sign groups; heterogeneous by construction)",
    }

    for _, r in df.iterrows():
        d_str = f"**{r['pooled_d']:+.3f}**" if float(r["p_value"]) < 0.05 else f"{r['pooled_d']:+.3f}"
        dl_ci = f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"
        hk_ci = f"[{r['hksj_ci_lower']:+.3f}, {r['hksj_ci_upper']:+.3f}]"
        q_str = f"{r['Q_stat']:.2f} ({int(r['Q_df'])})"
        tau_str = f"{r['tau2']:.3f} ({r['tau']:.3f})"
        if r["domain_group"] == "Overall_Pooled":
            qb = "—"
        elif "Depress" in r["domain_group"]:
            qb = f"{r['Q_between']:.2f} ({fmt_p_sci(float(r['p_between']))})"
        else:
            qb = "(same test)"
        lines.append(
            f"| {scope_label(r['analysis_set'])} | {group_map.get(r['domain_group'], r['domain_group'])} | {int(r['k_studies'])} | "
            f"{d_str} | {dl_ci} | {hk_ci} | {r['z_score']:+.3f} | {fmt_p_sci(float(r['p_value']))} | {fmt_p_sci(float(r['p_hksj']))} | "
            f"{q_str} | {tau_str} | {r['I2_percent']:.1f}% | {qb} |"
        )

    qb_all = df.drop_duplicates("analysis_set")[["Q_between", "p_between"]]
    prim = df[df["analysis_set"].str.startswith("Primary_Independent_K6")]
    dep = prim[prim["domain_group"] == "Depressive_HypoOscillatory_Domain"].iloc[0]
    neu = prim[prim["domain_group"] == "Neurodegenerative_Slowing_Domain"].iloc[0]

    note = (
        "\n*Note: Study-level effect sizes and standard errors come from `reports/master_results.json` (identical to Main Text Table 1 and "
        "Supplementary Table S4). To avoid shared controls and repeated sessions, the primary analysis uses K = 6 mutually exclusive "
        "between-subject cohorts (N = 495 participants): depression cohorts (k = 2: Study 2 outpatient MDD, d = -0.826 ± 0.273; Study 3 "
        "university depression screening, d = -0.676 ± 0.195) and PD/dementia cohorts (k = 4: Study 4a UCSD PD-OFF, d = +0.559 ± 0.366; "
        "Study 5c combined dementia, d = +0.903 ± 0.236; Study 6a UNM PD-OFF, d = +0.865 ± 0.296; Study 7 Iowa PD, d = +0.899 ± 0.184). "
        "DL = DerSimonian–Laird normal-theory interval. The HKSJ column is an ad-hoc variance-floored Hartung–Knapp–Sidik–Jonkman "
        "t-interval (df = k − 1) with SE = sqrt(max(1, Q/(k − 1))) · SE_DL, so that it cannot be narrower than the DL interval; unfloored "
        "HKSJ intervals (sqrt(Q/(k − 1)) · SE_DL) are [-1.622, +0.170] for the k = 2 depression group and [+0.662, +1.051] for the k = 4 PD/dementia group. "
        f"**Interpretation limits.** (i) For the depression group (k = 2), the pooled estimate is d_RE = {dep['pooled_d']:+.3f} "
        f"(DL 95% CI [{dep['ci_lower']:+.3f}, {dep['ci_upper']:+.3f}]) but the HKSJ interval [{dep['hksj_ci_lower']:+.3f}, {dep['hksj_ci_upper']:+.3f}] "
        f"includes zero (p_HKSJ = {dep['p_hksj']:.3f}); inference for this group rests on the two individually significant cohorts (p = 0.0027 and p = 4.96 × 10^-4), "
        "one of which is a symptom-screening sample in which only about half of the BDI-II ≥ 14 participants had SCID-confirmed current or past MDD. "
        f"For PD/dementia (k = 4), d_RE = {neu['pooled_d']:+.3f} (DL 95% CI [{neu['ci_lower']:+.3f}, {neu['ci_upper']:+.3f}]; "
        f"HKSJ [{neu['hksj_ci_lower']:+.3f}, {neu['hksj_ci_upper']:+.3f}], p_HKSJ = {neu['p_hksj']:.4f}). "
        "(ii) At k = 2–4, τ² = 0.000 and I² = 0.0% indicate only that observed dispersion did not exceed sampling variance and carry wide uncertainty. "
        f"(iii) The between-group Q statistic (primary Q = {dep['Q_between']:.2f}, p = {dep['p_between']:.2e}; range across all specifications "
        f"{qb_all['Q_between'].min():.2f}–{qb_all['Q_between'].max():.2f}) tests whether two groups of effect sizes with opposite signs differ; it is large "
        "almost by construction and is not an independent test of disease specificity. (iv) The two groups differ simultaneously in diagnosis, age, "
        "recording hardware, eye state, medication status, and site, none of which can be separated with this design; the result therefore describes "
        "opposite directions in these cohorts, not a disease-intrinsic polarity. The sensitivity specifications (participant-averaged UCSD, PD-ON states, "
        "AD-only or FTD-only dementia, adding Study 1 on the Morris–DeShon independent-groups scale with d_IG = +0.267 ± 0.112, and four leave-one-cohort-out deletions) "
        "leave the sign pattern unchanged; Sensitivity 5 introduces expected heterogeneity (τ² = 0.100, I² = 70.0%) between acute sleep deprivation and chronic neurodegeneration.*"
    )
    lines.append(note)
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
    for name, tbl_md in zip(["S19", "S20", "S21", "S22", "S23"], [t19, t20, t21, t22, t23]):
        assert tbl_md.strip(), f"Table {name} is empty!"
    content = "\n\n---\n\n".join([t19, t20, t21, t22, t23])
    out_file.write_text(content, encoding="utf-8")
    print(f"✅ Saved extended tables to: {out_file}")


if __name__ == "__main__":
    main()
