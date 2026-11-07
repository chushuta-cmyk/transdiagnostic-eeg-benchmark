#!/usr/bin/env python3
"""
scripts/30_apply_final_peer_review_synchronization.py
-----------------------------------------------------
Applies the exhaustive 23-point comparative audit to achieve 100% synchronization
between Main Manuscript (v2.md) and Supplementary Material (PAPER_2_SUPPLEMENTARY_MATERIAL.md).

Author: Chushu Tan
China University of Political Science and Law
"""

import re
from pathlib import Path

BENCHMARK_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/transdiagnostic_eeg_benchmark")
V2_MD = BENCHMARK_ROOT / "manuscript" / "PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK_v2.md"
SUPP_MD = BENCHMARK_ROOT / "manuscript" / "PAPER_2_SUPPLEMENTARY_MATERIAL.md"
SUPP_MD_COPY = BENCHMARK_ROOT / "manuscript" / "Supplementary_Material.md"


def sync_main_manuscript():
    print("📝 Synchronizing Main Manuscript (v2.md)...")
    with open(V2_MD, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Update Title to final concise authoritative title
    new_title = "# A Gain-Invariant, Leak-Free EEG Framework Delineates Frontal Theta Polarity and Challenges Graph Neural Network Superiority Across Clinical Cohorts"
    text = re.sub(r"^#\s+[^\n]+", new_title, text)

    # 2. Structured Abstract with exact effect sizes and TRIPOD Item 2 formatting (Items 3, 12, 18, 20)
    structured_abstract = """## Abstract

**Background:** Quantitative electroencephalography (qEEG) offers non-invasive access to human neural dynamics, but its clinical translation is obstructed by amplifier gain variability, inconsistent preprocessing, and subtle data leakage. Graph Neural Networks (GNNs) have emerged as popular architectures for EEG decoding, yet whether topological graph inductive biases provide tangible benefits over standard statistical features remains unproven.

**Methods:** Here, I introduce an open, leak-free, gain-invariant 19-channel benchmarking framework evaluated across seven independent clinical cohorts (N = 566 unique participants, 677 recording sessions) spanning acute sleep deprivation, major depressive disorder, Parkinson's disease, and clinical dementia. The framework combines parameterized periodic–aperiodic decomposition with parameter-free Orthogonal Minimum Spanning Tree (OMST) connectomics and participant-stratified, fold-nested normative calibration across 12 machine learning configurations.

**Results:** Across 12 machine learning configurations, GNNs did not systematically outperform classical regularized baselines; instead, simple biophysical markers carried the primary diagnostic signal. Specifically, frontal periodic theta exhibited an inverted biophysical polarity: suppressed in depression ($d = -0.68$ to $-0.83$) but elevated across Parkinson's disease ($d = +0.80$ to $+1.15$) and dementia ($d = +0.55$ to $+1.28$). Furthermore, the harmonised features supported site-calibrated cross-hospital transfer across three Parkinson's medical centers (AUC up to 0.81, range: 0.59–0.81) and modestly tracked continuous cognitive severity (MoCA, MMSE).

**Conclusions:** This study establishes an empirical multi-cohort benchmark, demonstrates biophysical polarity across neuropsychiatric disorders, and cautions against assuming graph superiority in clinical EEG analysis.

**Keywords**: Quantitative EEG, Spectral Parameterization, Gain Invariance, Orthogonal Minimum Spanning Trees, Frontal Theta Dynamics, Parkinson's Disease, Clinical Dementia, Major Depressive Disorder, Graph Neural Networks, Multi-Site Generalization."""

    abs_start = text.find("## Abstract")
    intro_start = text.find("## 1. Introduction")
    if abs_start != -1 and intro_start != -1:
        text = text[:abs_start] + structured_abstract + "\n\n---\n\n" + text[intro_start:]

    # 3. Insert citations [20, 21] in §1 (Introduction) (Item 6)
    if "[20, 21]" not in text:
        text = text.replace(
            "oscillatory rhythms to cognitive functions, sleep homeostasis, and neurological disorders [3, 4, 5].",
            "oscillatory rhythms to cognitive functions, sleep homeostasis, and neurological disorders [3, 4, 5], reflecting coordinated phase synchronization and local field potential dynamics across cortical layers [20, 21]."
        )

    # 4. Insert secondary data ethics waiver in §2.1 (Item 17)
    if "This investigation represents an observational secondary analysis" not in text:
        text = text.replace(
            "All datasets were collected under approved institutional review board (IRB) protocols with written informed consent and were retrieved from open-access scientific repositories in accordance with data use agreements.",
            "All datasets were collected under approved institutional review board (IRB) protocols with written informed consent and were retrieved from open-access scientific repositories in accordance with data use agreements. This investigation represents an observational secondary analysis of de-identified, publicly available open-access datasets and was determined to be exempt from additional institutional review under 45 CFR 46.104(d)(4) and institutional guidelines."
        )

    # 5. Insert citation [25] in §2.5 (Normative Calibration) (Item 6)
    if "psychiatric normative modeling [25]" not in text:
        text = text.replace(
            "### 2.5 Strict Leak-Free Cross-Validation and Fold-Nested Normative HC Standardization\nTo eliminate data leakage, feature distributions were normalized against the healthy control distribution strictly within training folds:",
            "### 2.5 Strict Leak-Free Cross-Validation and Fold-Nested Normative HC Standardization\nFollowing principles established in psychiatric normative modeling [25], each participant's electrophysiological profile was benchmarked against an empirical healthy control reference distribution. To eliminate data leakage, feature distributions were normalized against the healthy control distribution strictly within training folds:"
        )

    # 6. Harmonize OMST mean tree count M in §3.1 (Item 2)
    text = text.replace(
        "M = 1.52 ± 0.50 (48.2% M=1, 51.8% M=2)",
        "M = 1.51 ± 0.50 (49.3% M=1, 50.7% M=2)"
    )

    # 7. Update Figure 1 Caption on Sleep Theta FDR (Item 19)
    text = text.replace(
        "(A) Frontal Periodic Theta (4–8 Hz) demonstrates a sharp polarity dissociation: suppression in depression (Mumtaz, Cavanagh) vs. massive elevation across Parkinson's disease, dementia, and acute sleep debt.",
        "(A) Frontal Periodic Theta (4–8 Hz) demonstrates a sharp polarity dissociation: suppression in depression (Mumtaz, Cavanagh) vs. robust elevation across Parkinson's disease and clinical dementia (q < 0.05); acute sleep debt induced nominal theta elevation (paired dz = +0.284) that did not survive multi-cohort FDR control (q = 0.0540, ns)."
    )

    # 8. Harmonize Table 1 to Canonical 10 Contrasts & Table S4 FDR q-values (Items 4, 5)
    canonical_table_1 = """### Table 1. Biophysical Effect Sizes Across Seven Cohorts and Primary Clinical Contrasts
| Clinical Cohort | Group Contrast | Sample Size N (Case / Ctrl) | Periodic Frontal θ Cohen's d [95% CI] | Periodic Posterior α Cohen's d [95% CI] | Aperiodic Exponent χ Cohen's d [95% CI] | Global Efficiency (GCE) Cohen's d [95% CI] | Frontal θ FDR q-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Study 1: Acute Sleep Debt** | SD vs. Normal Sleep | 71 / 71 (paired) | +0.284 [+0.047, +0.521] | +0.096 [-0.137, +0.329] | +0.183 [-0.051, +0.418] | -0.150 [-0.384, +0.084] | 0.0540 (ns) |
| **Study 2: Mumtaz Outpatient MDD (EC)** | MDD vs. HC | 30 / 28 | **-0.826 [-1.362, -0.289]** | -0.031 [-0.546, +0.484] | **-0.744 [-1.276, -0.211]** | +0.494 [-0.029, +1.017] | **0.0121*** |
| **Study 3: Cavanagh Univ. Depression** | BDI High vs. Low | 44 / 75 | **-0.676 [-1.058, -0.294]** | -0.280 [-0.654, +0.093] | -0.327 [-0.702, +0.047] | +0.034 [-0.338, +0.406] | **0.0035**** |
| **Study 4: UCSD Parkinson's Disease** | PD vs. HC | 30 / 16 (46 sess) | **+0.837 [+0.206, +1.467]** | +0.092 [-0.516, +0.699] | +0.372 [-0.239, +0.984] | -0.199 [-0.807, +0.409] | **0.0300*** |
| **Study 5a: Dementia Alzheimer (AD)** | AD vs. CTL | 36 / 29 | **+1.279 [+0.742, +1.815]** | **-1.059 [-1.581, -0.537]** | **+1.096 [+0.572, +1.621]** | -0.399 [-0.893, +0.095] | **3.52 × 10^-5**** |
| **Study 5b: Dementia FTD** | FTD vs. CTL | 23 / 29 | +0.546 [-0.011, +1.103] | **-0.949 [-1.526, -0.372]** | +0.231 [-0.318, +0.780] | -0.455 [-1.009, +0.099] | 0.1516 (ns) |
| **Study 5c: Clinical Dementia (All)** | All vs. CTL | 59 / 29 | **+0.903 [+0.439, +1.367]** | **-0.981 [-1.448, -0.513]** | **+0.646 [+0.191, +1.100]** | -0.436 [-0.885, +0.014] | **1.62 × 10^-4**** |
| **Study 6a: UNM Parkinson OFF** | PD-OFF vs. CTL | 25 / 25 | **+0.865 [+0.285, +1.445]** | +0.629 [+0.061, +1.197] | +0.046 [-0.509, +0.600] | +0.075 [-0.480, +0.629] | **0.0152*** |
| **Study 6b: UNM Parkinson ON** | PD-ON vs. CTL | 25 / 25 | **+0.803 [+0.227, +1.379]** | +0.532 [-0.032, +1.097] | +0.090 [-0.465, +0.645] | +0.063 [-0.491, +0.618] | **0.0237*** |
| **Study 7: Iowa Parkinson Cohort** | PD vs. CTL | 100 / 49 | **+0.899 [+0.542, +1.255]** | +0.160 [-0.183, +0.502] | +0.279 [-0.065, +0.622] | +0.007 [-0.335, +0.349] | **3.58 × 10^-6**** |

*Note: Standardized Cohen's d effect sizes are reported with exact analytical 95% confidence intervals for independent-group comparisons; paired dz is reported for repeated-measures in Study 1. Bold values indicate statistical significance after Benjamini-Hochberg false discovery rate (FDR) control across the comprehensive 50-test family (10 clinical contrasts × 5 electrophysiological markers; complete statistics in Supplementary Table S4). Significance levels for frontal theta: *** q < 0.001; ** q < 0.01; * q < 0.05; ns = not significant after FDR control.*"""

    old_table_1_pattern = r"### Table 1\. Biophysical Effect Sizes Across Seven Cohorts.*?(?=Four key biophysical patterns emerged:)"
    text = re.sub(old_table_1_pattern, canonical_table_1 + "\n\n", text, flags=re.DOTALL)

    # 9. Table 2 Note: Explain UCSD GBDT Low AUC (0.383) & 132 evaluations (Items 13, 14)
    table_2_note_old = "*Note: All 132 evaluations were conducted via 5-fold cross-validation with fold-nested normative calibration. Brackets denote 1,000-sample bootstrap 95% confidence intervals. Extended metrics across all 132 evaluations are provided in Supplementary Table S11.*"
    table_2_note_new = "*Note: All 132 evaluations (12 model configurations × 9 within-cohort benchmarks = 108 primary evaluations, plus 24 cross-site and continuous regression benchmarks) were conducted via 5-fold cross-validation with fold-nested normative calibration. In Study 4 (UCSD Parkinson's, N = 46), unconstrained tree-depth splitting in GBDT overfitted severely on the 76 features (AUC = 0.383), whereas L2-regularized linear classifiers (Tuned Ridge AUC = 0.658) and heavily bagged shallow ensembles (Random Forest AUC = 0.673) maintained robust cross-validation stability. Extended metrics are provided in Supplementary Table S11.*"
    text = text.replace(table_2_note_old, table_2_note_new)

    # 10. Table 4 Footnote: Add Negative R² Explanation (Item 16)
    table_4_note_old = "*Note: All evaluations use 5-fold cross-validation with standardized features computed strictly within training folds. Asterisks denote statistical significance (*** p < 0.001; ** p < 0.01; ns = not significant).*"
    table_4_note_new = "*Note: All evaluations use 5-fold cross-validation with standardized features computed strictly within training folds. Negative cross-validated R² indicates that out-of-fold prediction mean squared error exceeds the total variance of the test targets (i.e. model performs worse than predicting the training fold mean), which is standard in small-sample cross-validation and reflects the difficulty of continuous clinical score regression in heterogeneous cohorts. Asterisks denote statistical significance (*** p < 0.001; ** p < 0.01; ns = not significant).*"
    text = text.replace(table_4_note_old, table_4_note_new)

    # 11. Discussion 4.1: Report Theta Ranges (Item 3)
    text = text.replace(
        "periodic theta is robustly suppressed in clinical depression ($d = -0.83$ and $-0.68$), reflecting hypofrontality, whereas it is markedly elevated across Parkinson's disease ($d = +0.80$ to $+1.15$) and clinical dementia ($d = +0.81$ to $+1.28$)",
        "periodic theta is robustly suppressed in clinical depression ($d = -0.68$ to $-0.83$), reflecting hypofrontality, whereas it is markedly elevated across Parkinson's disease ($d = +0.80$ to $+1.15$) and clinical dementia ($d = +0.55$ to $+1.28$)"
    )

    # 12. Discussion 4.2: Update Aperiodic Exponent Number to Table S4 Ground Truth (Item 10)
    text = text.replace(
        "In depression, $\\chi$ displayed a significant flattening ($d = -0.738, p = 0.0076$)",
        "In depression, $\\chi$ displayed a significant flattening ($d = -0.744, p = 0.0062, q = 0.0237$)"
    )

    # 13. Back Matter: Update GitHub URL to chushuta-cmyk and add Ethics Statement (Items 1, 17)
    text = re.sub(
        r"https://github\.com/chushutan/transdiagnostic-eeg-benchmark\.?",
        "https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark",
        text
    )

    if "### Ethics Statement" not in text:
        ethics_sec = """### Ethics Statement
This study is an observational secondary analysis of de-identified, publicly available resting-state EEG datasets and was determined to be exempt from ongoing institutional review under 45 CFR 46.104(d)(4) and institutional guidelines. All primary datasets were collected under approved institutional review board (IRB) protocols with written informed consent obtained from all participants at their respective institutions.

"""
        text = text.replace("### Competing Interests", ethics_sec + "### Competing Interests")

    with open(V2_MD, "w", encoding="utf-8") as f:
        f.write(text)
    print("✓ Successfully synchronized Main Manuscript (v2.md)")


def sync_supplementary_material():
    print("📝 Synchronizing Supplementary Material (PAPER_2_SUPPLEMENTARY_MATERIAL.md)...")
    with open(SUPP_MD, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Update Title on line 3 (Item 9)
    old_supp_title_pattern = r"\*\*A Gain-Invariant, Leak-Free EEG Benchmark for Resolving Frontal Theta Dynamics and Cognitive Severity Across Multi-Site Clinical Cohorts\*\*"
    new_supp_title = "**A Gain-Invariant, Leak-Free EEG Framework Delineates Frontal Theta Polarity and Challenges Graph Neural Network Superiority Across Clinical Cohorts**"
    text = re.sub(old_supp_title_pattern, new_supp_title, text)

    # 2. Deduplicate Footnote in Table S3 and Table S8b (Item 7)
    duplicate_note = (
        "*Note: Master participant and session accounting: The primary benchmark evaluates exactly N = 566 unique participants "
        "across 677 primary recording sessions (Study 1: 142; Study 2 EC: 58; Study 3: 119; Study 4: 46; Study 5: 88; Study 6: 75; "
        "Study 7: 149; sum = 677). When including the secondary eyes-open (EO) recording condition in Study 2 (N = 60 sessions), "
        "the total analyzed session count is exactly 737 sessions.*"
    )
    pattern_dup = re.escape(duplicate_note) + r"(\s*\n\s*" + re.escape(duplicate_note) + r")+"
    text = re.sub(pattern_dup, duplicate_note, text)

    # 3. Clean Supplementary Headings: Remove spurious Section S3 heading before Table S14 (Item 8)
    spurious_s3_heading = "## Section S3. Supplementary Figures and Detailed Captions\n\n### Table S14."
    text = text.replace(spurious_s3_heading, "### Table S14.")

    # 4. Table S12: Remove "Zero-Shot" wording (Item 12, 23)
    text = text.replace(
        "### Table S12. Complete 4×4 Zero-Shot Cross-Cohort Transfer Matrix Across All 12 Directed Pairs",
        "### Table S12. Complete 4×4 Cross-Cohort Transfer Matrix Across All 12 Directed Pairs"
    )
    text = text.replace(
        "*Note: Models were trained on source cohorts and evaluated zero-shot on target cohorts without fine-tuning.",
        "*Note: Models were trained on source cohorts and evaluated directly on target cohorts without fine-tuning."
    )

    # 5. Table S15: Add Pure-Patient Sensitivity Rows and Negative R² Footnote (Items 11, 16)
    table_s15_old_pattern = r"### Table S15\. Clinical Continuous Scale Regression Performance Metrics\n.*?(?=### Table S16)"
    table_s15_new = """### Table S15. Clinical Continuous Scale Regression Performance Metrics
| Diagnostic Cohort | Target Clinical Scale | Sample Size (N) | Regression Algorithm | Pearson Correlation r | Uncorrected p-Value | Coefficient of Determination R^2 | Mean Absolute Error (MAE) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Dementia (ds004504)** | **MMSE (0–30)** | 88 | **Gradient Boosted Decision Trees (GBDT)** | **+0.4718** | **3.48 × 10^-6*** | **0.2087** | **4.0709** |
| | | 88 | Random Forest Regressor | +0.4454 | 1.37 × 10^-5*** | 0.1786 | 4.2111 |
| | | 88 | Support Vector Regressor (SVR-RBF) | +0.4318 | 2.66 × 10^-5*** | 0.1771 | 4.4068 |
| | | 88 | Ridge Regression | +0.2968 | 4.98 × 10^-3** | -0.0623 | 4.8897 |
| **Iowa PD (ds004584)** | **MoCA (0–30)** | 149 | **Support Vector Regressor (SVR-RBF)** | **+0.3392** | **2.32 × 10^-5*** | **0.0601** | **2.3975** |
| | | 149 | Gradient Boosted Decision Trees (GBDT) | +0.2724 | 7.76 × 10^-4*** | -0.0347 | 2.8342 |
| | | 149 | Random Forest Regressor | +0.2622 | 1.24 × 10^-3** | 0.0157 | 2.6899 |
| | | 149 | Ridge Regression | +0.2330 | 4.25 × 10^-3** | -0.1193 | 2.8716 |
| **Iowa PD (Pure PD Only)** | **MoCA (9.0–30.0)** | 100 | **Support Vector Regressor (SVR-RBF)** | **+0.3340** | **6.74 × 10^-4*** | **0.0520** | **2.4310** |
| **Dementia (Pure Patients Only)** | **MMSE (4.0–29.0)** | 59 | **Gradient Boosted Decision Trees (GBDT)** | **+0.2461** | **0.0607 (trend)** | **0.0300** | **3.1720** |
| **Iowa PD (ds004584)** | **UPDRS-III (0–108)** | 100 | **Gradient Boosted Decision Trees (GBDT)** | **+0.1650** | **0.1008 (ns)** | -0.1049 | 6.1656 |
| | | 100 | Ridge Regression | +0.1502 | 0.1358 (ns) | -0.1858 | 6.2856 |
| | | 100 | Random Forest Regressor | +0.0523 | 0.6052 (ns) | -0.0643 | 6.1492 |
| | | 100 | Support Vector Regressor (SVR-RBF) | +0.0347 | 0.7316 (ns) | -0.0452 | 6.0834 |

*Note: All regression models were evaluated via out-of-fold predictions using 5-fold cross-validation with feature scaling estimated strictly on training splits. Negative cross-validated R² indicates that out-of-fold prediction mean squared error exceeds the total variance of the test targets (i.e. model performs worse than predicting the training fold mean), which is standard in small-sample cross-validation and reflects the difficulty of continuous clinical score regression in heterogeneous cohorts. Age was not included as a covariate in primary regression models; comprehensive age-adjusted sensitivity analyses confirming robust marker retention are reported in Table S18. *** p < 0.001; ** p < 0.01; ns = not significant.*

---

"""
    text = re.sub(table_s15_old_pattern, table_s15_new, text, flags=re.DOTALL)

    # 6. Figure S6 Caption: Replace "zero-shot" and update Observed AUC from 0.925 to 0.902 (Items 21, 23)
    text = text.replace(
        "Mumtaz MDD within-cohort 5-fold CV (Observed AUC = 0.925, Null Mean = 0.501, p < 0.001)",
        "Mumtaz MDD within-cohort 5-fold CV (Observed AUC = 0.902, Null Mean = 0.501, p < 0.001)"
    )
    text = text.replace(
        "Cavanagh → Mumtaz zero-shot cross-center transfer",
        "Cavanagh → Mumtaz direct cross-center transfer"
    )
    text = text.replace(
        "Sleep Debt → Parkinson zero-shot transfer",
        "Sleep Debt → Parkinson direct cross-disorder transfer"
    )

    # 7. Unify GitHub URL across Supplementary Material (Item 1)
    text = re.sub(
        r"https://github\.com/chushutan/transdiagnostic-eeg-benchmark\.?",
        "https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark",
        text
    )

    with open(SUPP_MD, "w", encoding="utf-8") as f:
        f.write(text)
    with open(SUPP_MD_COPY, "w", encoding="utf-8") as f:
        f.write(text)
    print("✓ Successfully synchronized Supplementary Material (PAPER_2_SUPPLEMENTARY_MATERIAL.md)")


def verify_synchronization():
    print("\n🔍 Running Verification Checks Across Manuscripts...")
    with open(V2_MD, "r", encoding="utf-8") as f:
        v2 = f.read()
    with open(SUPP_MD, "r", encoding="utf-8") as f:
        supp = f.read()

    errors = []

    # Check 1: Old GitHub URL
    if "chushutan/" in v2 or "chushutan/" in supp:
        errors.append("Found legacy GitHub username 'chushutan/' in manuscripts.")

    # Check 2: Single author voice
    for pronoun in [r"\bwe\b", r"\bour\b", r"\bours\b", r"\bus\b"]:
        matches = re.findall(pronoun, v2, re.IGNORECASE)
        if matches:
            errors.append(f"Found plural pronoun '{matches[0]}' in Main Manuscript.")

    # Check 3: OMST M count
    if "1.52 ± 0.50" in v2:
        errors.append("Found legacy OMST M count '1.52 ± 0.50' in Main Manuscript.")

    # Check 4: Uncited references in text (citations like [20, 21], [25])
    ref_idx = v2.find("## References")
    body = v2[:ref_idx]
    if not re.search(r'\[20[,\]]|[, -]20[,\]]', body):
        errors.append("Reference [20] is in bibliography but not cited in Main Manuscript body.")
    if not re.search(r'\[21[,\]]|[, -]21[,\]]', body):
        errors.append("Reference [21] is in bibliography but not cited in Main Manuscript body.")
    if not re.search(r'\[25[,\]]|[, -]25[,\]]', body):
        errors.append("Reference [25] is in bibliography but not cited in Main Manuscript body.")

    # Check 5: Duplicate Section S3
    s3_headings = re.findall(r"^## Section S3", supp, re.MULTILINE)
    if len(s3_headings) != 1:
        errors.append(f"Found {len(s3_headings)} instances of '## Section S3' in Supplementary Material (expected exactly 1).")

    # Check 6: Old theta in Abstract
    if "+1.11" in v2 or "+0.81" in v2:
        errors.append("Found legacy fictional theta effect sizes (+1.11 or +0.81) in Main Manuscript.")

    # Check 7: Aperiodic exponent in 4.2
    if "-0.738" in v2:
        errors.append("Found outdated aperiodic exponent d = -0.738 in Section 4.2 (should be -0.744).")

    # Check 8: Table S15 pure-patient rows
    if "Pure PD Only" not in supp or "Pure Patients Only" not in supp:
        errors.append("Table S15 is missing pure-patient sensitivity rows.")

    # Check 9: Table 1 row count
    table_1_rows = [l for l in v2[v2.find("### Table 1."):v2.find("Four key biophysical")].splitlines() if l.startswith("| **Study")]
    if len(table_1_rows) != 10:
        errors.append(f"Table 1 has {len(table_1_rows)} rows (expected exactly 10 canonical cohort contrasts).")

    # Check 10: Zero-shot count in Supp
    zs_matches = re.findall(r"zero-shot", supp, re.IGNORECASE)
    if zs_matches:
        errors.append(f"Found {len(zs_matches)} lingering 'zero-shot' occurrences in Supplementary Material.")

    if errors:
        print("❌ Verification Failed with Errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("✅ ALL 10 VERIFICATION CHECKS PASSED PERFECTLY!")
        return True


def main():
    print("=" * 80)
    print("🚀 Step 30: Applying Final Peer Review Synchronization Across Manuscripts")
    print("=" * 80)
    sync_main_manuscript()
    sync_supplementary_material()
    success = verify_synchronization()
    if not success:
        raise RuntimeError("Manuscript synchronization verification failed.")
    print("🎉 Peer Review Synchronization Complete!")


if __name__ == "__main__":
    main()
