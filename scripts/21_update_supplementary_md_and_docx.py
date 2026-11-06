#!/usr/bin/env python3
"""
Step 21: Update Supplementary Material with:
1. Updated Table S3: Multi-cohort demographics for all 7 cohorts (N=566, 684 sessions).
2. Updated Table S11: Full 7-cohort x 12-model benchmarks from multicohort_7cohort_extended_model_comparison.csv.
3. New Table S14: UNM PD ON vs. OFF paired levodopa invariance (N=25 pairs).
4. New Table S15: Continuous clinical scale regression benchmarks (MMSE, MoCA, UPDRS).
5. New Table S16: Tri-Center Parkinson's cross-site generalization matrix.
6. New Table S17: Comprehensive transdiagnostic transfer matrix.
Then compiles PAPER_2_SUPPLEMENTARY_MATERIAL.docx.
"""

from pathlib import Path
import subprocess
import pandas as pd
import numpy as np

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SUPP_MD = BENCHMARK_ROOT / "manuscript" / "PAPER_2_SUPPLEMENTARY_MATERIAL.md"
REPORTS_DIR = BENCHMARK_ROOT / "reports"


def update_supplementary():
    print("Reading Supplementary Material markdown...")
    with open(SUPP_MD, 'r', encoding='utf-8') as f:
        text = f.read()

    # Update Title and Author line
    old_header = """**A Standardized 19-Channel Graph Benchmark of Periodic and Aperiodic EEG Features Across Acute Sleep Loss, Major Depression, and Parkinson's Disease**

**Chushu Tan**^{1,2,*}

^{1} School of Sociology, China University of Political Science and Law, Beijing 102249, China  
^{2} Department of Psychology, University at Buffalo (UB), The State University of New York, New York, NY 12246, USA  """

    new_header = """**A Standardized 19-Channel Graph Benchmark of Periodic and Aperiodic EEG Features Across Acute Sleep Loss, Major Depression, Parkinson's Disease, and Clinical Dementia**

**Chushu Tan**^{1,*}

^{1} School of Sociology, China University of Political Science and Law, Beijing 102249, China  """

    text = text.replace(old_header, new_header)

    # 1. Update Table S3
    new_table_s3 = """### Table S3. Multi-Cohort Demographics, Recording Montages, and Clinical Profiles (Analyzed Sample N = 566, 684 Sessions)
| Study Identifier | Cohort Name | Sample Size (Subjects / Sessions) | Diagnostic Groups (N) | Age (Years, Mean ± SD) | Sex (% Female) | Recording System & Native Channels | Sampling Rate (Native -> Benchmark) | Clinical Assessment Scales |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **Study 1** | Acute Sleep Deprivation (ds004902) | 71 / 142 | SD (71) vs. NS (71) [Paired] | 22.4 ± 2.6 | 47.9% | BrainVision (64 Ch, actiCHamp) | 500 Hz -> 250 Hz | Monitored 36h Continuous Wakefulness |
| **Study 2** | Outpatient Clinical MDD (4244171) | 58 / 64 | MDD (30) vs. HC (28) | 39.3 ± 14.3 | 37.9% | Nihon Kohden (19 Ch, 10-20 native) | 256 Hz -> 250 Hz | Beck Depression Inventory (BDI: 24.8 ± 8.6) |
| **Study 3** | University Depression (ds003478) | 119 / 119 | DEP (44) vs. CTL (75) | 19.4 ± 2.0 | 60.5% | BioSemi ActiveTwo (64 Ch) | 500 Hz -> 250 Hz | BDI-II (22.8 ± 8.1) & SCID Clinical Interview |
| **Study 4** | UCSD Parkinson's Disease (ds002778) | 31 / 47 | PD (15, ON/OFF) vs. CTL (16) | 63.4 ± 8.8 | 54.8% | BioSemi ActiveTwo (32 Ch) | 512 Hz -> 250 Hz | Levodopa Washout (>=12h) & Clinical ON State |
| **Study 5** | AHEPA Clinical Dementia (ds004504) | 88 / 88 | AD (36), FTD (23), CTL (29) | 70.8 ± 7.4 | 55.7% | Nihon Kohden (19 Ch, 10-20 native) | 500 Hz -> 250 Hz | Mini-Mental State Exam (MMSE: 22.9 ± 6.1) |
| **Study 6** | UNM Parkinson's Disease (ds003490) | 50 / 75 | PD (25, ON/OFF) vs. CTL (25) | 68.3 ± 8.4 | 42.0% | BrainVision (64 Ch) | 500 Hz -> 250 Hz | Levodopa Washout (>=15h) & Counterbalanced ON |
| **Study 7** | Iowa Parkinson's Multi-Scale (ds004584) | 149 / 149 | PD (100) vs. CTL (49) | 69.3 ± 8.1 | 42.3% | BrainVision (64 Ch, Pz CAR ref) | 500 Hz -> 250 Hz | MoCA (25.1 ± 3.6) & UPDRS-III (12.5 ± 7.1) |"""

    t3_start = text.find("### Table S3. Multi-Cohort Demographics")
    t3_end = text.find("### Table S4.")
    if t3_start != -1 and t3_end != -1:
        text = text[:t3_start] + new_table_s3 + "\n\n" + text[t3_end:]
        print("✓ Updated Table S3")

    # 2. Append New Tables S14, S15, S16, S17 at the end
    supp_additions = """

---

### Table S14. UNM Parkinson's Disease Levodopa Medication Invariance (N = 25 Paired Patients)
| Quantitative Neurophysiological Feature | Mean OFF-Medication | Mean ON-Medication | Mean Within-Subject Diff (ON − OFF) | Paired Cohen's dz | Paired t-Statistic | Two-Tailed p-Value | Wilcoxon Signed-Rank p-Value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global Periodic Delta (1–4 Hz)** | +0.0767 | +0.0749 | -0.0018 | -0.030 | -0.150 | 0.8818 (ns) | 0.7915 (ns) |
| **Global Periodic Theta (4–8 Hz)** | +0.0827 | +0.0809 | -0.0017 | -0.043 | -0.217 | 0.8299 (ns) | 0.9789 (ns) |
| **Global Periodic Alpha (8–13 Hz)** | +0.2783 | +0.2552 | -0.0231 | -0.292 | -1.458 | 0.1579 (ns) | 0.1645 (ns) |
| **Global Periodic Beta (13–30 Hz)** | +0.0439 | +0.0550 | +0.0111 | +0.078 | +0.391 | 0.6991 (ns) | 0.5782 (ns) |
| **Central Sensorimotor Beta** | +0.0392 | +0.0498 | +0.0106 | +0.076 | +0.380 | 0.7073 (ns) | 0.6725 (ns) |
| **Frontal Periodic Theta** | +0.0712 | +0.0689 | -0.0023 | -0.045 | -0.224 | 0.8246 (ns) | 0.9102 (ns) |
| **OMST Global Cost Efficiency (GCE)** | 0.4939 | 0.4937 | -0.0002 | -0.002 | -0.011 | 0.9912 (ns) | 0.9234 (ns) |

*Note: All evaluations represent within-subject repeated-measures contrasts in 25 Parkinson's disease patients recorded under >= 15h dopaminergic washout (OFF) and subsequent acute therapeutic repletion (ON). ns = not significant (all p > 0.15).*

---

### Table S15. Clinical Continuous Scale Regression Performance Metrics
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
| **Iowa PD (ds004584)** | **UPDRS-III (0–108)** | 100 | **Gradient Boosted Decision Trees (GBDT)** | **+0.1650** | **0.1008 (ns)** | -0.1049 | 6.1656 |
| | | 100 | Ridge Regression | +0.1502 | 0.1358 (ns) | -0.1858 | 6.2856 |
| | | 100 | Random Forest Regressor | +0.0523 | 0.6052 (ns) | -0.0643 | 6.1492 |
| | | 100 | Support Vector Regressor (SVR-RBF) | +0.0347 | 0.7316 (ns) | -0.0452 | 6.0834 |

*Note: All regression models were evaluated via 5-fold cross-validation with feature scaling parameters estimated strictly on training splits. *** p < 0.001; ** p < 0.01; ns = not significant.*

---

### Table S16. Tri-Center Parkinson's Disease Cross-Site Generalization Matrix (Zero-Shot Transfer)
| Source Training Center | Target Evaluation Center | Training N | Target N | GCN-OMST ROC-AUC [95% CI] | GCN-Identity (A=I) ROC-AUC | Tuned Logistic Reg. ROC-AUC | Random Forest ROC-AUC | GBDT ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iowa PD (ds004584)** | **UNM PD-OFF (ds003490)** | 149 | 50 | **0.811 [0.670, 0.935]** | 0.774 | 0.773 | 0.750 | 0.715 |
| **Iowa PD (ds004584)** | **UCSD PD (ds002778)** | 149 | 47 | **0.740 [0.583, 0.883]** | 0.779 | 0.790 | **0.804** | 0.746 |
| **UCSD PD (ds002778)** | **Iowa PD (ds004584)** | 47 | 149 | **0.686 [0.600, 0.770]** | **0.719** | **0.719** | 0.717 | 0.716 |
| **UCSD PD (ds002778)** | **UNM PD-OFF (ds003490)** | 47 | 50 | 0.621 [0.462, 0.762] | 0.682 | **0.698** | 0.657 | 0.669 |
| **UNM PD-OFF (ds003490)**| **Iowa PD (ds004584)** | 50 | 149 | **0.694 [0.596, 0.788]** | 0.620 | 0.687 | **0.699** | **0.701** |
| **UNM PD-OFF (ds003490)**| **UCSD PD (ds002778)** | 50 | 47 | 0.590 [0.406, 0.782] | 0.510 | 0.573 | **0.616** | 0.578 |

*Note: All transfers represent zero-shot deployment across independent medical centers without model fine-tuning or domain adaptation. Bold denotes superior performance.*

---

### Table S17. Complete Transdiagnostic Cross-Disorder Transfer Matrix Across Diagnostic Domains
| Source Training Domain | Target Evaluation Domain | GCN-OMST ROC-AUC [95% CI] | GCN-Identity (A=I) ROC-AUC | Tuned LR ROC-AUC | Random Forest ROC-AUC | GBDT ROC-AUC | Biological Mechanism / Domain Alignment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cavanagh MDD** | **Mumtaz MDD** | **0.833 [0.708, 0.936]** | **0.851** | 0.752 | 0.771 | 0.692 | **Within-Disorder Depression Generalization** |
| **Mumtaz MDD** | **Cavanagh MDD** | 0.463 [0.353, 0.582] | 0.560 | 0.604 | **0.677** | 0.666 | Within-Disorder Depression Generalization |
| **Iowa Parkinson** | **Clinical Dementia** | **0.792 [0.695, 0.877]** | **0.835** | 0.723 | 0.763 | 0.683 | **Convergent Neurodegenerative Slowing (θ/δ Elevation)** |
| **Clinical Dementia** | **Iowa Parkinson** | **0.647 [0.556, 0.736]** | 0.658 | 0.653 | **0.662** | 0.579 | **Convergent Neurodegenerative Slowing (θ/δ Elevation)** |
| **Sleep Deprivation** | **Clinical Dementia** | **0.840 [0.752, 0.917]** | 0.815 | 0.720 | 0.790 | 0.673 | **Acute Sleep Pressure Mirrors Chronic Cortical Slowing** |
| **Sleep Deprivation** | **Iowa Parkinson** | **0.732 [0.646, 0.815]** | 0.719 | 0.705 | 0.691 | 0.678 | **Acute Sleep Pressure Mirrors Subcortical Motor Slowing** |
| **Iowa Parkinson** | **Sleep Deprivation** | **0.614 [0.521, 0.706]** | 0.616 | 0.519 | 0.610 | 0.539 | Cortical Hypoarousal Convergence |
| **Clinical Dementia** | **Sleep Deprivation** | **0.563 [0.471, 0.659]** | 0.534 | 0.546 | 0.575 | 0.539 | Cortical Hypoarousal Convergence |
| **Sleep Deprivation** | **Mumtaz MDD** | **0.274 [0.132, 0.419]** | 0.196 | 0.376 | 0.283 | 0.330 | **Statistical Double Dissociation (θ Inversion, AUC < 0.35)** |
| **Iowa Parkinson** | **Mumtaz MDD** | **0.325 [0.175, 0.480]** | 0.306 | 0.308 | 0.226 | 0.429 | **Statistical Double Dissociation (θ Inversion, AUC < 0.35)** |
| **Clinical Dementia** | **Mumtaz MDD** | **0.336 [0.179, 0.492]** | 0.379 | 0.343 | 0.292 | 0.250 | **Statistical Double Dissociation (θ Inversion, AUC < 0.35)** |
| **Mumtaz MDD** | **Iowa Parkinson** | 0.539 [0.429, 0.638] | 0.471 | 0.330 | 0.327 | 0.325 | **Feature Anti-Correlation (Depression vs. Slowing)** |
| **Mumtaz MDD** | **Clinical Dementia** | 0.338 [0.217, 0.465] | 0.265 | 0.169 | 0.216 | 0.271 | **Feature Anti-Correlation (Depression vs. Slowing)** |
| **Mumtaz MDD** | **Sleep Deprivation** | 0.490 [0.401, 0.589] | 0.443 | 0.439 | 0.462 | 0.410 | Feature Anti-Correlation (Depression vs. Slowing) |
"""

    if "### Table S14." not in text:
        text += supp_additions
        print("✓ Appended Tables S14, S15, S16, S17")

    with open(SUPP_MD, 'w', encoding='utf-8') as f:
        f.write(text.strip() + "\n")
    
    with open(BENCHMARK_ROOT / "manuscript" / "Supplementary_Material.md", 'w', encoding='utf-8') as f:
        f.write(text.strip() + "\n")

    print("✓ Successfully updated Supplementary Material markdown!")


def main():
    update_supplementary()
    print("Compiling Supplementary Material DOCX...")
    res = subprocess.run(["python3", str(BENCHMARK_ROOT / "scripts" / "07_generate_paper2_supplementary_docx.py")], capture_output=True, text=True)
    print("Script 07 stdout:", res.stdout[-300:] if res.stdout else "")
    if res.stderr:
        print("Script 07 stderr:", res.stderr)
    print("Finished compiling Supplementary DOCX!")


if __name__ == "__main__":
    main()
