# A Gain-Invariant, Leak-Free EEG Benchmark: 19-Channel OMST Connectomics & Graph Neural Network Utility Across 7 Clinical Cohorts

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

An open-source, leak-free computational benchmark for evaluating **Universal 19-Channel Spatial Harmonization**, **Spectral Parameterization (`specparam`)**, **Orthogonal Minimum Spanning Trees (OMST)**, and **Graph Neural Networks vs. Classical Tabular Baselines** across seven public resting-state EEG cohorts ($N = 566$ unique participants, $677$ primary recording sessions).

---

## 🔬 Multi-Cohort Dataset Manifest ($N = 566$ participants, $677$ sessions)

The benchmark harmonizes seven publicly available resting EEG repositories into standardized 19-channel montages (standard 10–20 array):

| Cohort | Study & Accession | N (Sessions / Subj) | Clinical / Experimental Focus | Hardware Array & Sample Rate |
| :--- | :--- | :---: | :--- | :--- |
| **Study 1** | Acute Sleep Deprivation ([OpenNeuro `ds004902`](https://openneuro.org/datasets/ds004902)) | 142 / 71 | Paired crossover acute sleep debt (36h TSD vs. Normal Sleep) | 64-ch BrainAmp / actiCHamp (250 Hz) |
| **Study 2** | Outpatient Major Depression ([Figshare `4244171`](https://doi.org/10.6084/m9.figshare.4244171)) | 58 / 58 | Clinical MDD (unmedicated outpatients) vs. Healthy Controls | 19-ch Grass Telefactor (250 Hz) |
| **Study 3** | University Depression Screening ([OpenNeuro `ds003478`](https://openneuro.org/datasets/ds003478)) | 119 / 119 | Undergraduates screened with BDI-II (Depressed vs. Controls) | 64-ch BrainAmp (250 Hz) |
| **Study 4** | UCSD Parkinson's Disease ([OpenNeuro `ds002778`](https://openneuro.org/datasets/ds002778)) | 46 / 31 | Idiopathic PD under Levodopa on/off state vs. Controls | 32-ch BioSemi ActiveTwo (250 Hz) |
| **Study 5** | Clinical Dementia ([OpenNeuro `ds004504`](https://openneuro.org/datasets/ds004504)) | 88 / 88 | Probable AD (N=36), FTD (N=23), and Controls (N=29) | 19-ch Nihon Kohden (250 Hz) |
| **Study 6** | UNM Parkinson's Disease ([OpenNeuro `ds003490`](https://openneuro.org/datasets/ds003490)) | 75 / 50 | Tri-center PD: Levodopa washout (OFF/ON) vs. Controls | 64-ch BrainAmp (250 Hz) |
| **Study 7** | Iowa Parkinson's Disease ([OpenNeuro `ds004584`](https://openneuro.org/datasets/ds004584)) | 149 / 149 | Tri-center PD: Large single-site resting cohort | 64-ch BrainAmp (250 Hz) |

---

## 🛠️ Methodological Framework

1. **Multiplicative Voltage Gain Invariance**:
   - Analytically and numerically proves that periodic residual power and aperiodic slope $\chi$ derived from `specparam` are invariant under arbitrary sensor amplifier voltage scaling ($|\Delta \mathbf{X}| = 0.0000$).
   - Phase Lag Index (PLI) signed phase differences are scale-free by construction.
2. **Topological Connectome Regularization**:
   - Full 19×19 functional connectivity matrices are filtered via Orthogonal Minimum Spanning Trees (OMST), maximizing Global Cost Efficiency (GCE = $E_{\text{global}} - \text{Cost}$) without arbitrary proportional thresholding heuristics.
3. **Strict Leak-Free Validation**:
   - Stratified participant-grouped cross-validation (`StratifiedGroupKFold`) for repeated-measurement cohorts.
   - Fold-nested normative calibration ($z$-scoring against training healthy controls only, computed within inner loops).
4. **Architectural Evaluation of Graph Neural Networks**:
   - Systematic comparison of `NormativeGCN` against 12 regularized classical and deep tabular baselines (`Random Forest`, `Tuned Ridge`, `Tuned Logistic Regression`, `Linear/RBF SVM`, `GBDT`, `MLP`).
   - Graph ablations: `GCN-OMST` vs. `GCN-Random` (density-matched random topologies) vs. `GCN-Identity` (isolated nodes).
5. **Riemannian Geometry Audit**:
   - Empirical eigenvalue spectrum audit across all 677 primary sessions demonstrates that 12.56% of sensor PLI matrices violate positive semi-definiteness, requiring eigenvalue-adaptive regularization before projection to Riemannian SPD manifolds.

---

## 📦 Installation & Setup

```bash
git clone https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark.git
cd transdiagnostic-eeg-benchmark

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install numpy scipy pandas scikit-learn torch mne specparam statsmodels matplotlib seaborn pyyaml
```

---

## 🚀 Execution & Reproduction Pipeline

All scripts execute deterministically:

```bash
# 1. Extract 19-channel features and construct OMST brain graphs
python3 scripts/01_build_multicohort_graphs.py
python3 scripts/11_build_dementia_ds004504_graphs.py
python3 scripts/13_build_unm_pd_ds003490_graphs.py
python3 scripts/15_build_iowa_pd_ds004584_graphs.py

# 2. Biophysical effect size profiling & clinical scale regression
python3 scripts/02_multicohort_biophysical_profiling.py
python3 scripts/12_analyze_dementia_biophysics.py
python3 scripts/14_analyze_unm_pd_biophysics.py
python3 scripts/16_analyze_iowa_pd_biophysics.py

# 3. Model benchmarking & cross-site transfer
python3 scripts/17_run_comprehensive_7cohort_benchmark.py
python3 scripts/18_cross_site_pd_and_transdiagnostic_transfer.py

# 4. Methodological ablations (leakage, gain perturbation, Riemannian geometry, multi-seed)
python3 scripts/33_run_leakage_and_gain_ablation_benchmarks.py
python3 scripts/34_run_multiseed_and_riemannian_benchmarks.py
python3 scripts/37_run_gcn_hyperparameter_sensitivity.py

# 5. Generate high-resolution publication figures (Figures 1–4, S1–S6)
python3 scripts/22_update_all_figures_and_flowchart.py
```

---

## 📁 Repository Organization

```
transdiagnostic_eeg_benchmark/
├── config/
│   └── pipeline_config.yaml                  # Pipeline parameters and frequency band definitions
├── reports/
│   ├── master_results.json                   # Consolidated benchmark performance metrics
│   ├── Figure_1_Multicohort_Biophysical_Effect_Sizes.png/.pdf
│   ├── Figure_2_Tri_Center_Parkinsons_Generalization.png/.pdf
│   ├── Figure_3_Transdiagnostic_Transfer_and_Ablations.png/.pdf
│   ├── Figure_4_Clinical_Scale_Continuous_Prediction.png/.pdf
│   └── *.csv                                 # Master benchmark results, effect sizes, and ablations
├── scripts/                                  # Modular, numbered execution pipeline scripts
├── src/
│   ├── harmonizer_and_graph.py               # 19-ch adapter, OMST, GCE, NormativeGCN PyTorch module
│   └── multi_cohort_loader.py                # BIDS data loader and epoch standardizer
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📄 License

This repository is distributed under the open-source **MIT License**. See `LICENSE` for details.
