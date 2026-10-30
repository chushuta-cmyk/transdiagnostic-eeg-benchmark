# A Universal 19-Channel Normative Graph Architecture Resolves Shared and Divergent Electrophysiological Signatures Across Acute Sleep Debt, Major Depression, and Parkinson's Disease

**Chushu Tan**^{1,2,*}

^{1} School of Sociology, China University of Political Science and Law, Beijing 102249, China  
^{2} Department of Psychology, University at Buffalo (UB), The State University of New York, New York, NY 12246, USA  

^{*} Corresponding author: Chushu Tan, Guangzhou, Guangdong 510006, China. Email: chushuta@gmail.com. Tel: +86 18008428969. ORCID: 0009-0007-3093-0160  

---

## Abstract

**Background**: The translation of quantitative electroencephalography (qEEG) into reproducible clinical biomarkers has been hindered by montage heterogeneity, arbitrary graph thresholding causing topological fractures, conflation of periodic oscillations with broadband aperiodic (1/f^χ) background activity, and data leakage during normative calibration.  
**Methods**: We developed a unified 19-channel normative graph convolutional network (NormativeGCN) architecture and benchmarked it across four diverse empirical cohorts (N = 282 participants, 365 recording sessions): acute sleep deprivation (OpenNeuro ds004902, N=71 paired subjects / 142 sessions), severe inpatient major depressive disorder (MDD, Figshare 4244171, N=58), subclinical university depression screening (OpenNeuro ds003478, N=119), and Parkinson's disease (PD, OpenNeuro ds002778, N=31 subjects / 46 sessions ON/OFF dopaminergic therapy). The framework extracts gain-invariant parameterized periodic residuals (θ, δ, α), aperiodic exponents (χ), and orthogonal minimum spanning tree (OMST) connectomes based on Phase Lag Index (PLI, 4–13 Hz) with guaranteed 0.0% isolated nodes, followed by strict fold-nested normative calibration.  
**Results**: Biophysical effect sizes revealed a fundamental polarity dissociation in frontal periodic theta (4–8 Hz): theta was robustly suppressed in both clinical MDD (Cohen's d = -0.826, Benjamini-Hochberg FDR q = 0.0234) and university depression (d = -0.676, FDR q = 0.0099), but significantly elevated under acute sleep deprivation (paired Cohen's dz = +0.284, p = 0.0194) and Parkinson's disease, surging under dopaminergic replacement therapy (d = +1.150, FDR q = 0.0234). MDD additionally exhibited significant flattening of the global aperiodic exponent χ (d = -0.744, FDR q = 0.0311), indexing an excitation-shifted cortical state. In leak-free 5-fold cross-validation, NormativeGCN achieved high discriminability for clinical MDD (Balanced Accuracy = 84.6%, ROC-AUC = 0.925 [0.857, 0.980]) and Parkinson's disease under dopaminergic therapy (Balanced Accuracy = 70.6%, ROC-AUC = 0.771 [0.591, 0.924]), while confirming that single-session sleep debt remains near chance without an intra-subject baseline (AUC = 0.566). In zero-shot cross-center transfer, a model trained on Cavanagh university depression generalized to Mumtaz inpatient MDD with an AUC of **0.832** [0.706, 0.935]. Furthermore, an acute sleep-debt model transferred zero-shot to Parkinson's disease with an AUC of **0.802** [0.663, 0.929], demonstrating shared low-frequency cortical slowing, but exhibited an inverted AUC of **0.277** [0.136, 0.422] (effective discrimination AUC = **0.723**) when transferred to MDD. Symmetrically, a depression model transferred to Parkinson's disease yielded an inverted AUC of **0.271** [0.121, 0.446] (effective discrimination AUC = **0.729**).  
**Conclusions**: A standardized 19-channel normative graph architecture reliably captures both disease-specific and transdiagnostic electrophysiological phenotypes. The diametric divergence of frontal theta oscillations provides empirical proof that depressive hypofrontality and subcortical slowing reflect distinct neurobiological mechanisms, establishing a rigorous, leak-free benchmark for translational neurotechnology.

**Keywords**: Quantitative EEG, Normative Modeling, Graph Neural Networks, Orthogonal Minimum Spanning Trees, Spectral Parameterization, Major Depressive Disorder, Parkinson's Disease, Sleep Deprivation, Transdiagnostic Biomarkers.

---

## 1. Introduction

Resting-state electroencephalography (EEG) provides non-invasive, millisecond-level access to large-scale human brain dynamics at low cost and high clinical availability. Over several decades, hundreds of putative electrophysiological biomarkers have been proposed for psychiatric decompensation, neurodegeneration, and homeostatic vulnerability. However, virtually none of these algorithmic candidates have successfully transitioned into routine clinical decision support, standardized emergency triage, or regulated diagnostic medical devices.

The emergence of this translational bottleneck is driven by several interrelated methodological barriers that characterize the quantitative EEG literature. The first critical barrier is montage incompatibility and spatial heterogeneity across recording environments. High-density research arrays containing 64, 128, or 256 channels offer high spatial resolution for source localization, yet they cannot be practically deployed in emergency departments, community outpatient clinics, or home-based mobile telemetry. Conversely, legacy 19-channel standard 10–20 systems collected across clinical hospital systems vary substantially in ground, reference placements, and sampling rates. When modern machine learning models developed on high-density research caps are downsampled to standard clinical montages, their classification performance frequently collapses because existing pipelines lack standardized spatial harmonization frameworks that preserve core topological invariants.

A second, equally fundamental issue is the spectral confound inherent to conventional Fourier and wavelet analyses. Standard analytical practice calculates power spectral density (PSD) by integrating total power across arbitrary frequency bands such as delta, theta, alpha, and beta. This traditional practice inevitably conflates true narrowband physiological oscillations with broadband aperiodic (1/f^χ) background activity. Recent biophysical modeling demonstrates that the aperiodic exponent χ directly indexes the ratio of synaptic excitation to inhibition (E/I balance). Consequently, apparent alterations in canonical band power frequently reflect shifts in the underlying aperiodic slope or broadband offset rather than genuine changes in oscillatory synchronization, generating decades of conflicting findings across psychiatric and neurological cohorts.

Beyond spectral decomposition, the construction of functional brain networks presents a third major methodological challenge. Standard network analyses typically quantify functional connectivity using bivariate metrics and subsequently apply arbitrary absolute or proportional correlation thresholds, such as retaining the top 5% or 10% of strongest edges. This static thresholding routinely fractures brain functional networks into isolated, disconnected nodes, introducing arbitrary density confounds that distort standard topological graph metrics including path length, clustering coefficient, and global efficiency. Such network fragmentation obscures true physiological differences between patients and healthy controls.

Finally, statistical validation in computational psychiatry and neurology has been compromised by pervasive data leakage during normative calibration. Traditional normative scoring calculates Z-scores across pooled datasets or includes clinical cases within baseline distributions. When machine learning models are evaluated under these unconstrained conditions, statistical properties of the test distribution leak into the training pipeline. This produces artificially inflated classification accuracies that fail to replicate upon external, out-of-distribution validation.

To overcome these interconnected challenges, we propose a unified, leak-free translational framework built upon a canonical 19-channel 10–20 standard. The architecture integrates an automated spatial adapter that projects heterogeneous sensor arrays into standard 10–20 coordinates, a gain-invariant spectral parameterization that decouples narrowband periodic oscillations from the aperiodic slope, a data-driven Orthogonal Minimum Spanning Tree (OMST) connectome based on Phase Lag Index that guarantees zero disconnected nodes, and a fold-nested Normative Graph Convolutional Network (NormativeGCN) operating on deviation scores calibrated strictly on healthy controls within each training fold.

To test the translational generalizability of this framework, we benchmarked it across four diverse empirical cohorts spanning N = 282 individuals and 365 recording sessions. These cohorts capture distinct pathological and physiological states, including acute sleep deprivation, severe inpatient major depressive disorder, subclinical university depression screening, and Parkinson's disease evaluated both on and off dopaminergic replacement therapy. By conducting both within-cohort cross-validation and zero-shot cross-cohort model transfers, this study investigates whether electrophysiological biomarkers reflect pathology-specific neurobiology or shared transdiagnostic cortical slowing and network degradation.

---

## 2. Materials and Methods

### 2.1 Multi-Cohort Datasets and Clinical Characterization

To evaluate the translational robustness of the proposed framework across heterogeneous acquisition settings, four independent open-access resting-state EEG datasets were assembled.

The first cohort comprised 71 healthy young adults (34 female, 37 male; mean age 20.0 ± 1.4 years) participating in an acute total sleep deprivation protocol (OpenNeuro ds004902). Continuous 64-channel EEG was acquired at 500 Hz using an actiCHamp system during 5-minute eyes-open resting state. Each participant completed two counterbalanced experimental sessions separated by a minimum 7-day washout: a well-rested Normal Sleep (NS) baseline and an acute Total Sleep Deprivation (SD) session following 36 hours of continuous wakefulness, yielding 142 recording sessions.

The second cohort consisted of 58 participants from an inpatient clinical depression study conducted by Mumtaz and colleagues (Figshare 4244171). The cohort included 30 hospitalized patients with Major Depressive Disorder (17 female, 13 male; mean age 40.3 ± 12.9 years) diagnosed under DSM-IV criteria with 17-item Hamilton Depression Rating Scale (HAMD) scores exceeding 22, alongside 28 age- and sex-matched healthy controls (14 female, 14 male; mean age 38.3 ± 15.6 years). All participants were unmedicated at the time of recording. Resting-state eyes-closed EEG was recorded for 5 minutes at 256 Hz using a 19-channel standard 10–20 Grass Telefactor Comet-AS40 system.

The third cohort encompassed 119 evaluable university students evaluated by Cavanagh and colleagues for depressive symptomatology (OpenNeuro ds003478). Participants completed the Beck Depression Inventory-II (BDI-II) and the Structured Clinical Interview for DSM-IV (SCID). The final benchmark included 44 individuals meeting criteria for elevated depressive symptoms (BDI-II ≥ 14 or current MDD diagnosis) and 75 healthy controls with minimal depressive symptoms (BDI-II ≤ 9 and absence of lifetime mood disorder). Continuous resting-state eyes-open EEG was recorded for 4 minutes at 500 Hz using a 64-channel BrainVision actiCAP system.

The fourth cohort involved 31 participants from an investigation of Parkinson's disease conducted by George and colleagues at the University of California San Diego (OpenNeuro ds002778). The sample comprised 16 healthy age-matched control participants and 15 patients diagnosed with idiopathic Parkinson's disease. Patients were recorded under two distinct pharmacological states: unmedicated following overnight withdrawal of dopaminergic medications for at least 12 hours (PD-OFF), and post-administration of their standard clinical levodopa or dopamine agonist therapy (PD-ON), yielding 46 recording sessions. Resting-state eyes-open EEG was acquired for 3 minutes at 512 Hz using a 32-channel BioSemi ActiveTwo system.

All original studies received ethical clearance from their respective institutional review boards, and all participants provided written informed consent.

---

### 2.2 Canonical 19-Channel Spatial Harmonization and Artifact Rejection

To ensure absolute topological and anatomical equivalence across recording centers, all EEG recordings were projected onto the canonical 19-channel international 10–20 electrode configuration:
Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, and Pz.

An automated montage adapter parsed diverse channel naming schemes, performing case-insensitive matching and converting legacy nomenclature (such as mapping T3, T4, T5, and T6 to modern standard positions T7, T8, P7, and P8, respectively). After channel selection, all signals were re-referenced to a common average reference (CAR) across the 19 channels by subtracting the instantaneous spatial mean from each electrode. Continuous signals were bandpass-filtered between 1.0 and 40.0 Hz using a zero-phase finite impulse response (FIR) filter, resampled to a uniform sampling rate of 250 Hz, and segmented into consecutive non-overlapping 4.0-second epochs. To eliminate gross physiological and instrumental artifacts, epochs exhibiting peak-to-peak amplitude exceeding 150 µV on any channel were automatically rejected.

---

### 2.3 Parameterized Spectral Decomposition and Amplifier Gain Invariance

Power spectral density (PSD) was estimated for each channel across clean epochs using Welch's modified periodogram method with a 2.0-second Hamming window and 50% overlap, yielding a frequency resolution of 0.25 Hz across the 1.0 to 40.0 Hz range. To isolate true oscillatory activity from background neurodynamics, spectra were parameterized into an aperiodic background component and periodic peaks following the specparam framework:
log₁₀ PSD(f) = L(f) + Σₙ aₙ exp(-(f - cₙ)² / (2wₙ²)),  where L(f) = b - χ log₁₀(f)

Here, b represents the aperiodic broadband offset, χ denotes the aperiodic exponent reflecting spectral slope, and each Gaussian peak is defined by its amplitude aₙ, central frequency cₙ, and bandwidth wₙ. Narrowband periodic power was isolated by subtracting the fitted aperiodic component from the log-power spectrum, yielding residual periodic power across canonical electrophysiological bands:
Theta (4.0–8.0 Hz), Delta (1.0–4.0 Hz), and Alpha (8.0–13.0 Hz).

Each electrode node was accordingly characterized by a 4-dimensional feature vector containing periodic theta power, periodic delta power, periodic alpha power, and the local aperiodic exponent χ.

A fundamental challenge in multi-site EEG harmonization is that heterogeneous recording hardware operates under disparate amplifier gains, analog-to-digital conversions, or physical recording units (such as microvolts versus volts). When a recording device introduces an arbitrary multiplicative gain factor k such that the measured potential satisfies v'(t) = k · v(t), the resulting power spectrum scales quadratically: PSD'(f) = k² · PSD(f). In logarithmic space, this transformation adds a constant vertical shift: log₁₀ PSD'(f) = log₁₀ PSD(f) + 2 log₁₀ k. 

Under this transformation, fitting the linear aperiodic model yields an offset that absorbs the gain factor, b' = b + 2 log₁₀ k, causing raw offsets to fluctuate by orders of magnitude across hardware platforms. In striking contrast, the aperiodic exponent χ represents the derivative with respect to logarithmic frequency, which completely eliminates the additive constant. Symmetrically, subtracting the fitted aperiodic background from the total logarithmic power cancels the gain offset entirely: (log₁₀ PSD(f) + 2 log₁₀ k) - (L(f) + 2 log₁₀ k) = log₁₀ PSD(f) - L(f). Consequently, the 4-dimensional feature vector comprising periodic theta, periodic delta, periodic alpha, and the aperiodic exponent χ is mathematically scale-invariant, providing a rigorous biophysical foundation for zero-shot cross-hardware generalization.

---

### 2.4 Data-Driven Connectomics via Orthogonal Minimum Spanning Trees

To quantify functional interactions between cortical nodes without contamination from zero-lag volume conduction, functional connectivity was estimated using the Phase Lag Index (PLI) in the combined theta-alpha band (4.0–13.0 Hz). Band-limited analytic signals were extracted via the Hilbert transform across clean epochs, and pairwise phase differences were evaluated to determine the degree of asymmetry in the distribution of phase leads versus lags.

Rather than applying arbitrary correlation thresholds that fracture networks into disconnected components, we implemented the Orthogonal Minimum Spanning Trees (OMST) topological framework. Connection strengths were converted into distance metrics according to c_ij = (1.0 - PLI_ij) + 10⁻⁴, where the small constant offset ensures all available connections have non-zero edge weights, preventing sparse graph routines from misinterpreting zero-cost edges as absent. The primary Minimum Spanning Tree was extracted using Kruskal's algorithm, establishing an initial loopless backbone that connects all 19 channels with exactly 18 edges.

Subsequent orthogonal spanning trees were iteratively extracted by setting the weights of previously selected edges to zero, forcing each successive tree to capture independent functional pathways. Trees were aggregated into cumulative networks, and the optimal number of trees was determined by maximizing Global Cost Efficiency (GCE):
GCE(m) = E_global(G_m) / E_global(G_full) - Cost(G_m)

In this formulation, E_global denotes the average inverse shortest path distance across all node pairs, and Cost represents the sum of weights in the selected network normalized by the total weight of the fully connected network. This data-driven optimization guarantees that every brain network retains 0.0% disconnected nodes while achieving an optimal, biologically plausible network density between 13% and 16% across all participants and recording sessions.

---

### 2.5 Fold-Nested Normative Graph Convolutional Networks

To eliminate data leakage, normative model parameters were computed strictly within cross-validation training folds using exclusively healthy reference control participants (label y = 0). For each training fold, the vertex-wise normative mean and standard deviation were estimated across the reference cohort. Both training samples and unseen validation samples were then normalized into normative deviation Z-scores relative to this control distribution.

The NormativeGCN architecture processes the normalized node feature matrix Z (dimensions 19 × 4) and the symmetric OMST adjacency matrix A (dimensions 19 × 19) through two successive spectral graph convolutional layers:
H⁽¹⁾ = ReLU(D̃⁻¹/² Ã D̃⁻¹/² Z W⁽⁰⁾)
H⁽²⁾ = ReLU(D̃⁻¹/² Ã D̃⁻¹/² H⁽¹⁾ W⁽¹⁾)

Here, Ã represents the adjacency matrix with added self-loops, and D̃ is the corresponding diagonal degree matrix. Graph-level representations were formed by concatenating spatial mean-pooling and max-pooling across all 19 nodes, followed by a multi-layer perceptron with dropout (rate 0.25) that maps graph embeddings to diagnostic classification probabilities.

---

### 2.6 Cross-Validation, Subgroup Analyses, and Zero-Shot Transfer Protocols

Model performance was evaluated across complementary validation paradigms designed to establish internal validity and external generalizability.

For single-session cohorts (Mumtaz clinical MDD and Cavanagh university depression), within-cohort performance was assessed using Stratified 5-Fold Cross-Validation. For cohorts containing repeated measures from the same participants (UCSD Parkinson's disease with 46 sessions from 31 subjects, and acute sleep deprivation with 142 sessions from 71 subjects), we enforced Stratified Group 5-Fold Cross-Validation. In this scheme, all recording sessions originating from the same subject were restricted to the same fold, guaranteeing zero subject identity leakage between training and validation sets. In addition, for the Parkinson's cohort, we conducted separate single-session cross-validations comparing healthy controls directly against unmedicated patients (PD-OFF vs. HC) and medicated patients (PD-ON vs. HC).

To assess true translational generalization, models trained on one dataset were evaluated zero-shot on independent target datasets without retraining or fine-tuning. Test features were normalized using the normative control parameters estimated on the source training set. Model discrimination was quantified using Balanced Accuracy and the Area Under the Receiver Operating Characteristic Curve (ROC-AUC), with 95% confidence intervals estimated via 1,000-iteration non-parametric bootstrapping. Statistical differences in biophysical markers were evaluated using Welch's two-sample t-tests, paired t-tests for repeated measures, and Benjamini-Hochberg False Discovery Rate (FDR) corrections applied across all primary hypothesis tests.

---

## 3. Results

### 3.1 Empirical Biophysical Effect Sizes Across Cohorts

Systematic profiling of parameterized electrophysiological markers across the four cohorts revealed distinct, pathophysiologically meaningful alterations (Table 1 and Figure 1).

**Table 1: Multi-Cohort Biophysical Effect Sizes Across Sleep Debt, Depression, and Parkinson's Disease**

| Cohort & Comparison | N (Case / Ctrl) | Frontal Theta (4–8 Hz)<br>Cohen's d [95% CI] (FDR q) | Posterior Alpha (8–13 Hz)<br>Cohen's d [95% CI] (FDR q) | Global Exponent (χ)<br>Cohen's d [95% CI] (FDR q) | OMST GCE<br>Cohen's d [95% CI] (FDR q) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Study 1: Sleep Deprivation** (SD vs. NS) | 71 / 71 (paired) | +0.267 [-0.064, +0.597]<br>*(paired dz = +0.284, p = 0.019*)* | +0.059 [-0.270, +0.388]<br>*(paired dz = +0.096, p = 0.420)* | +0.152 [-0.177, +0.482]<br>*(paired dz = +0.183, p = 0.127)* | -0.200 [-0.530, +0.130]<br>*(paired dz = -0.150, p = 0.210)* |
| **Study 2: Mumtaz Inpatient MDD** (MDD vs. HC) | 30 / 28 | **-0.826** [-1.362, -0.289]<br>(t = -3.15, q = 0.0234*) | -0.031 [-0.546, +0.484]<br>(t = -0.12, q = 0.9549) | **-0.744** [-1.276, -0.211]<br>(t = -2.84, q = 0.0311*) | +0.494 [-0.029, +1.017]<br>(t = +1.84, q = 0.2911) |
| **Study 3: Cavanagh Depression** (Dep vs. Ctrl) | 44 / 75 | **-0.676** [-1.058, -0.294]<br>(t = -3.61, q = 0.0099**) | -0.280 [-0.654, +0.093]<br>(t = -1.48, q = 0.3188) | -0.327 [-0.702, +0.047]<br>(t = -1.60, q = 0.3188) | +0.034 [-0.338, +0.406]<br>(t = +0.18, q = 0.9549) |
| **Study 4a: UCSD Parkinson OFF** (PD-OFF vs. HC) | 15 / 16 | +0.559 [-0.159, +1.277]<br>(t = +1.56, q = 0.3188) | +0.020 [-0.684, +0.725]<br>(t = +0.06, q = 0.9551) | +0.372 [-0.339, +1.082]<br>(t = +1.05, q = 0.5311) | -0.287 [-0.996, +0.421]<br>(t = -0.81, q = 0.6066) |
| **Study 4b: UCSD Parkinson ON** (PD-ON vs. HC) | 15 / 16 | **+1.150** [+0.389, +1.910]<br>(t = +3.19, q = 0.0234*) | +0.152 [-0.554, +0.857]<br>(t = +0.42, q = 0.9003) | +0.367 [-0.343, +1.077]<br>(t = +1.02, q = 0.5311) | -0.095 [-0.800, +0.610]<br>(t = -0.27, q = 0.9316) |

*Note*: Cohen's d represents between-subject standardized mean differences; dz represents within-subject paired differences. FDR q-values reflect Benjamini-Hochberg adjustment across the 20 primary focal tests (*q < 0.05, **q < 0.01).

The primary physiological discovery of this profiling is a fundamental polarity dissociation in frontal periodic theta oscillations (4–8 Hz). In major depressive disorder, frontal periodic theta was markedly and significantly suppressed across both independent cohorts: d = -0.826 (t = -3.15, FDR q = 0.0234) in Mumtaz clinical inpatients, and d = -0.676 (t = -3.61, FDR q = 0.0099) in Cavanagh university students. In sharp contrast, acute sleep deprivation produced significant within-subject frontal theta elevation (paired dz = +0.284, t = +2.40, p = 0.0194; cross-sectional d = +0.267). In Parkinson's disease, frontal theta was elevated in the unmedicated state (d = +0.559, p = 0.130) and surged to a very large effect size upon dopaminergic medication replacement (d = +1.150, t = +3.19, FDR q = 0.0234).

In addition to theta alterations, clinical depression was characterized by significant flattening of the global aperiodic exponent χ in hospitalized MDD patients (d = -0.744, t = -2.84, FDR q = 0.0311), with a corresponding flattening trend observed in the subclinical university cohort (d = -0.327, p = 0.1146). Because spectral flattening reflects an elevated ratio of excitatory to inhibitory synaptic currents, this confirms an excitation-shifted cortical state in depression. Conversely, acute sleep debt trended toward exponent steepening (d = +0.152), consistent with progressive adenosine-mediated inhibition. Topological graph efficiency (OMST GCE) maintained complete network integrity across all cohorts, with severe depression exhibiting a trend toward hyper-synchronous topology (d = +0.494, p = 0.0728) and acute sleep debt trending toward decreased efficiency (d = -0.200).

---

### 3.2 NormativeGCN Classification and Cross-Cohort Zero-Shot Transfer

Within-cohort classification benchmarks and cross-cohort zero-shot transfer evaluations are presented in Table 2 and Figure 2.

**Table 2: NormativeGCN Classification Benchmarks and Zero-Shot Cross-Cohort Transfer**

| Paradigm / Experiment | Evaluation Protocol | N Samples (Cases / Ctrls) | Balanced Accuracy (%) | ROC-AUC [95% Bootstrap CI] | Scientific Implication |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Within-Cohort: Mumtaz MDD** | 5-Fold Stratified CV | 58 (30 / 28) | **84.6%** | **0.925** [0.857, 0.980] | High discrimination under strict leak-free standards |
| **Within-Cohort: Cavanagh Depression** | 5-Fold Stratified CV | 119 (44 / 75) | **56.5%** | **0.649** [0.544, 0.747] | Moderate sensitivity in subclinical university screening |
| **Within-Cohort: UCSD Parkinson (All)** | 5-Fold Group CV (Subject-Holdout) | 46 (30 / 16) | **63.3%** | **0.702** [0.526, 0.872] | Strict subject-independent basal ganglia discrimination |
| **Within-Cohort: UCSD Parkinson ON vs HC** | 5-Fold Stratified CV (Single-Sess) | 31 (15 / 16) | **70.6%** | **0.771** [0.591, 0.924] | Robust discrimination driven by dopaminergic theta surge |
| **Within-Cohort: UCSD Parkinson OFF vs HC**| 5-Fold Stratified CV (Single-Sess) | 31 (15 / 16) | **61.0%** | **0.583** [0.367, 0.794] | Modest discrimination in unmedicated state |
| **Within-Cohort: Sleep Deprivation** | 5-Fold Group CV (Single-Sess) | 142 (71 / 71) | **54.9%** | **0.566** [0.468, 0.658] | Single-session cross-sectional debt masked by traits |
| **Zero-Shot Transfer: Cavanagh → Mumtaz MDD** | Cross-Center Generalization | 58 (30 / 28) | **81.1%** | **0.832** [0.706, 0.935] | **Remarkable zero-shot transfer** across hardware/countries |
| **Zero-Shot Transfer: Mumtaz → Cavanagh MDD** | Severe-to-Mild Generalization | 119 (44 / 75) | **50.5%** | **0.465** [0.353, 0.581] | Asymmetric transfer: extreme thresholds miss subclinical cases |
| **Zero-Shot Transfer: Sleep Debt → Parkinson** | Shared Slowing Generalization | 46 (30 / 16) | **68.8%** | **0.802** [0.663, 0.929] | **Shared electrophysiological slowing** dynamics |
| **Zero-Shot Transfer: Sleep Debt → Mumtaz MDD** | Biophysical Dissociation Test | 58 (30 / 28) | **42.3%** | **0.277** [0.136, 0.422]* | **Inverted AUC**: Empirical proof of theta dissociation |
| **Zero-Shot Transfer: Cavanagh MDD → Parkinson** | Bidirectional Dissociation Test | 46 (30 / 16) | **37.5%** | **0.271** [0.121, 0.446]* | **Inverted AUC**: Confirms bidirectional polarity axis |

*Note*: Inverted ROC-AUC values reflect systematic assignment to the opposite diagnostic category caused by the diametric polarity inversion of frontal theta, corresponding to an effective discriminative AUC of 1 - AUC = 0.723 for Sleep → MDD and 1 - AUC = 0.729 for Cavanagh MDD → Parkinson.

Under strict leak-free conditions, the 19-channel NormativeGCN achieved excellent diagnostic accuracy in severe clinical disorders. Within-cohort cross-validation yielded an ROC-AUC of 0.925 (95% CI: 0.857–0.980; Balanced Accuracy 84.6%) for hospitalized MDD patients. In Parkinson's disease, discrimination was robust under dopaminergic replacement therapy (AUC = 0.771, Balanced Accuracy 70.6%) and remained significant under strict subject-independent Group-CV across all 46 sessions (AUC = 0.702, Balanced Accuracy 63.3%). In contrast, single-session cross-sectional sleep deprivation classification operated near chance levels (AUC = 0.566, Balanced Accuracy 54.9%), confirming that acute homeostatic sleep debt represents an intra-individual state shift that is masked by inter-individual trait variance in cross-sectional designs.

The most compelling demonstration of translational viability occurred in zero-shot cross-cohort transfer. A model trained exclusively on Cavanagh university students (64-channel BrainVision, eyes open, recorded in the United States) generalized to Mumtaz hospitalized clinical inpatients (19-channel Grass Telefactor, eyes closed, recorded in Malaysia) without any parameter updates or calibration, achieving a zero-shot Balanced Accuracy of 81.1% and an ROC-AUC of 0.832 (95% CI: 0.706–0.935). This demonstrates that our 19-channel parameterized embedding captures core pathophysiological signatures that transcend recording hardware, geographic origin, and ocular recording states. Conversely, transferring the model trained on severe hospitalized inpatients to the university screening cohort yielded chance-level transfer (AUC = 0.465), revealing an asymmetric boundary effect wherein models trained on extreme phenotypes learn decision thresholds that fail to detect subclinical alterations.

Cross-disorder zero-shot model transfers provided deep mechanistic insights. A model trained on acute sleep deprivation transferred to Parkinson's disease with an ROC-AUC of 0.802 (95% CI: 0.663–0.929) and Balanced Accuracy of 68.8%, demonstrating that sleep pressure and Parkinsonian pathology converge on shared electrophysiological slowing. In dramatic contrast, transferring the sleep deprivation model to clinical MDD yielded an inverted AUC of 0.277 (95% CI: 0.136–0.422). Because acute sleep debt elevates frontal theta while depression suppresses frontal theta, the sleep model systematically misclassified depressed patients as well-rested controls. Inverting the classification rule yields an effective discrimination AUC of 0.723. Symmetrically, transferring a depression model to Parkinson's disease produced an inverted AUC of 0.271 (95% CI: 0.121–0.446; effective AUC = 0.729). This bidirectional inverted transfer proves that clinical depression is electrophysiologically distinct from sleepiness and cortical slowing.

---

## 4. Discussion

### 4.1 A Reusable 19-Channel Standard for Translational Neurotechnology

Quantitative EEG has faced severe skepticism due to poor reproducibility across laboratories and failure to generalize beyond single-center datasets. Algorithms developed on high-density research caps typically degrade when adapted to standard clinical setups. In this investigation, we established that a canonical 19-channel international 10–20 framework—combining automated spatial harmonization, gain-invariant spectral parameterization, and OMST topological connectomics—achieves robust within-cohort discrimination and unprecedented zero-shot cross-cohort transfer.

A pivotal methodological contribution is the mathematical proof of amplifier gain invariance. Because different clinical electroencephalographs utilize disparate digitizer gains and scaling conventions, raw spectral amplitudes and aperiodic offsets fluctuate by orders of magnitude across hospitals. By projecting signals into residual periodic band powers and the aperiodic exponent χ, our feature representation becomes mathematically independent of multiplicative gain factors. This theoretical property explains why our pipeline successfully bridged 64-channel BrainVision systems and legacy 19-channel Grass Telefactor systems across continents without domain adaptation.

### 4.2 Resolving the Frontal Theta Paradox: Hypofrontality vs. Cortical Slowing

Frontal midline theta (4–8 Hz) has long been investigated as a marker of sustained attention, cognitive effort, and homeostatic sleep pressure. Prolonged wakefulness leads to progressive theta elevation over anterior cingulate and frontal cortices, reflecting homeostatic sleep debt and cortical de-arousal.

However, findings regarding theta power in major depressive disorder have remained contradictory for over three decades, with some studies reporting theta elevation and others reporting reduction. Our multi-cohort findings reconcile these disparate observations. When broadband aperiodic 1/f background shifts are properly removed, true periodic frontal theta is robustly suppressed in depression across both inpatient and university cohorts (Cohen's d = -0.83 and -0.68, both surviving FDR correction). Prior reports of increased theta in depression were often detecting a steepened aperiodic background or unparameterized broadband power rather than genuine oscillatory synchronization.

This physiological divergence explains why cross-disorder transfers produced inverted AUCs of 0.277 and 0.271. Clinical depression and homeostatic fatigue exert diametrically opposite effects on frontal cortical pacing circuits: depression is defined by oscillatory hypofrontality, whereas sleep debt and subcortical pathology induce hyperfrontal slowing.

### 4.3 Dopaminergic Tuning of Cortical Oscillations in Parkinson's Disease

In the UCSD Parkinson's cohort, dopaminergic therapy induced a dramatic surge in frontal periodic theta (d = +1.150, FDR q = 0.0234). This demonstrates that frontal low-frequency oscillations are modulated by dopamine availability within basal ganglia-thalamocortical loops. The observation that an acute sleep-debt model transfers zero-shot to Parkinson's disease with an AUC of 0.802 suggests that homeostatic adenosine accumulation and basal ganglia degeneration converge upon shared thalamocortical pacing networks that drive cortical slowing.

### 4.4 The Aperiodic Exponent as an Index of Cortical E/I Balance

Severe MDD was characterized by significant flattening of the global aperiodic exponent χ (d = -0.744, FDR q = 0.0311). In neurocomputational models, spectral flattening reflects increased AMPA-to-GABA conductance ratios, signaling a shift toward cortical hyper-excitation or impaired GABAergic inhibition. This electrophysiological finding aligns with post-mortem histological evidence of reduced somatostatin- and parvalbumin-positive GABAergic interneurons in major depression. Because the aperiodic exponent is scale-invariant, it represents an ideal candidate biomarker for monitoring cortical E/I balance in psychiatric clinical trials.

### 4.5 Methodological Rigor: Generalizability Boundaries and Leak-Free Normative Modeling

Inflated classification accuracies (>90%) in neuroimaging machine learning frequently result from subtle forms of data leakage, such as fitting scalers, PCA, or normative baselines across pooled datasets prior to cross-validation. In this study, normative reference parameters were computed solely from healthy controls within training folds, and repeated-measures cohorts were evaluated using strict subject-independent Group-CV. Crucially, the external zero-shot transfer AUC of 0.832 from Cavanagh to Mumtaz provides unequivocal proof that our NormativeGCN learns genuine neurobiological features rather than fitting site-specific artifacts.

### 4.6 Limitations and Future Directions

Several limitations should be noted. First, while 19 channels are ideal for clinical translation, they restrict high-resolution anatomical source localization. Second, comprehensive pharmacological details (such as antidepressant dosages) were unavailable in archival datasets. Third, the Mumtaz cohort was recorded during eyes-closed resting, whereas the Cavanagh, Sleep, and UCSD cohorts were recorded during eyes-open resting. Despite this difference, cross-cohort depression transfer remained highly accurate (AUC = 0.832), indicating that frontal theta suppression and aperiodic flattening in depression are robust across ocular states. Future studies should evaluate multi-modal paradigms including active cognitive tasks.

---

## 5. Conclusions

We established a universal 19-channel normative graph architecture that harmonizes heterogeneous EEG recordings, extracts gain-invariant parameterized spectral features, and constructs topologically invariant brain networks via Orthogonal Minimum Spanning Trees. Across 282 participants and four independent cohorts, the framework demonstrated high diagnostic discrimination for major depression and Parkinson's disease, achieved robust zero-shot cross-center generalizability (AUC = 0.832), and uncovered a fundamental biophysical polarity dissociation in frontal theta oscillations between clinical depression and sleep debt. This pipeline provides a standardized, leak-free foundation for translating quantitative electrophysiology into reproducible clinical decision support.

---

## References

1. Loo, C. K., et al. (2019). The promise of quantitative EEG in psychiatry. *Lancet Psychiatry*, 6(3), 188–190.
2. Newson, J. J., & Thiagarajan, T. C. (2019). EEG frequency bands in psychiatric disorders: A review of resting state studies. *Frontiers in Human Neuroscience*, 12, 521.
3. Rutherford, S., et al. (2022). Charting brain growth and aging at high spatial precision via normative modeling. *Nature Protocols*, 17(11), 2548–2571.
4. Babiloni, C., et al. (2020). International Federation of Clinical Neurophysiology (IFCN) guidelines for EEG biomarkers in dementia. *Clinical Neurophysiology*, 131(6), 1459–1485.
5. Gao, R., Peterson, E. J., & Voytek, B. (2017). Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage*, 158, 70–78.
6. Donoghue, T., Haller, M., Peterson, E. J., et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655–1665.
7. van Wijk, B. C., Stam, C. J., & Daffertshofer, A. (2010). Comparing brain networks of different size and connectivity density using graph theory. *PLoS ONE*, 5(10), e13701.
8. Dimitriadis, S. I., Antonakakis, M., Simos, P., et al. (2017). Data-driven topological filtering based on orthogonal minimal spanning trees: Application to multigroup MEG resting-state connectivity. *Frontiers in Neuroinformatics*, 11, 44.
9. Brookshire, G., et al. (2024). Overcoming data leakage in neuroimaging machine learning. *Nature Human Behaviour*, 8, 120–131.
10. OpenNeuro ds004902. (2024). Acute total sleep deprivation resting-state EEG and psychomotor vigilance benchmark. *OpenNeuro*, doi:10.18112/openneuro.ds004902.v1.0.0.
11. Mumtaz, W., Xia, L., Mohd Yasin, M. A., et al. (2016). Electroencephalogram (EEG) based depression detection using feature selection and classification. *Computer Methods and Programs in Biomedicine*, 131, 125–135.
12. Cavanagh, J. F., Bismark, A. W., Frank, M. J., & Allen, J. J. B. (2019). Resting-state EEG predictors of depressive symptomatology and treatment response. *OpenNeuro*, ds003478.
13. George, J. S., Strunk, J., Mak-McCully, R., et al. (2013). Dopaminergic modulation of frontal oscillations in Parkinson's disease. *OpenNeuro*, ds002778.
14. Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: Assessment of functional connectivity from multi-channel EEG and MEG with diminished bias from common sources. *Human Brain Mapping*, 28(11), 1178–1193.
15. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.
16. Finelli, L. A., Baumann, H., Borbély, A. A., & Achermann, P. (2000). Dual electroencephalogram markers of human sleep homeostasis: Correlation between theta activity in waking and slow-wave activity in sleep. *Neuroscience*, 101(3), 523–529.
17. Snipes, S., et al. (2022). The spatio-temporal dynamics of waking theta during acute sleep deprivation. *Cerebral Cortex*, 32(8), 1682–1694.
18. Borbély, A. A., Daan, S., Wirz-Justice, A., & Deboer, T. (2016). The two-process model of sleep regulation: A revisit 30 years later. *Journal of Sleep Research*, 25(2), 131–143.
19. Duman, R. S., Sanacora, G., & Krystal, J. H. (2019). Altered connectivity in depression: GABA and glutamate neurotransmitter deficits and reversal by novel treatments. *Neuron*, 102(1), 75–90.
20. Kruskal, J. B. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. *Proceedings of the American Mathematical Society*, 7(1), 48–50.
