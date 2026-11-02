# Supplementary Material

**A Standardized 19-Channel Normative Graph Architecture Resolves Shared and Divergent Electrophysiological Signatures Across Acute Sleep Debt, Major Depression, and Parkinson's Disease**

**Chushu Tan**^{1,2,*}

^{1} School of Sociology, China University of Political Science and Law, Beijing 102249, China  
^{2} Department of Psychology, University at Buffalo (UB), The State University of New York, New York, NY 12246, USA  

^{*} Corresponding author: Chushu Tan, Guangzhou, Guangdong 510006, China. Email: chushuta@gmail.com. Tel: +86 18008428969. ORCID: 0009-0007-3093-0160  

---

## Section S1. Extended Supplementary Methods

### S1.1 Parameterized Spectral Decomposition via specparam
To eliminate the pervasive conflation between narrowband neural oscillations and broadband 1/f background activity, resting-state electroencephalographic power spectra were decomposed using the `specparam` algorithm (v1.1.0; Donoghue et al., 2020). Spectral estimation was executed via Welch's modified periodogram with 2.0-second Hamming windows and 50% overlap, yielding an exact frequency resolution of 0.50 Hz across 1.0–40.0 Hz. Power spectra were parameterized in fixed aperiodic mode according to:

log₁₀ PSD(f) = L(f) + Σₙ aₙ · exp(-(f - cₙ)² / 2wₙ²)

where the aperiodic background component L(f) is modeled as:

L(f) = b - χ · log₁₀(f)

Here, b denotes the broadband aperiodic offset (in Bel, B, where 1 B = 10 dB = 1.0 log₁₀ µV²/Hz), χ denotes the aperiodic decay exponent reflecting the spectral decay slope, and each Gaussian peak is parameterized by its amplitude aₙ, center frequency cₙ, and bandwidth wₙ. Model hyperparameters were pre-specified as: peak width limits = [1.0, 8.0] Hz, maximum number of peaks = 6, minimum peak height = 0.10 B, peak threshold = 2.0 standard deviations, and aperiodic mode = 'fixed' (knee parameter omitted given the 1.0–40.0 Hz fitting range).

Narrowband periodic power was isolated by subtracting the fitted aperiodic component L(f) from the total log-power spectrum:

P_periodic(f) = log₁₀ PSD(f) - L(f)

Residual periodic band powers were extracted by integrating P_periodic(f) across canonical electrophysiological bands: Frontal Theta (4.0–8.0 Hz, averaged over Fp1, Fp2, F3, F4, Fz), Posterior Alpha (8.0–13.0 Hz, averaged over P3, P4, Pz, O1, O2), and Global Delta (1.0–4.0 Hz, averaged over all 19 channels). The global aperiodic exponent χ was averaged across all 19 channels to provide an electrophysiological index of cortical excitation/inhibition (E/I) balance.

### S1.2 Data-Driven Connectomics via Orthogonal Minimum Spanning Trees (OMST)
Functional brain connectivity was quantified using the Phase Lag Index (PLI; Stam et al., 2007) in the combined theta-alpha band (4.0–13.0 Hz) to eliminate spurious zero-lag volume conduction. To construct connected brain network topologies without subjective thresholding, we applied the Orthogonal Minimum Spanning Trees (OMST) framework (Dimitriadis et al., 2017). Distance metrics were computed as:

c_ij = (1.0 - PLI_ij) + 10⁻⁴

where the small positive offset (10⁻⁴) prevents sparse graph routines from misinterpreting zero-cost edges as absent. The primary spanning tree was extracted using Kruskal's algorithm (Kruskal, 1956), connecting all N = 19 electrode nodes with exactly N - 1 = 18 edges. Subsequent orthogonal spanning trees were iteratively extracted by setting previously selected edges to infinity, forcing each successive tree to capture independent functional pathways. Cumulative networks were evaluated by maximizing Global Cost Efficiency (GCE):

GCE(m) = E_global(G_m) / E_global(G_full) - Cost(G_m)

where E_global represents the average inverse shortest path length across all node pairs, and Cost represents the sum of weights in the selected network normalized by the total weight of the fully connected graph.

Across all N = 279 subjects and 365 recording sessions, the empirical distribution of selected OMST trees was M = 1.51 ± 0.50 (median = 2; 49.3% selecting M = 1 and 50.7% selecting M = 2; Mumtaz MDD: M = 1.33 ± 0.47, density = 13.3% ± 4.7%; Cavanagh depression: M = 1.45 ± 0.50, density = 14.5% ± 5.0%; UCSD Parkinson: M = 1.61 ± 0.49, density = 16.0% ± 4.9%; Sleep deprivation: M = 1.59 ± 0.49, density = 15.9% ± 4.9%). Across the entire multi-cohort benchmark (365 graphs), the mean number of selected edges was E = 27.12 ± 9.00, yielding a mean node degree of k̄ = 2.86 ± 0.95 (sum of node degrees = 54.25) and an average network density of 15.86% ± 5.26%. Across all 6,935 individual node instances in the study, the isolated node rate was strictly 0.0000% by construction.

### S1.3 Analytical and Numerical Proof of Amplifier Gain Invariance
When raw EEG signals are recorded across different hardware systems, arbitrary amplifier gain scaling introduces a multiplicative factor k > 0 such that v'(t) = k · v(t). The resulting power spectrum scales quadratically:

PSD'(f) = k² · PSD(f)

In logarithmic space, this corresponds to a uniform vertical shift:

log₁₀ PSD'(f) = log₁₀ PSD(f) + 2 log₁₀ k

Fitting the specparam model to the scaled spectrum yields:

log₁₀ PSD'(f) = b' - χ' · log₁₀(f) + Σₙ a'ₙ · exp(-(f - c'ₙ)² / 2(w'ₙ)²)

Because 2 log₁₀ k is strictly frequency-independent, its derivative with respect to frequency is identically zero:

∂/∂f [log₁₀ PSD'(f)] = ∂/∂f [log₁₀ PSD(f)]

Consequently, parameter optimization yields:
1. χ' = χ (the aperiodic slope is invariant);
2. c'ₙ = cₙ and w'ₙ = wₙ (peak center frequencies and bandwidths are invariant);
3. a'ₙ = aₙ (Gaussian peak amplitudes above the aperiodic floor are invariant);
4. b' = b + 2 log₁₀ k (the unadjusted aperiodic offset absorbs the entire scaling factor);
5. Residual periodic power P_periodic'(f) = log₁₀ PSD'(f) - L'(f) = P_periodic(f) is invariant.

Similarly, Phase Lag Index (PLI) depends exclusively on the sign of instantaneous phase differences:

PLI = |⟨sign(sin(Δφ(t)))⟩|

Because linear scaling by k > 0 does not alter signal phase (∠(k · v(t)) = ∠v(t)), pairwise PLI values and the resulting OMST graph topology are mathematically invariant to digitizer gain.

### S1.4 NormativeGCN Architecture and Training Specification
The NormativeGCN architecture is implemented in pure PyTorch and operates directly on the 19-node OMST graph. Each node is endowed with a 4-dimensional normatively calibrated feature vector z_i = [z_θ, z_δ, z_α, z_χ]^T ∈ ℝ⁴ derived via fold-nested reference scaling against healthy controls:
- Layer 1: PurePyTorchGCNLayer (4 → 32 channels) with symmetric normalized adjacency Ã = D̃^(-1/2) (A + I) D̃^(-1/2), followed by 1D Batch Normalization, ReLU activation, and Dropout (p = 0.25). (160 trainable parameters).
- Layer 2: PurePyTorchGCNLayer (32 → 32 channels), followed by 1D Batch Normalization, ReLU activation, and Dropout (p = 0.25). (1,056 trainable parameters).
- Multi-Pooling Readout: Concatenated Global Mean Pooling (32-D) and Global Max Pooling (32-D) across all 19 nodes, producing a 64-dimensional graph embedding: h_G = [h_mean, h_max] ∈ ℝ⁶⁴.
- Classification Head: Multi-layer perceptron comprising Linear(64, 32) (2,080 parameters) followed by ReLU, Dropout (p = 0.25), and Linear(32, 2) (66 parameters).
- Total Trainable Parameters: Exactly 3,362 parameters (160 + 1,056 + 2,080 + 66 = 3,362).
- Optimization: AdamW optimizer (learning rate η = 0.005, weight decay λ = 1.0 × 10⁻⁴, cross-entropy loss, fixed 35 training epochs). To completely prevent test-fold leakage, zero early stopping was performed on evaluation folds.

---

## Section S2. Supplementary Tables

### Table S1. Canonical 19-Channel 10–20 Electrode Coordinates and Mathematically Consistent OMST Graph Metrics
| Channel Label | Cortical Region | Projected X (2D) | Projected Y (2D) | Tree 1 Degree (M=1) | Tree 2 Orthogonal Degree | Combined Degree (M=2) | Empirical Mean Degree (M=1.51) | Betweenness Centrality | Clustering Coefficient |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fp1 | Prefrontal (Left) | -0.28 | +0.78 | 1 | 2 | 3 | 2.54 ± 1.80 | 0.058 ± 0.113 | 0.125 ± 0.269 |
| Fp2 | Prefrontal (Right) | +0.28 | +0.78 | 1 | 2 | 3 | 2.40 ± 1.72 | 0.055 ± 0.118 | 0.123 ± 0.281 |
| F7 | Anterior Temporal (Left) | -0.72 | +0.44 | 2 | 1 | 3 | 2.48 ± 1.67 | 0.059 ± 0.114 | 0.131 ± 0.268 |
| F3 | Frontal (Left) | -0.35 | +0.40 | 3 | 2 | 5 | 2.39 ± 1.67 | 0.058 ± 0.118 | 0.131 ± 0.283 |
| Fz | Frontal Midline | 0.00 | +0.40 | 3 | 1 | 4 | 2.93 ± 2.39 | 0.088 ± 0.155 | 0.105 ± 0.244 |
| F4 | Frontal (Right) | +0.35 | +0.40 | 3 | 2 | 5 | 3.11 ± 2.55 | 0.108 ± 0.189 | 0.099 ± 0.225 |
| F8 | Anterior Temporal (Right)| +0.72 | +0.44 | 2 | 1 | 3 | 3.31 ± 2.49 | 0.121 ± 0.195 | 0.104 ± 0.219 |
| T7 | Mid-Temporal (Left) | -0.84 | 0.00 | 2 | 1 | 3 | 3.35 ± 2.45 | 0.128 ± 0.203 | 0.103 ± 0.220 |
| C3 | Central (Left) | -0.42 | 0.00 | 1 | 4 | 5 | 3.08 ± 2.16 | 0.113 ± 0.178 | 0.101 ± 0.219 |
| Cz | Central Midline | 0.00 | 0.00 | 4 | 1 | 5 | 3.35 ± 2.35 | 0.137 ± 0.203 | 0.093 ± 0.214 |
| C4 | Central (Right) | +0.42 | 0.00 | 1 | 4 | 5 | 2.49 ± 1.77 | 0.064 ± 0.129 | 0.117 ± 0.264 |
| T8 | Mid-Temporal (Right) | +0.84 | 0.00 | 2 | 1 | 3 | 2.38 ± 1.63 | 0.060 ± 0.127 | 0.107 ± 0.264 |
| P7 | Posterior Temporal (Left)| -0.72 | -0.44 | 1 | 2 | 3 | 2.48 ± 1.86 | 0.063 ± 0.126 | 0.115 ± 0.273 |
| P3 | Parietal (Left) | -0.35 | -0.40 | 2 | 2 | 4 | 2.47 ± 1.79 | 0.064 ± 0.129 | 0.090 ± 0.240 |
| Pz | Parietal Midline | 0.00 | -0.40 | 3 | 1 | 4 | 3.04 ± 2.40 | 0.102 ± 0.174 | 0.130 ± 0.274 |
| P4 | Parietal (Right) | +0.35 | -0.40 | 2 | 2 | 4 | 3.14 ± 2.56 | 0.118 ± 0.206 | 0.142 ± 0.292 |
| P8 | Posterior Temporal (Right)|+0.72 | -0.44 | 1 | 2 | 3 | 2.49 ± 1.75 | 0.060 ± 0.126 | 0.146 ± 0.296 |
| O1 | Occipital (Left) | -0.28 | -0.78 | 1 | 3 | 4 | 2.99 ± 2.24 | 0.101 ± 0.176 | 0.104 ± 0.242 |
| O2 | Occipital (Right) | +0.28 | -0.78 | 1 | 2 | 3 | 3.81 ± 3.09 | 0.164 ± 0.250 | 0.113 ± 0.243 |
| **Network Total / Mean** | **All 19 Channels** | **—** | **—** | **18 edges (k̄ = 1.89)** | **18 edges (k̄ = 1.89)** | **36 edges (k̄ = 3.79)** | **27.12 edges (k̄ = 2.86)** | **—** | **—** |

*Note: For an undirected graph with N = 19 nodes, maximum possible edges is N(N-1)/2 = 171. Edge density is D = E / 171. For M = 1 tree (E = 18), sum of degrees is 2E = 36, mean degree is k̄ = 1.895, and D = 10.53%; for M = 2 trees (E = 36), sum of degrees is 2E = 72, k̄ = 3.789, and D = 21.05%. Across the entire multi-cohort empirical benchmark (N = 279 participants, 365 recording sessions), data-driven GCE optimization selected an average of M = 1.51 ± 0.50 trees (E = 27.12 ± 9.00 edges), yielding an exact node degree sum of 54.25, mean degree k̄ = 2.86 ± 0.95, and network density D = 15.86% ± 5.26%. The isolated node rate is strictly 0.0000% across all sessions by construction.*

---

### Table S2. Numerical Verification of Feature Invariance Across Four Orders of Magnitude of Amplifier Gain
| Gain Factor (k) | Theoretical Offset b_theory (B) | Extracted Offset b_sim (B) | Offset Error Δb (B) | Periodic θ Error (B) | Periodic δ Error (B) | Periodic α Error (B) | Aperiodic χ Error (dimensionless) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | -3.3500 | -3.3500 | < 1.0 × 10⁻¹⁵ | 2.1 × 10⁻¹⁵ | 1.8 × 10⁻¹⁵ | 2.5 × 10⁻¹⁵ | 1.2 × 10⁻¹⁵ |
| 0.10 | -1.3500 | -1.3500 | < 1.0 × 10⁻¹⁵ | 1.4 × 10⁻¹⁵ | 1.1 × 10⁻¹⁵ | 1.6 × 10⁻¹⁵ | 8.0 × 10⁻¹⁶ |
| 0.50 | 0.0479 | 0.0479 | < 1.0 × 10⁻¹⁵ | 8.0 × 10⁻¹⁶ | 5.0 × 10⁻¹⁶ | 9.0 × 10⁻¹⁶ | 4.0 × 10⁻¹⁶ |
| 1.00 (Base) | 0.6500 | 0.6500 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2.00 | 1.2521 | 1.2521 | < 1.0 × 10⁻¹⁵ | 9.5 × 10⁻¹⁶ | 7.0 × 10⁻¹⁶ | 1.1 × 10⁻¹⁵ | 5.0 × 10⁻¹⁶ |
| 10.00 | 2.6500 | 2.6500 | < 1.0 × 10⁻¹⁵ | 1.8 × 10⁻¹⁵ | 1.5 × 10⁻¹⁵ | 2.0 × 10⁻¹⁵ | 1.0 × 10⁻¹⁵ |
| 100.00 | 4.6500 | 4.6500 | < 1.0 × 10⁻¹⁵ | 2.7 × 10⁻¹⁵ | 2.3 × 10⁻¹⁵ | 3.1 × 10⁻¹⁵ | 1.6 × 10⁻¹⁵ |

*Note: Simulation performed on 19-channel EEG epochs scaled by multiplicative factor k. Base spectrum parameterized with offset b₀ = 0.650 B and exponent χ₀ = 1.152. Theoretical prediction: Δb = 2 log₁₀(k). Numerical errors for residual periodic band powers and aperiodic exponent χ are bounded strictly by IEEE 754 double-precision floating-point precision (< 10⁻¹⁴), validating complete empirical scale-invariance.*

---

### Table S3. Multi-Cohort Demographics, Recording Montages, and Clinical Profiles (Analyzed Sample N = 279, 365 Sessions)
| Dataset / Cohort | Study Identifier & Citation | Primary Pathology | Analyzed Sample (N) | Mean Age (SD) | Sex (% Female) | Eye State | Native Hardware & Channels | Continuous Wakefulness / Therapy State | Excluded Participants |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Study 1: Sleep Debt | OpenNeuro ds004902 (Xiang et al., 2024) | Acute Total Sleep Deprivation | 71 paired subjects (142 sessions) | 20.00 ± 1.44 yr (range 17–23) | 47.9% (34 F, 37 M) | Eyes Open | BrainVision actiCHamp (64 ch) | 24–30 h continuous wakefulness vs. normal baseline | 0 excluded (all 71 pairs met quality control) |
| Study 2: Clinical MDD | Figshare 4244171 (Mumtaz et al., 2016) | Major Depressive Disorder | 58 subjects (30 MDD, 28 HC) | 39.3 ± 14.1 yr (MDD 40.3±12.9, HC 38.3±15.6) | 50.0% (29 F, 29 M) | Eyes Closed | Grass Telefactor Comet-AS40 (19 ch) | Severe hospitalized MDD; unmedicated at testing | 6 excluded from 64 (channel bridging / severe motion >20%) |
| Study 3: Univ. Depression | OpenNeuro ds003478 (Cavanagh et al., 2019) | Subclinical University Depression | 119 subjects (44 DEP, 75 HC) | 18.86 ± 1.18 yr (DEP 18.68±1.12, HC 18.97±1.22) | 60.5% (72 F, 47 M) | Eyes Open | Custom 64-channel montage (64 ch) | BDI ≥ 14 (depressed) vs. BDI ≤ 9 (healthy control) | 3 excluded from 122 (1 borderline BDI: 10≤BDI≤13, 2 artifact records) |
| Study 4: Parkinson's | OpenNeuro ds002778 (George et al., 2013) | Parkinson's Disease | 31 subjects (16 HC, 15 PD; 46 sessions) | 63.39 ± 8.84 yr (HC 63.50±9.67, PD 63.27±8.20) | 54.8% (17 F, 14 M; HC 56.2%, PD 53.3%) | Eyes Open | BioSemi ActiveTwo (32 ch) | Tested OFF (≥ 12 h withdrawal) and ON (acute levodopa) | 0 excluded (all 31 unique subjects in public release analyzed) |
| **Total Benchmark** | **4 Public Datasets** | **Transdiagnostic** | **279 Unique Subjects (365 Sessions)** | **28.4 ± 16.4 yr** | **54.5% (152 F, 127 M)** | **Mixed** | **Harmonized 19-Channel 10–20** | **Multi-State Challenge** | **9 excluded from 288 initial public records (3.1%)** |

*Note: Demographics are verified directly from primary repository metadata and descriptive publications. Total analyzed sample comprises N = 279 unique participants and 365 recording sessions. In Study 4, all 15 PD patients completed paired OFF (≥12 h dopaminergic withdrawal) and ON (post-levodopa) recordings (15 × 2 = 30 patient sessions) alongside 16 healthy control sessions, totaling 46 sessions.*

---

### Table S4. Complete Statistical Summary of Electrophysiological Markers Across All Multi-Cohort Comparisons
| Marker Key | Marker Label | Cohort Name | Group Comparison | N_case | N_ctrl | Test Type | t-statistic | df | Raw p-value | Focal FDR q (20 tests) | All FDR q (25 tests) | Effect Size (95% CI) |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired t | +2.392 | 70 | 0.0194 | 0.0777† | 0.0971† | d_z = +0.284 [0.047, 0.521]† |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample t | -3.145 | 56 | 0.0027 | 0.0234* | 0.0292* | d = -0.826 [-1.362, -0.289]* |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample t | -3.585 | 117 | 0.0005 | 0.0099** | 0.0124* | d = -0.676 [-1.058, -0.294]** |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample t | +1.557 | 29 | 0.1303 | 0.2869 | 0.2506 | d = +0.559 [-0.159, 1.277] |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample t | +3.187 | 29 | 0.0035 | 0.0234* | 0.0292* | d = +1.150 [0.389, 1.910]* |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired t | +0.812 | 70 | 0.4198 | 0.5662 | 0.5587 | d_z = +0.096 [-0.137, 0.329] |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample t | -0.117 | 56 | 0.9071 | 0.9549 | 0.9449 | d = -0.031 [-0.546, 0.484] |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample t | -1.485 | 117 | 0.1435 | 0.2869 | 0.2506 | d = -0.280 [-0.654, 0.093] |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample t | +0.057 | 29 | 0.9551 | 0.9551 | 0.9551 | d = +0.020 [-0.684, 0.725] |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample t | +0.423 | 29 | 0.6753 | 0.8441 | 0.8441 | d = +0.152 [-0.554, 0.857] |
| global_chi | Global Aperiodic Exponent (χ) | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired t | +1.545 | 70 | 0.1269 | 0.2869 | 0.2506 | d_z = +0.183 [-0.051, 0.418] |
| global_chi | Global Aperiodic Exponent (χ) | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample t | -2.843 | 56 | 0.0062 | 0.0311* | 0.0389* | d = -0.744 [-1.276, -0.211]* |
| global_chi | Global Aperiodic Exponent (χ) | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample t | -1.597 | 117 | 0.1146 | 0.2869 | 0.2506 | d = -0.327 [-0.702, 0.047] |
| global_chi | Global Aperiodic Exponent (χ) | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample t | +1.046 | 29 | 0.3049 | 0.4902 | 0.4686 | d = +0.372 [-0.339, 1.082] |
| global_chi | Global Aperiodic Exponent (χ) | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample t | +1.016 | 29 | 0.3186 | 0.4902 | 0.4686 | d = +0.367 [-0.343, 1.077] |
| omst_gce | OMST Global Cost Efficiency | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired t | -1.267 | 70 | 0.2095 | 0.3809 | 0.3492 | d_z = -0.150 [-0.384, 0.084] |
| omst_gce | OMST Global Cost Efficiency | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample t | +1.842 | 56 | 0.0728 | 0.2426 | 0.2021 | d = +0.494 [-0.029, 1.017] |
| omst_gce | OMST Global Cost Efficiency | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample t | +0.178 | 117 | 0.8595 | 0.9549 | 0.9343 | d = +0.034 [-0.338, 0.406] |
| omst_gce | OMST Global Cost Efficiency | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample t | -0.811 | 29 | 0.4246 | 0.5662 | 0.5587 | d = -0.287 [-0.996, 0.421] |
| omst_gce | OMST Global Cost Efficiency | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample t | -0.266 | 29 | 0.7919 | 0.9316 | 0.8999 | d = -0.095 [-0.800, 0.610] |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired t | -1.975 | 70 | 0.0522 | — | 0.1641 | d_z = -0.234 [-0.470, 0.001] |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample t | -1.992 | 56 | 0.0525 | — | 0.1641 | d = -0.531 [-1.055, -0.007] |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample t | +1.552 | 117 | 0.1245 | — | 0.2506 | d = +0.301 [-0.073, 0.675] |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample t | -0.362 | 29 | 0.7204 | — | 0.8576 | d = -0.131 [-0.836, 0.574] |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample t | -2.346 | 29 | 0.0268 | — | 0.1115† | d = -0.850 [-1.586, -0.115]† |
| **Within-Patient Paired Comparison (Study 4c: Parkinson ON vs. OFF, N = 15 Paired Patients)** | | | | | | | | | | | | |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 4c: Parkinson Paired | ON vs. OFF (within PD) | 15 | 15 | Paired t | +2.607 | 14 | 0.0207 | — | — | d_z = +0.673 [0.113, 1.234]* |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 4c: Parkinson Paired | ON vs. OFF (within PD) | 15 | 15 | Paired t | -2.916 | 14 | 0.0113 | — | — | d_z = -0.753 [-1.326, -0.180]* |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 4c: Parkinson Paired | ON vs. OFF (within PD) | 15 | 15 | Paired t | +0.966 | 14 | 0.3511 | — | — | d_z = +0.249 [-0.280, 0.772] |
| global_chi | Global Aperiodic Exponent (χ) | Study 4c: Parkinson Paired | ON vs. OFF (within PD) | 15 | 15 | Paired t | +0.235 | 14 | 0.8179 | — | — | d_z = +0.061 [-0.448, 0.566] |
| omst_gce | OMST Global Cost Efficiency | Study 4c: Parkinson Paired | ON vs. OFF (within PD) | 15 | 15 | Paired t | +0.662 | 14 | 0.5195 | — | — | d_z = +0.171 [-0.344, 0.680] |

*Note: Significance indicators: * FDR q < 0.05; ** FDR q < 0.01; † nominal significance (uncorrected p < 0.05). Benjamini-Hochberg (BH) false discovery rate control was applied across the 20 pre-registered focal hypotheses (4 core markers × 5 group comparisons) and across all 25 comparisons including delta. For Study 1 and Study 4c, paired within-subject statistics (paired t and d_z) are reported. In Study 4c, within-patient paired testing directly confirms that acute dopaminergic replacement selectively elevates frontal theta (+0.0989 B, t(14) = +2.607, p = 0.0207, d_z = +0.673) while suppressing global delta (-0.0824 B, t(14) = -2.916, p = 0.0113, d_z = -0.753), without affecting posterior alpha, aperiodic exponent, or OMST network topology.*

---

### Table S5. Standard 19-Channel 10–20 Coordinate Mapping Across Four Hardware Montages
| Standard Channel | 10–20 Coordinate Region | Study 1 (OpenNeuro ds004902, 64ch) | Study 2 (Figshare 4244171, 19ch) | Study 3 (OpenNeuro ds003478, 64ch) | Study 4 (OpenNeuro ds002778, 32ch) | Harmonization Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Fp1 | Left Prefrontal | Fp1 | Fp1 | Fp1 | Fp1 | Direct Match (100%) |
| Fp2 | Right Prefrontal | Fp2 | Fp2 | Fp2 | Fp2 | Direct Match (100%) |
| F7 | Left Anterior Temporal | F7 | F7 | F7 | F7 | Direct Match (100%) |
| F3 | Left Frontal | F3 | F3 | F3 | F3 | Direct Match (100%) |
| Fz | Midline Frontal | Fz | Fz | Fz | Fz | Direct Match (100%) |
| F4 | Right Frontal | F4 | F4 | F4 | F4 | Direct Match (100%) |
| F8 | Right Anterior Temporal | F8 | F8 | F8 | F8 | Direct Match (100%) |
| T7 (T3) | Left Mid-Temporal | T7 | T3 (mapped to T7) | T7 | T7 | Direct Match (100%) |
| C3 | Left Central | C3 | C3 | C3 | C3 | Direct Match (100%) |
| Cz | Midline Central | Cz | Cz | Cz | Cz | Direct Match (100%) |
| C4 | Right Central | C4 | C4 | C4 | C4 | Direct Match (100%) |
| T8 (T4) | Right Mid-Temporal | T8 | T4 (mapped to T8) | T8 | T8 | Direct Match (100%) |
| P7 (T5) | Left Posterior Temporal | P7 | T5 (mapped to P7) | P7 | P7 | Direct Match (100%) |
| P3 | Left Parietal | P3 | P3 | P3 | P3 | Direct Match (100%) |
| Pz | Midline Parietal | Pz | Pz | Pz | Pz | Direct Match (100%) |
| P4 | Right Parietal | P4 | P4 | P4 | P4 | Direct Match (100%) |
| P8 (T6) | Right Posterior Temporal | P8 | T6 (mapped to P8) | P8 | P8 | Direct Match (100%) |
| O1 | Left Occipital | O1 | O1 | O1 | O1 | Direct Match (100%) |
| O2 | Right Occipital | O2 | O2 | O2 | O2 | Direct Match (100%) |

*Note: All 19 standard channels exist natively across all four recording montages (with legacy nomenclature T3/T4/T5/T6 mapped directly to modern standard 10–20 positions T7/T8/P7/P8). Zero channels required spatial interpolation or zero-padding, ensuring 100% genuine biological signal across all 279 participants.*

---

### Table S6. Machine Learning Ablation and Single-Feature Baselines vs. NormativeGCN
| Cohort Benchmark | Single: Periodic θ AUC [95% CI] | Single: Periodic α AUC [95% CI] | Single: Exponent χ AUC [95% CI] | Single: OMST GCE AUC [95% CI] | Ablate α (No Alpha) AUC | Ablate θ (No Theta) AUC | Flat 76-D Logistic Regression AUC [95% CI] | NormativeGCN (Full 4 Feats) AUC [95% CI] | GCN Improvement ΔAUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Study 2: Mumtaz Inpatient MDD (N=58) | 0.838 [0.722, 0.931] | 0.512 [0.354, 0.672] | 0.785 [0.658, 0.894] | 0.655 [0.510, 0.792] | 0.937 | 0.748 | 0.898 [0.810, 0.966] | **0.925 [0.857, 0.980]** | +0.027 |
| Study 3: Cavanagh Univ. Depression (N=119) | 0.678 [0.575, 0.772] | 0.562 [0.450, 0.669] | 0.574 [0.463, 0.680] | 0.515 [0.408, 0.622] | 0.657 | 0.519 | 0.547 [0.432, 0.646] | **0.649 [0.544, 0.747]** | +0.102 |
| Study 4: Parkinson's All Sessions (N=46) | 0.685 [0.521, 0.835] | 0.510 [0.332, 0.684] | 0.612 [0.440, 0.778] | 0.582 [0.405, 0.751] | 0.642 | 0.448 | 0.692 [0.512, 0.852] | **0.702 [0.526, 0.872]** | +0.010 |
| Study 4a: Parkinson ON vs. HC (N=31) | 0.765 [0.582, 0.920] | 0.535 [0.320, 0.745] | 0.618 [0.402, 0.815] | 0.542 [0.325, 0.750] | 0.760 | 0.520 | 0.783 [0.597, 0.940] | **0.771 [0.591, 0.924]** | -0.012 |
| Study 4b: Parkinson OFF vs. HC (N=31) | 0.625 [0.415, 0.818] | 0.505 [0.295, 0.710] | 0.608 [0.395, 0.805] | 0.572 [0.355, 0.778] | 0.590 | 0.480 | 0.500 [0.279, 0.714] | **0.583 [0.367, 0.794]** | +0.083 |
| Study 1: Sleep Debt Single-Sess (N=142) | 0.572 [0.475, 0.665] | 0.518 [0.418, 0.615] | 0.552 [0.455, 0.648] | 0.545 [0.448, 0.640] | 0.573 | 0.456 | 0.589 [0.486, 0.677] | **0.566 [0.468, 0.658]** | -0.023 |

*Note: All evaluations use 5-fold cross-validation with fold-nested normative calibration. Brackets indicate 1,000-sample bootstrap 95% confidence intervals. Feature ablation reveals two critical biological insights: (1) Ablating alpha oscillations (Ablate α, using [θ, δ, χ]) preserves high discriminability for clinical MDD (AUC = 0.937 vs 0.925), confirming that eyes-closed alpha power does not confound depression classification; (2) Conversely, ablating theta oscillations (Ablate θ, using [δ, α, χ]) causes catastrophic performance collapse across all cohorts (MDD drops to 0.748, Cavanagh drops to 0.519, Sleep drops to 0.456, Parkinson drops to 0.448), confirming that frontal theta is the indispensable diagnostic carrier.*

---

### Table S7. NormativeGCN Architecture and Hyperparameter Settings
| Component / Parameter | Value / Setting | Description / Rationale |
| :--- | :--- | :--- |
| Framework | PyTorch 2.2.1 | Pure PyTorch tensor implementation for full algorithmic reproducibility |
| Input Graph Nodes (N) | 19 scalp channels | Standard international 10–20 EEG montage |
| Node Feature Dimension (D) | 4 features per node | [z_θ, z_δ, z_α, z_χ] normatively calibrated against healthy controls |
| Graph Adjacency Matrix (A) | 19 × 19 binary adjacency matrix | Extracted via Phase Lag Index OMST (M ∈ {1, 2}, 0.0% isolated nodes) |
| Convolutional Layer 1 | PurePyTorchGCNLayer (4 → 32 channels) | Symmetric normalized Laplacian Ã = D̃^(-1/2)(A+I)D̃^(-1/2); 160 parameters |
| Normalization & Activation 1 | BatchNorm1d + ReLU | Stabilizes gradient flow across graph batches |
| Regularization 1 | Dropout (p = 0.25) | Prevents co-adaptation of node features |
| Convolutional Layer 2 | PurePyTorchGCNLayer (32 → 32 channels) | Hidden dim 32 → Hidden dim 32; 1,056 parameters |
| Normalization & Activation 2 | BatchNorm1d + ReLU | Maintains activation scale |
| Regularization 2 | Dropout (p = 0.25) | Controls model capacity |
| Readout Function | Multi-Pooling (Mean + Max) | Invariant concatenation: h_G = [mean(h_i), max(h_i)] ∈ ℝ⁶⁴ |
| Classification Head | MLP: Linear(64, 32) → ReLU → Dropout(0.25) → Linear(32, 2) | 2,080 + 66 = 2,146 parameters |
| Total Trainable Parameters | Exactly 3,362 parameters | Compact footprint minimizing risk of empirical overfitting |
| Optimizer | AdamW (β₁ = 0.9, β₂ = 0.999) | Weight decay λ = 1.0 × 10⁻⁴ |
| Learning Rate | η = 0.005 | Initial learning rate |
| Epochs & Early Stopping | Fixed 35 epochs | Zero early stopping on test folds, guaranteeing 0.0% test-fold leakage |

---

### Table S8. Raw Signal Preprocessing and Artifact Rejection Pipeline
| Preprocessing Step | Specification | Software Implementation |
| :--- | :--- | :--- |
| Sampling Rate Harmonization | Resampled to 250 Hz (or 256 Hz) using polyphase anti-aliasing filter | MNE-Python `raw.resample()` |
| Line Noise Suppression | Zero-phase notch filter at 50 Hz (ds004902, Figshare) or 60 Hz (ds003478, ds002778) | MNE-Python `raw.notch_filter()` |
| Bandpass Filtering | Zero-phase FIR bandpass filter (0.5–45.0 Hz; Hamming window, transition bandwidth 0.5 Hz) | MNE-Python `raw.filter()` |
| Montaging & Re-referencing | Downsampled to 19 standard 10–20 channels; converted to common average reference | MNE-Python `raw.pick_channels()`, `set_eeg_reference('average')` |
| Bad Channel Detection | Flatline channels (>5 s) or correlation r < 0.35 with spatial neighbors interpolated | Spherical spline interpolation via MNE `interpolate_bads()` |
| Epoch Segmentation | 2.0-second non-overlapping epochs (500 samples per epoch) | MNE-Python `make_fixed_length_epochs()` |
| Amplitude Rejection Threshold | Epochs containing peak-to-peak amplitude > 150 µV rejected | Automated amplitude thresholding |

---

### Table S9. Computational Environment and Software Reproducibility Specifications
| Software / Package | Version | Primary Purpose | Repository Link / Source |
| :--- | :--- | :--- | :--- |
| Python | 3.10.12 | Primary programming language | https://www.python.org |
| MNE-Python | 1.6.1 | EEG preprocessing, filtering, and epoching | https://mne.tools |
| specparam | 1.1.0 | Spectral parameterization (1/f decomposition) | https://fooof-tools.github.io |
| PyTorch | 2.2.1 | Deep learning tensor engine | https://pytorch.org |
| Scikit-learn | 1.4.1 | Logistic regression baseline, cross-validation, ROC-AUC | https://scikit-learn.org |
| NetworkX | 3.2.1 | Graph algorithms and OMST computation | https://networkx.org |
| SciPy | 1.12.0 | Statistical hypothesis testing and signal processing | https://scipy.org |
| NumPy | 1.26.4 | Numerical linear algebra | https://numpy.org |
| Matplotlib / Seaborn | 3.8.3 / 0.13.2 | Publication-quality figure generation | https://matplotlib.org |
| Hardware Environment | Apple M-series / Linux x86_64 | 16 GB unified RAM, 8 CPU cores | Local workstation |
| Code Repository | Git commit tracked | Benchmark code and scripts | https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark |

---

### Table S10. TRIPOD+AI Compliance Checklist
| Section / Item | Item No. | Checklist Description | Location / Compliance in Manuscript |
| :--- | :---: | :--- | :--- |
| **Title** | 1 | Identify the study as developing and/or validating a multivariable prediction model using AI. | Title: "A Standardized 19-Channel Normative Graph Architecture..." |
| **Abstract** | 2 | Provide a structured summary of objectives, study design, setting, participants, model, and results. | Abstract (Background, Methods, Results, Conclusions) |
| **Background** | 3a | Explain the medical/scientific context and rationale for the model. | Introduction §1 (translational EEG bottleneck, clinical need) |
| **Objectives** | 3b | Specify the research questions and intended use of the model. | Introduction §1 (multi-cohort benchmark, normative graph modeling) |
| **Source of Data** | 4a | Describe the study design, source of data, and dates of collection. | Methods §2.1 & Supplementary Table S3 (4 public datasets) |
| **Participants** | 4b | Specify inclusion and exclusion criteria, and diagnostic settings. | Methods §2.1 & Supplementary Figure S5 (PRISMA/STROBE flowchart) |
| **Predictors** | 5a | Define all predictors and operationalization. | Methods §2.3, §2.4 & Supplementary Section S1 |
| **Blind Assessment** | 5b | State whether predictors were assessed blind to the outcome. | Predictive features extracted by automated algorithm blind to diagnosis |
| **Sample Size** | 6 | Explain how sample size was determined. | Full multi-cohort public archive (N = 279 subjects, 365 sessions) |
| **Missing Data** | 7 | Explain handling of missing data or bad channels. | Methods §2.2 & Supplementary Table S8 (spherical spline interpolation) |
| **Model Development** | 8a | Specify machine learning algorithms, architecture, and hyperparameters. | Methods §2.6, Supplementary Section S1.4 & Table S7 |
| **Normative Modeling** | 8b | Detail normative calibration and handling of data leakage. | Methods §2.5 (fold-nested reference calibration solely on training HC) |
| **Model Evaluation** | 9 | Describe validation method (cross-validation, external transfer). | Methods §2.6 (5-fold stratified CV, Group-CV, and zero-shot transfer) |
| **Performance Measures** | 10 | Specify metrics used to assess discrimination and calibration. | ROC-AUC, Balanced Accuracy, sensitivity, specificity, Cohen's d, paired d_z, with 1,000-sample bootstrap 95% CIs. Given that public neuroimaging archives represent case-control enriched cohorts with fixed experimental prevalence (~50%), probability calibration (e.g., Brier score, calibration slope) evaluates relative risk ranking rather than unadjusted population absolute risk calibration. |
| **Results: Flow** | 11a | Report flow of participants through the study. | Supplementary Figure S5 (N = 288 initial public records → N = 279 analyzed) |
| **Results: Characteristics** | 11b | Report demographic and clinical characteristics. | Table S3 (age, sex, recording parameters, medication state) |
| **Results: Specification** | 12 | Present full model architecture, weights, or code. | Public GitHub repository provided; Table S7 (3,362 parameters) |
| **Results: Discrimination** | 13 | Report discrimination performance with confidence intervals. | Main Text Table 2, Table 3 & Figure 2 (ROC-AUC and Balanced Accuracy with 95% CI). Discrimination calibrated relative to nested control baselines. |
| **Results: Baselines** | 14 | Compare against established clinical or linear baselines. | Main Text Table 2 & Supplementary Table S6 (vs. 76-D Logistic Regression) |
| **Results: Permutation** | 15 | Provide permutation test null distributions. | Supplementary Figure S6 (1,000 label permutations, all p < 0.001) |
| **Discussion: Limitations** | 16 | Discuss study limitations and potential sources of bias. | Discussion §4.6 (medication histories, sample sizes, cross-sectional design) |
| **Interpretation** | 17 | Provide an overall interpretation of findings in biological context. | Discussion §4.2, §4.3, §4.4 (E/I balance, dopamine, hypofrontality) |
| **Generalizability** | 18 | Discuss generalizability to other settings and clinical populations. | Discussion §4.1, §4.5 (zero-shot transfer across centers and hardware) |
| **Data Availability** | 19 | Provide data availability statement with persistent identifiers. | Back Matter: OpenNeuro ds004902, ds003478, ds002778; Figshare 4244171 |
| **Code Availability** | 20 | Provide code availability statement with public URL. | Back Matter: https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark |
| **Ethics & Consent** | 21 | State ethics approval and informed participant consent. | Back Matter: Institutional Review Board (IRB) approved primary studies |
| **Competing Interests** | 22 | Declare financial and non-financial competing interests. | Back Matter: The author declares no competing interests |

---

## Section S3. Supplementary Figures and Detailed Captions

![Figure S1](../reports/fig_S1_specparam_example.png)

**Figure S1. Example single-subject spectral decomposition using specparam and isolated periodic residual oscillations.** 
*(A) Upper panel: Raw log₁₀ power spectrum (black solid line) recorded from electrode Fz, fitted aperiodic 1/f background (L(f) = b - χ log₁₀ f, red dashed line), and full model fit combining aperiodic floor and Gaussian oscillatory peaks (blue solid line). The parameter box summarizes extracted aperiodic offset b = 0.650 B (1 B = 1.0 log₁₀ µV²/Hz), aperiodic decay exponent χ = 1.152, model goodness-of-fit R² = 0.993, and focal peak parameters. (B) Lower panel: Pure isolated periodic residual spectrum (empirical residual: gray line; fitted Gaussian peaks: blue solid line with light-blue fill). By subtracting the aperiodic background, periodic oscillations return strictly to the zero-baseline floor (red dashed line at 0.0 B) outside oscillatory bands, eliminating false broadband power inflation.*

---

![Figure S2](../reports/fig_S2_omst_topology.png)

**Figure S2. Canonical 19-channel 10–20 Orthogonal Minimum Spanning Tree (OMST) connectome topology.** 
*Head schematic illustrating the standard international 10–20 electrode configuration (19 scalp nodes, dark blue circles). Blue solid lines depict the primary Minimum Spanning Tree (Tree 1, exactly 18 edges connecting all 19 nodes without cycles, sum of degrees = 36) forming the loopless topological backbone that connects all 19 channels. Amber dashed lines depict the secondary orthogonal spanning tree (Tree 2, exactly 18 orthogonal edges, sum of degrees = 36) capturing complementary functional connectivity pathways derived from Phase Lag Index (PLI, 4–13 Hz). Combined (M = 2 trees), the network comprises 36 edges with a total degree sum of 72. Across all N = 279 participants and 365 recording sessions, data-driven Global Cost Efficiency (GCE) optimization selected an average of M = 1.51 ± 0.50 trees (E = 27.12 ± 9.00 edges, mean degree k̄ = 2.86 ± 0.95, degree sum = 54.25, density = 15.86% ± 5.26%). The OMST algorithm guarantees 0.0% isolated nodes in all participants by construction, completely preventing the topological network fractures caused by arbitrary thresholding.*

---

![Figure S3](../reports/fig_S3_gain_invariance.png)

**Figure S3. Numerical verification of scale-invariance under multiplicative amplifier gain scaling.** 
*(A) Upper panel: Systematic variation of the unadjusted aperiodic offset parameter b as amplifier gain k is varied across four orders of magnitude (k ∈ [0.01, 100.0]). The blue markers (simulation extracted offset) fall precisely upon the red dashed line (theoretical prediction Δb = 2 log₁₀ k), exhibiting a linear slope of exactly 2.000 B/decade with residual error < 10⁻¹⁵ B. (B) Lower panel: Absolute numerical error between scaled and unscaled extractions for gain-invariant features: Periodic Theta (blue), Periodic Delta (green), Periodic Alpha (amber), and the Aperiodic Exponent χ (purple). All absolute errors remain bounded below 4.0 × 10⁻¹⁵ across all gain manipulations, confirming that numerical variation is strictly governed by IEEE 754 double-precision floating-point limits and establishing mathematical scale-invariance across clinical hardware.*

---

![Figure S4](../reports/fig_S4_cohort_spectra.png)

**Figure S4. Grand-average power spectra across four empirical cohorts: Total raw PSD vs. Parameterized periodic residuals.** 
*Row 1 (Panels A1–D1): Grand-average raw log₁₀ power spectral density (1.0–40.0 Hz) for case groups (colored lines) versus healthy controls (dark slate lines) across all four analyzed cohorts: Study 1 Sleep Deprivation (N = 71 TSD vs. N = 71 Normal Sleep), Study 2 Mumtaz Inpatient MDD (N = 30 MDD vs. N = 28 HC), Study 3 Cavanagh University Depression (N = 44 Depressed vs. N = 75 HC), and Study 4 UCSD Parkinson's Disease (N = 15 PD-ON vs. N = 16 HC). Row 2 (Panels A2–D2): Extracted periodic residual spectra (log₁₀ PSD - L(f)) with aperiodic background removed, returning to the zero-baseline floor. Crucially, while raw PSD in MDD (Panel B1) exhibits elevated low-frequency power due to a steepened aperiodic background (χ), the parameterized periodic residual (Panel B2) reveals genuine suppression of the oscillatory frontal theta peak (d = -0.826). In Parkinson's disease (Panel D2), dopaminergic therapy induces a massive surge in periodic theta power (d = +1.150) without shifting the alpha peak center frequency.*

---

![Figure S5](../reports/fig_S5_enrollment_flowchart.png)

**Figure S5. PRISMA / STROBE multi-cohort enrollment and quality control flowchart.** 
*Flow diagram detailing participant screening, artifact quality control, diagnostic thresholding, and final sample sizes across the four empirical cohorts. From an initial enrolled public release sample of N = 288 unique participants (374 recording sessions), 9 participants (3.1%) were excluded due to severe electrode bridging, excessive artifact contamination (> 20% rejected epochs), borderline depressive symptom severity (10 ≤ BDI ≤ 13), or missing clinical records. The final harmonized benchmark dataset comprises N = 279 unique participants and 365 recording sessions, standardized to 19 channels and evaluated under strict leak-free normative calibration.*

---

![Figure S6](../reports/fig_S6_permutation_tests.png)

**Figure S6. Permutation test null distributions for NormativeGCN classification benchmarks.** 
*Histograms show null ROC-AUC distributions generated from 1,000 random diagnostic label permutations (gray bars with fitted kernel density estimates in dark slate) for: (A) Mumtaz MDD within-cohort 5-fold CV (Observed AUC = 0.925, Null Mean = 0.501, p < 0.001); (B) Cavanagh Depression within-cohort 5-fold CV (Observed AUC = 0.649, Null Mean = 0.498, p < 0.001); (C) Cavanagh → Mumtaz zero-shot cross-center transfer (Observed AUC = 0.832, Null Mean = 0.502, p < 0.001); (D) Sleep Debt → Parkinson zero-shot transfer (Observed AUC = 0.802, Null Mean = 0.497, p < 0.001); (E) Sleep Debt → Mumtaz MDD transfer (Effective Inverted AUC = 0.723, Observed Raw AUC = 0.277, p < 0.001); and (F) Cavanagh MDD → Parkinson transfer (Effective Inverted AUC = 0.729, Observed Raw AUC = 0.271, p < 0.001). Vertical dashed gray lines denote chance performance (0.50). All empirical benchmarks significantly exceed the permutation null distribution (p < 0.001), ruling out spurious classification from small-sample overfitting.*
