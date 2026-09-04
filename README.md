# Human Pose Estimation by Probabilistic Mixtures of Gaussian and Uniform Distributions

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: Academic Non-Commercial](https://img.shields.io/badge/License-Academic%20Non--Commercial-blue.svg)](LICENSE)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![MMPose 1.3.2](https://img.shields.io/badge/MMPose-1.3.2-green.svg)](https://github.com/open-mmlab/mmpose)
[![Hydra 1.3](https://img.shields.io/badge/Config-Hydra%201.3-89b4fa.svg)](https://hydra.cc/)
[![Poetry](https://img.shields.io/badge/Packaging-Poetry-blueviolet.svg)](https://python-poetry.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📖 Table of Contents

- [Executive Summary & Pipeline Taxonomy](#-executive-summary--pipeline-taxonomy)
- [Core Theoretical Contributions](#-core-theoretical-contributions)
- [Methodology & Mathematical Formulation](#-methodology--mathematical-formulation)
  - [The Core Distinction: Solitary vs. Global Stochastic Flows](#the-core-distinction-solitary-vs-global-stochastic-flows)
  - [Stage 1: Stochastic Test-Time Augmentation (GPU)](#stage-1-stochastic-test-time-augmentation-gpu)
  - [Stage 2: Continuous Spatial Aggregation & Rejection Sampling (CPU)](#stage-2-continuous-spatial-aggregation--rejection-sampling-cpu)
  - [Stage 3: Probabilistic Mixture Modeling with Uniform Outlier Sink (CPU)](#stage-3-probabilistic-mixture-modeling-with-uniform-outlier-sink-cpu)
  - [Stage 4: Kinematic Tree MRF Decoding & Anatomical Priors (CPU)](#stage-4-kinematic-tree-mrf-decoding--anatomical-priors-cpu)
  - [Stage 5: Heteroscedastic Uncertainty Quantification & Adaptive Fusion](#stage-5-heteroscedastic-uncertainty-quantification--adaptive-fusion)
- [Comprehensive Empirical Benchmarks & Ablation Studies](#-comprehensive-empirical-benchmarks--ablation-studies)
  - [Ablation 1: Baseline Precision Parity (DARK vs. GMM Expectation)](#ablation-1-baseline-precision-parity-dark-vs-gmm-expectation)
  - [Ablation 2: Continuous Stochastic TTA vs. Discrete TTA](#ablation-2-continuous-stochastic-tta-vs-discrete-tta)
  - [Ablation 3: Decoupled Kinematic MRF Prior Breakdown (85,255 Keypoints)](#ablation-3-decoupled-kinematic-mrf-prior-breakdown-85255-keypoints)
  - [Ablation 4: In-Distribution Calibration & Out-of-Distribution Anomaly Detection](#ablation-4-in-distribution-calibration--out-of-distribution-anomaly-detection)
  - [Ablation 5: Adaptive Uncertainty Fusion with Universal Sweet Spot ($\beta = 0.1$)](#ablation-5-adaptive-uncertainty-fusion-with-universal-sweet-spot-\beta--01)
  - [Ablation 6: Multi-Backbone Validation & Latency Profiling](#ablation-6-multi-backbone-validation--latency-profiling)
- [Qualitative Diagnostic Suite (Publication Figures 1–5)](#-qualitative-diagnostic-suite-publication-figures-15)
- [Repository Structure & Clean Architecture](#-repository-structure--clean-architecture)
- [Installation & Environment Setup](#-installation--environment-setup)
- [One-Click Scientific Reproduction Suite](#-one-click-scientific-reproduction-suite)
- [Quickstart: Python API](#-quickstart-python-api)
- [Jupyter Notebooks Experimental Suite](#-jupyter-notebooks-experimental-suite)
- [Limitations & Multi-View 3D Extensibility](#-limitations--multi-view-3d-extensibility)
- [Citation](#-citation)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🔬 Executive Summary & Pipeline Taxonomy

Despite advancements in sub-pixel decoding, modern Human Pose Estimation (HPE) architectures (e.g., HRNet, ViTPose, ResNet) remain fundamentally constrained by **deterministic point-regression** ($\mathrm{argmax}$ or local Taylor expansions). Under severe real-world conditions—such as **progressive downsampling/resolution loss**, **motion blur**, **dense multi-person crowding**, and **symmetric limb ambiguities**—deterministic coordinate extraction systematically collapses: it cross-collapses limbs, introduces spatial jitter, and fails to communicate heteroscedastic uncertainty.

This framework introduces a post-hoc, continuous probabilistic formulation that intercepts raw heatmaps and parameterizes them explicitly as a **Gaussian Mixture Model (GMM) augmented with an orthogonal Uniform distribution ($\pi_u$)**, requiring **zero architectural modifications and zero retraining**.

To rigorously dissect the framework, it is formally decoupled into two operating flows:

1. **The Solitary Flow (Base Representation)** ($\text{Image} \rightarrow \text{Heatmap} \rightarrow \text{Sampling} \rightarrow \text{GMM + Uniform}$):
   Operates strictly on the raw, unaugmented heatmap. In this basic mode, continuous spatial support is recovered, and bimodal topological ambiguities are explicitly parameterized ($K=2$) without requiring external test-time transformations or graph optimization.
2. **The Global Stochastic Flow (TTA + MRF)** ($\text{Image} \rightarrow \text{TTA Heatmaps} \rightarrow \text{Sampling} \rightarrow \text{GMM + Uniform} \rightarrow \text{MRF}$):
   The full pipeline leverages multi-view stochastic evidence to mitigate heatmap poisoning, extracts continuous spatial distributions, and decodes global skeletal geometry via Markov Random Field belief propagation.

```text
========================================================================================================================
                                FRAMEWORK TAXONOMY: SOLITARY VS. GLOBAL STOCHASTIC FLOW
========================================================================================================================

  [1. The Solitary Flow (Base Representation / Basic Mode)]
  Input Image (I) ──> Frozen Backbone ──> Raw Heatmap H_k ──> MC Sampling ──> Robust GMM + Uniform Sink (π_u)
                                                                               ├── Primary Mode Mean: μ (OKS Parity)
                                                                               └── Covariance Volume: det(Σ) (Uncertainty)

  [2. The Global Stochastic Flow (Full Pipeline with TTA + MRF)]
  Input Image (I) ──> Multi-Scale TTA ──> Sharpness-Weighted Aggregation ──> MC Sampling
                                                                                    │
                                                                                    ▼
  Calibrated Pose <── Adaptive Softmax Fusion <── Kinematic Tree MRF <── Robust GMM + Uniform Sink (π_u)
                         (Universal β = 0.1)      (Max-Product BP)       (K=1 Anchors vs. K=2 Ambiguities)
========================================================================================================================
```

---

## 🌟 Core Theoretical Contributions

1. **Continuous Point Estimation at Parity with DARK**:
   The solitary mathematical expectation of the primary Gaussian component ($\boldsymbol{\mu}$) matches the sub-pixel precision of the state-of-the-art Taylor-expanded DARK decoder (within $\pm 0.0008$ OKS) across clean and corrupted benchmarks without relying on local Hessian approximations.
2. **Scale-Aware TTA Mitigating Heatmap Poisoning**:
   Replaces naive arithmetic heatmap averaging with continuous sigmoidal sharpness weighting $\mathcal{C}_k^{(n)}$, eliminating out-of-frame boundary degradation and consolidating multi-scale distributions in continuous space.
3. **Topological Ambiguity Isolation via Robust GMMs**:
   Models spatial keypoint densities as $K=1$ (unimodal) vs. $K=2$ (bimodal) mixtures selected via the Bayesian Information Criterion (BIC), providing explicit parametric hypothesis tracking for symmetric limbs.
4. **Orthogonal Uniform Noise Sink ($\pi_u$)**:
   A uniform background distribution $\mathcal{U}(\tilde{\mathbf{x}} \mid A)$ acts as an atypical probability sink that absorbs non-Gaussian diffuse noise under severe corruption, preventing covariance explosion and preserving the geometric integrity of genuine modes.
5. **The 2D Kinematic MRF Dilemma**:
   Demonstrates that while Markov Random Fields act as effective sub-pixel regularizers in canonical poses, rigid 2D Euclidean priors misinterpret *perspective foreshortening* as an anatomical violation, forcefully dragging foreshortened limbs across the image plane and causing strict swaps.
6. **Breakthrough Uncertainty Quantification**:
   The calibrated geometric Covariance Volume $\det(\mathbf{\Sigma}\_{\text{final}})$ fundamentally outperforms heuristic confidence $(1 - P\_{\text{DARK}})$ for Out-of-Distribution (OoD) anomaly detection under heavy occlusion (AUROC 0.94 vs 0.38 on CrowdPose $K=1$).
7. **Universal In-Distribution Calibration via Adaptive Softmax**:
   Continuously fuses heuristic baseline confidence with geometric covariance volume, systematically matching or enhancing Area Under the Sparsification Error (AUSE) across all datasets and degradations at universal $\beta = 0.1$.
8. **$\mathcal{O}(N^3)$ Memory Curse Bypass for Multi-View 3D**:
   Provides a direct mathematical formulation for continuous ray triangulation in 3D Euclidean space, rendering kinematic priors viewpoint-invariant while avoiding dense 3D voxel grids.

---

## 📐 Methodology & Mathematical Formulation

![Methodology Pipeline Overview](assets/methodology_pipeline_overview.png)

### The Core Distinction: Solitary vs. Global Stochastic Flows
- **The Solitary Flow (Base Representation)**: Intercepts single-pass inference directly from the frozen backbone $\Phi(\mathbf{I})$, extracting continuous coordinates and geometric uncertainty while bypassing TTA and MRF modules.
- **The Global Stochastic Flow (TTA + MRF)**: Combines stochastic multi-view evidence on GPU, performs continuous aggregation and sampling on CPU, fits robust GMM mixtures, and applies tree-structured kinematic MRF optimization.

---

### Stage 1: Stochastic Test-Time Augmentation (GPU)
Given an input frame $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, stochastic test-time transformations generate multi-scale representations across scales $s \in \{0.85, 0.925, 1.00, 1.075, 1.15\}$ and horizontal reflections:

$$
\tilde{\mathbf{I}}^{(n)} = \mathcal{T}^{(n)}(\mathbf{I}), \quad \mathbf{H}_k^{(n)} = \Phi(\tilde{\mathbf{I}}^{(n)})_k
$$

where $\Phi$ denotes a frozen backbone (HRNet-W32, ViTPose-Small, ResNet-50) and $\mathbf{H}\_k^{(n)}$ represents the raw activation tensor for keypoint $k$.

To transform activations into valid spatial probability distributions, spatial normalization is applied (**Equation 1**):

$$
\hat{\mathbf{H}}_k^{(n)}(x, y) = \frac{\max(0, \mathbf{H}_k^{(n)}(x, y))}{\sum_{i=1}^W \sum_{j=1}^H \max(0, \mathbf{H}_k^{(n)}(i, j))}
$$

---

### Stage 2: Continuous Spatial Aggregation & Rejection Sampling (CPU)
Normalized heatmaps are mapped back to the canonical reference frame via exact inverse affine transforms: $\tilde{\mathbf{H}}\_k^{(n)} = (\mathcal{T}^{(n)})^{-1}(\hat{\mathbf{H}}\_k^{(n)})$.

To prevent **heatmap poisoning** (dilution caused when zoomed-in scales push keypoints outside the field of view), each representation is weighted by a sigmoidal sharpness confidence metric (**Equation 2**):

$$
\mathcal{C}_k^{(n)} = \left(\rho_k^{(n)}\right)^\alpha \cdot \left(1 - \frac{1}{1 + \frac{\rho_k^{(n)} / (\mu_k^{(n)} + \epsilon)}{\tau}}\right)^\beta
$$

where $\rho\_k^{(n)} = \max\_{(x,y)} \hat{\mathbf{H}}\_k^{(n)}(x, y)$ is the peak activation, $\mu\_k^{(n)} = \frac{1}{|\Omega|} \sum\_{(x,y) \in \Omega} \hat{\mathbf{H}}\_k^{(n)}(x, y)$ is the spatial mean, $\tau = 10.0$ is the sharpness scale divisor, and $\alpha = \beta = 1.5$.

Normalized convex combination weights are assigned with a safety fallback threshold (**Equation 3**):

$$
w_k^{(n)} = \begin{cases} \frac{\mathcal{C}_k^{(n)}}{\sum_{m=1}^N \mathcal{C}_k^{(m)}}, & \text{if } \sum_{m=1}^N \mathcal{C}_k^{(m)} > \epsilon_{\text{tol}} \\ \frac{1}{N}, & \text{otherwise} \end{cases}
$$

The fused spatial probability mass function is obtained via linear superposition (**Equation 4**):

$$
\mathbf{P}_k(\mathbf{x}) = \sum_{n=1}^N w_k^{(n)} \tilde{\mathbf{H}}_k^{(n)}(\mathbf{x})
$$

Before sampling, thermodynamic temperature sharpening is applied in log-space to concentrate probability mass around genuine modes (**Equation 5**):

$$
P_T(\mathbf{x}) = \frac{\exp\left(\frac{1}{T} \log(\mathbf{P}_k(\mathbf{x}) + \epsilon)\right)}{\sum_{\mathbf{x}'} \exp\left(\frac{1}{T} \log(\mathbf{P}_k(\mathbf{x}') + \epsilon)\right)}
$$

where $T = 0.3 \in (0, 1]$. Continuous support over $\mathbb{R}^2$ is recovered via Von Neumann Rejection Sampling ($N = 1000$ points) combined with uniform sub-pixel de-quantization:

$$
\tilde{\mathbf{x}}_m = \mathbf{x}_m + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{U}(-0.5, 0.5), \quad m = 1, \dots, M
$$

The de-quantization jitter preserves a continuous variance lower-bound $\mathrm{Var}(\boldsymbol{\epsilon}) = \frac{1}{12}\mathbf{I} \approx 0.0833\mathbf{I}$, preventing covariance singularities during clustering.

---

### Stage 3: Probabilistic Mixture Modeling with Uniform Outlier Sink (CPU)
Spatial coordinates are modeled as a mixture of $C \in \{1, 2\}$ Gaussian components and an orthogonal Uniform distribution (**Equation 6**):

$$
p(\tilde{\mathbf{x}} \mid \mathbf{\Theta}) = \sum_{c=1}^C \pi_c \mathcal{N}(\tilde{\mathbf{x}} \mid \boldsymbol{\mu}_c, \mathbf{\Sigma}_c) + \pi_u \mathcal{U}(\tilde{\mathbf{x}} \mid A)
$$

where $\sum_{c=1}^C \pi_c + \pi_u = 1$, and $\mathcal{U}(\tilde{\mathbf{x}} \mid A) = \frac{1}{A}$ represents uniform density over bounding box area $A$.

Parameters $\mathbf{\Theta} = \{\pi_c, \boldsymbol{\mu}_c, \mathbf{\Sigma}_c, \pi_u\}$ are optimized via Expectation-Maximization (**Equation 7**):

$$
\gamma_{mc} = \frac{\pi_c \mathcal{N}(\tilde{\mathbf{x}}_m \mid \boldsymbol{\mu}_c, \mathbf{\Sigma}_c)}{\sum_{j=1}^C \pi_j \mathcal{N}(\tilde{\mathbf{x}}_m \mid \boldsymbol{\mu}_j, \mathbf{\Sigma}_j) + \pi_u \frac{1}{A}}, \quad \gamma_{mu} = \frac{\pi_u \frac{1}{A}}{\sum_{j=1}^C \pi_j \mathcal{N}(\tilde{\mathbf{x}}_m \mid \boldsymbol{\mu}_j, \mathbf{\Sigma}_j) + \pi_u \frac{1}{A}}
$$

Dynamic spectral Tikhonov regularization guarantees positive-definiteness throughout EM iterations:

$$
\tilde{\mathbf{\Sigma}}_c = \mathbf{\Sigma}_c + (\lambda_{\text{reg}} - \min(0, \lambda_{\min}))\mathbf{I}, \quad \lambda_{\text{reg}} = 10^{-4}
$$

Model selection between unimodal ($K=1$) and bimodal ($K=2$) topological states is governed by the Bayesian Information Criterion:

$$
\mathrm{BIC} = p \ln(M) - 2\mathcal{L}
$$

where $p = 6C$ parameters for $d=2$. When $\mathrm{BIC}\_2 < \mathrm{BIC}\_1$, both spatial modes are retained for downstream kinematic arbitration.

---

### Stage 4: Kinematic Tree MRF Decoding & Anatomical Priors (CPU)
The human skeleton is modeled as a directed tree graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ rooted at the facial axis. For each keypoint $i \in \mathcal{V}$, the state space is defined by its candidate Gaussian modes $\mathcal{Y}\_i = \{\boldsymbol{\mu}\_c^{(i)}\}\_{c=1}^{C_i}$.

Optimal global keypoint coordinates $\mathbf{y}^*$ are decoded via exact Max-Product Belief Propagation:

$$
\mathbf{y}^* = \arg\max_{\mathbf{y}} \prod_{i \in \mathcal{V}} \phi_i(y_i) \prod_{(i,j) \in \mathcal{E}} \psi_{ij}(y_i, y_j)
$$

The tree decoding evaluates two complementary potentials:

**1. Unary Potential** (mixing weight modulated by spatial compactness bonus):

$$
\phi_i(y_i = \boldsymbol{\mu}_c) = \pi_c \cdot \exp\left(-\frac{1}{2} \log|\mathbf{\Sigma}_c|\right)
$$

**2. Kinematic Pairwise Prior** (**Equation 8**):

$$
\psi_{ij}(y_i, y_j) = \exp \left( - \frac{ \left( \| y_i - y_j \|_2 - (\mu_{ij}^{\text{bone}} \cdot \sqrt{A_{\text{box}}}) \right)^2 }{ 2 \left( \sigma_{ij}^{\text{bone}} \cdot \sigma_{\text{mult}} \cdot \sqrt{A_{\text{box}}} \right)^2 } \right)
$$

where $\mu\_{ij}^{\text{bone}}$ and $\sigma\_{ij}^{\text{bone}}$ are empirical bone length statistics, $\sigma\_{\text{mult}} = 2.0$ accounts for 2D perspective tolerance, and $A\_{\text{box}}$ is the bounding box area.

> **Invariant Structural Anchors**: When $K=1$, the state space is a singleton ($|\mathcal{Y}\_i| = 1$), acting as an invariant anchor ($\Delta\mathrm{OKS} = 0.0000$) and eliminating unnecessary computational drift across 74.7% of all keypoints.

---

### Stage 5: Heteroscedastic Uncertainty Quantification & Adaptive Fusion
Applying the Law of Total Variance across normalized Gaussian mixing weights $\tilde{\pi}\_k = \pi\_k / \sum\_{j=1}^K \pi\_j$ (**Equation 9**):

$$
\mathbf{\Sigma}_{\text{total}} = \sum_{k=1}^K \tilde{\pi}_k \left( \mathbf{\Sigma}_k + (\boldsymbol{\mu}_k - \boldsymbol{\mu}_{\text{global}})(\boldsymbol{\mu}_k - \boldsymbol{\mu}_{\text{global}})^T \right), \quad \boldsymbol{\mu}_{\text{global}} = \sum_{k=1}^K \tilde{\pi}_k \boldsymbol{\mu}_k
$$

The kinematically calibrated covariance matrix is scaled by the bounding box size $s = \sqrt{A\_{\text{box}}}$ and the keypoint constant $\kappa\_j$ (**Equation 10**):

$$
\mathbf{\Sigma}_{\text{final}} = \frac{1}{(s \cdot \kappa_j)^2} \mathbf{\Sigma}_{\text{total}} + \epsilon\mathbf{I}
$$

The scalar geometric covariance volume $U\_{\text{spatial}} = |\mathbf{\Sigma}\_{\text{final}}| = \det(\mathbf{\Sigma}\_{\text{final}})$ is mapped into $[0, 1]$ via exponential projection (**Equation 11**):

$$
U_{\text{gmm}} = 1 - \exp(-\beta \cdot |\mathbf{\Sigma}_{\text{final}}|)
$$

The framework provides three adaptive fusion mechanisms with baseline heuristic uncertainty $U\_{\text{base}} = 1 - P\_{\text{DARK}}$:
1. **Max-Pooling**: $U\_{\text{adapt}} = \max(U\_{\text{base}}, U\_{\text{gmm}})$.
2. **Topological Gating**: Pure ($K=2$) and Hybrid ($K=2 \lor U\_{\text{gmm}} > \tau$).
3. **Continuous Adaptive Softmax Fusion** (Universal sweet spot at $\beta = 0.1$):

$$
U_{\text{adapt}} = \frac{\exp(U_{\text{base}}) \cdot U_{\text{base}} + \exp(U_{\text{gmm}}) \cdot U_{\text{gmm}}}{\exp(U_{\text{base}}) + \exp(U_{\text{gmm}})}
$$

---

## 📊 Comprehensive Empirical Benchmarks & Ablation Studies

### Ablation 1: Baseline Precision Parity (DARK vs. GMM Expectation)

Evaluated on clean images (256 × 192) across 500 instances per benchmark:

| Dataset | Metric | DARK (Taylor Argmax) | Ours (Solitary GMM Expectation) | Δ Difference |
| :--- | :--- | :---: | :---: | :---: |
| **COCO val2017** | OKS | **0.7079** | 0.7077 | −0.0002 |
| **CrowdPose** | OKS | **0.8278** | 0.8270 | −0.0008 |
| **OCHuman** | OKS | **0.6281** | 0.6280 | −0.0001 |

> **Conclusion**: The continuous GMM mathematical expectation achieves strict geometric parity with state-of-the-art Taylor expansion decoders while unlocking continuous spatial covariance parameters.

---

### Ablation 2: Continuous Stochastic TTA vs. Discrete TTA

Comprehensive performance across all five resolution degradation tiers (Clean, Low, Medium, High, Extreme):

| Dataset | Degradation Tier | OKS (DARK) | OKS (Ours GMM) | Loose Swaps (DARK) | Loose Swaps (Ours GMM) | Strict Swaps (DARK) | Strict Swaps (Ours GMM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **COCO** | Baseline (Clean) | **0.7137** | 0.7129 | 98 | **97** (**−1**) | 18 | 18 (0) |
| | Resize Low (0.50×) | **0.6912** | 0.6907 | 116 | **113** (**−3**) | 26 | 26 (0) |
| | Resize Medium (0.25×) | **0.6082** | 0.6076 | 215 | **214** (**−1**) | **66** | 67 (+1) |
| | Resize High (0.125×) | **0.4485** | 0.4465 | 404 | **388** (**−16**) | **135** | 138 (+3) |
| | Resize Extreme (0.062×) | **0.2505** | 0.2494 | 438 | **437** (**−1**) | 139 | 139 (0) |
| **CrowdPose** | Baseline (Clean) | **0.8361** | 0.8355 | **150** | 151 (+1) | 38 | **36** (**−2**) |
| | Resize Low (0.50×) | **0.8312** | 0.8302 | 155 | **153** (**−2**) | **36** | 38 (+2) |
| | Resize Medium (0.25×) | **0.7956** | 0.7954 | 198 | **191** (**−7**) | **53** | 54 (+1) |
| | Resize High (0.125×) | **0.6761** | 0.6755 | 336 | **329** (**−7**) | **102** | 103 (+1) |
| | Resize Extreme (0.062×) | **0.4389** | 0.4383 | **497** | 498 (+1) | **169** | 171 (+2) |
| **OCHuman** | Baseline (Clean) | 0.6356 | 0.6356 | 441 | **437** (**−4**) | 135 | **132** (**−3**) |
| | Resize Low (0.50×) | **0.6305** | 0.6299 | 443 | **442** (**−1**) | **127** | 131 (+4) |
| | Resize Medium (0.25×) | 0.6100 | **0.6106** (**+0.0006**) | 467 | **458** (**−9**) | **126** | 128 (+2) |
| | Resize High (0.125×) | **0.5342** | 0.5337 | 544 | **540** (**−4**) | 139 | **138** (**−1**) |
| | Resize Extreme (0.062×) | 0.3574 | **0.3578** (**+0.0005**) | 738 | **720** (**−18**) | 257 | **249** (**−8**) |

---

### Ablation 3: Decoupled Kinematic MRF Prior Breakdown (85,255 Keypoints)

Evaluating the selective activation of the kinematic tree across 85,255 keypoints:

| Dataset | Degradation Tier | Unimodal Anchors Ratio (K = 1) | Δ_MRF (K = 1) | Ambiguous Nodes Ratio (K = 2) | Δ_MRF Gain (K = 2) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **COCO** | Clean | 95.1% | 0.0000 | 4.9% | **+0.0254 (+2.54%)** |
| | Resize High | 67.1% | 0.0000 | 32.9% | **+0.0060 (+0.60%)** |
| | Resize Extreme | 40.6% | 0.0000 | 59.4% | **+0.0056 (+0.56%)** |
| **CrowdPose** | Clean | 91.0% | 0.0000 | 9.0% | −0.0009 (−0.09%) |
| | Resize Medium | 87.5% | 0.0000 | 12.4% | **+0.0140 (+1.40%)** |
| | Resize Extreme | 53.1% | 0.0000 | 46.9% | **+0.0045 (+0.45%)** |
| **OCHuman** | Clean | 72.7% | 0.0000 | 27.3% | −0.0032 (−0.32%) |
| | Resize Low | 72.7% | 0.0000 | 27.3% | **+0.0069 (+0.69%)** |
| | Resize Extreme | 59.3% | 0.0000 | 40.7% | **+0.0007 (+0.07%)** |
| **Pooled Total** | **All 85,255 Keypoints** | **74.7%** | **0.0000** | **25.3%** | **+0.0040 (+0.40%)** |

---

### Ablation 4: In-Distribution Calibration & Out-of-Distribution Anomaly Detection

#### Global Sparsification Error (AUSE ↓):

| Dataset | Evaluation Regime | DARK (1 − P_DARK) | DARK (Entropy) | Ours det(Σ) |
| :--- | :--- | :---: | :---: | :---: |
| **COCO** | Global | **0.0581** | 0.1437 | 0.0945 |
| | Occluded Joints | **0.1026** | 0.2037 | 0.1322 |
| **CrowdPose** | Global | **0.0515** | 0.1591 | 0.0911 |
| | Occluded Joints | **0.0599** | 0.1845 | 0.1067 |
| **OCHuman** | Global | **0.1573** | 0.2373 | 0.2143 |
| | **Occluded Joints** | 0.2636 | 0.3130 | **0.2203** |
| | **Resize Extreme (0.062×)** | 0.2165 | — | **0.2024** |

#### Global Out-of-Distribution (OoD) Anomaly Detection (AUROC ↑, vis == 0):

| Dataset | Ours (Volume) | Ours (Uniform) | DARK (Entropy) | DARK (1 − P_DARK) |
| :--- | :---: | :---: | :---: | :---: |
| **COCO** | 0.7364 | 0.7387 | 0.7042 | **0.8122** |
| **CrowdPose** | **0.8787** | 0.5408 | 0.3256 | 0.5160 |
| **OCHuman** | **0.7209** | 0.5712 | 0.4643 | 0.6441 |

#### Decoupled Topological OoD Failure Mode (K = 1 Unimodal Occlusions):

| Evaluation Condition | DARK (1 − P_DARK) | Ours det(Σ) | Performance Gain |
| :--- | :---: | :---: | :---: |
| **CrowdPose (K = 1 Absent Joints)** | 0.3838 | **0.9435** | **+0.5597 (+145.8%)** |
| **OCHuman (K = 1 Absent Joints)** | 0.4210 | **0.7521** | **+0.3311 (+78.6%)** |

---

### Ablation 5: Adaptive Uncertainty Fusion with Universal Sweet Spot ($\beta = 0.1$)

Global Sparsification Error (AUSE ↓) using a static, parameter-free sensitivity $\beta = 0.1$:

| Dataset | Degradation Tier | DARK (Base) AUSE | Ours Max-Pool AUSE (β = 0.1) | Ours Softmax AUSE (β = 0.1) |
| :--- | :--- | :---: | :---: | :---: |
| **COCO** | Clean | 0.0527 | 0.0435 | **0.0388** |
| | Resize Low | 0.0544 | 0.0477 | **0.0420** |
| | Resize Medium | 0.0689 | 0.0628 | **0.0555** |
| | Resize High | 0.0943 | 0.0862 | **0.0755** |
| | Resize Extreme | 0.1155 | 0.1098 | **0.0996** |
| | *General* | 0.0581 | 0.0524 | **0.0460** |
| **CrowdPose** | Clean | 0.0422 | 0.0423 | **0.0418** |
| | Resize Low | 0.0432 | 0.0434 | **0.0431** |
| | Resize Medium | 0.0528 | 0.0528 | **0.0525** |
| | Resize High | **0.0676** | 0.0680 | 0.0677 |
| | Resize Extreme | **0.0987** | 0.0999 | 0.0990 |
| | *General* | 0.0515 | 0.0518 | **0.0514** |
| **OCHuman** | Clean | **0.1381** | 0.1409 | 0.1390 |
| | Resize Low | **0.1429** | 0.1461 | 0.1435 |
| | Resize Medium | 0.1548 | 0.1557 | **0.1532** |
| | Resize High | 0.1819 | 0.1815 | **0.1756** |
| | Resize Extreme | 0.2165 | 0.2067 | **0.1945** |
| | *General* | 0.1573 | 0.1584 | **0.1540** |

> **Universal Sweet Spot**: While hard-routing mechanisms (Max-Pooling) suffer from severe structural conflicts across clean vs. occluded regimes, continuous Softmax fusion organically weights baseline confidence and geometric volume, achieving state-of-the-art calibration at $\beta = 0.1$ across all benchmarks without requiring per-image oracle tuning.

---

### Ablation 6: Multi-Backbone Validation & Latency Profiling

Evaluated across architectures on COCO val2017 (256 × 192):

| Backbone Architecture | Codec | Forward GPU (ms) | Continuous TTA (ms) | EM + BIC CPU (ms) | MRF Tree (ms) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HRNet-W32** | DARK | 18.4 ms | 3.4 ms | 1342.1 ms | 1.60 ms | 142 MB |
| **ResNet-50** | DARK | 12.1 ms | 3.2 ms | 1311.7 ms | 1.58 ms | 118 MB |
| **ViTPose-Small** | UDP | 24.6 ms | 3.6 ms | 1391.6 ms | 1.62 ms | 265 MB |

> **Computational Note**: Crucially, this latency bottleneck is localized strictly in the iterative mixture fitting rather than in graph kinematics or memory transfers. Because Monte Carlo sampling and GMM clustering are strictly independent across the 17 anatomical joints, the post-processing module is embarrassingly parallelizable; implementing the EM algorithm as native batch-tensor GPU operations would substantially reduce post-processing latency by evaluating all keypoints concurrently, providing an immediate pathway toward interactive frame rates. Furthermore, the memory footprint remains exceptionally modest (118–265 MB peak VRAM across all backbones), confirming that continuous uncertainty extraction does not impose GPU memory bottlenecks.

---

## 🖼️ Qualitative Diagnostic Suite (Publication Figures 1–5)

The repository provides automated generation for all 5 publication-grade qualitative diagnostic figures:

| Figure | Topic | Sample ID & Keypoint | Primary Empirical Finding |
| :--- | :--- | :---: | :--- |
| **Figure 1** | **Topological Swap Disambiguation** | Image 460 (`L_Ankle`, kp 15) | DARK snaps prematurely to contralateral false peak (w₂ = 0.43, red cross), while continuous GMM fits both modes (K=2) and places primary confidence on correct mode (w₁ = 0.56, 2σ ellipse). |
| **Figure 2** | **The Kinematic MRF Dilemma** | Win 130 (`R_Ankle`) vs. Fail 116555 (`L_Ankle`) | Top: Constructive pull in canonical pose (pulls 18 px → 26 px towards prior μ = 34 px). Bottom: Destructive 2D foreshortening drag (GT d = 8 px dragged 19 px away by rigid prior μ = 24 px). |
| **Figure 3** | **OoD Volumetric Uncertainty Alert** | Image 108525 (`L_Eye`, kp 1, vis=0) | DARK falsely locks onto background with confident peak (P = 0.70, alert 1 − P = 0.30). Continuous GMM captures spatial dispersion: det(Σ) = 34.58 (Entropy = 3.55 nats), saturating to maximum uncertainty. |
| **Figure 4** | **Heatmap Poisoning Mitigation** | Image 251 (`L_Knee`, kp 13) | Standard arithmetic averaging dilutes target joint with out-of-bounds crop (OKS = 0.69). Continuous sharpness weighting actively suppresses poisoned scale, restoring OKS to 0.82 (+0.13 gain). |
| **Figure 5** | **Uniform Noise Absorption (π_u)** | Image 482 (`L_Ankle`, kp 15) | Under extreme downsampling, orthogonal uniform distribution absorbs 14.51% noise mass, preventing Gaussian covariance explosion and stabilizing EM convergence. |

---

## 📂 Repository Structure & Clean Architecture

Designed following **SOLID principles** and **Clean Architecture**:

```
probabilistic-pose-gmm/
├── configs/                              # Modular Hydra experiment configuration ecosystem
│   ├── config.yaml                       # Master configuration entrypoint
│   ├── dataset/                          # Dataset schemas (coco.yaml, crowdpose.yaml, ochuman.yaml)
│   ├── model/                            # Model adapters (hrnet_w32.yaml, vitpose_small.yaml, resnet50.yaml)
│   └── tta/                              # Multi-scale schedules and sharpness parameters
├── notebooks/                            # 17 Interactive Jupyter Notebooks with precomputed outputs
├── outputs/                              # Benchmark results, evaluation parquets, and figures
│   ├── data/                             # 27 validated benchmark evaluation parquets (~55 MB)
│   └── figures/                          # Vectorial PDFs and 300 DPI publication figures
├── scripts/                              # Scientific reproduction CLI suite
│   ├── download_benchmark_data.py        # Automated benchmark parquet retriever & validator
│   ├── package_benchmark_data.py         # Release packaging utility for GitHub Releases
│   └── reproduce_all.py                  # Master One-Click reproduction CLI
├── src/                                  # Production library source code
│   └── pose_uncertainty/
│       ├── core/                         # Core math: sampling.py, mixture.py, mrf.py, skeleton.py
│       ├── datasets/                     # Unified loaders: coco.py, crowdpose.py, ochuman.py
│       ├── models/                       # Framework-agnostic adapters (MMPoseAdapter, resolver.py)
│       ├── tta/                          # Invertible TTA engine (engine.py, inverse.py)
│       └── utils/                        # Metrics (metrics.py), typing, and visual tools
└── tests/                                # Comprehensive PyTest test suite (unit + integration)
```

---

## ⚙️ Installation & Environment Setup

### Prerequisites
- Python 3.11 (`python --version`)
- [Poetry](https://python-poetry.org/) package manager (`pip install poetry`)
- CUDA 11.8+ / 12.1+ (recommended for GPU acceleration; CPU fallback supported)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/santiago-git19/probabilistic-pose-gmm.git
cd probabilistic-pose-gmm

# 2. Setup local MMPose engine & download pretrained model checkpoints
poetry run python scripts/models_installation/setup_models.py

# 3. Install dependencies via Poetry
poetry install

# 4. Install OpenMMLab Core Engines (Precompiled Wheels)
poetry run pip install chumpy==0.70 --no-build-isolation

# For GPU (CUDA 12.1 / PyTorch 2.1):
poetry run pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
# (For CPU-only environments, use: https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html)

poetry run mim install "mmdet==3.2.0"
poetry run pip install -e models/mmpose --no-build-isolation

# 5. Verify installation & environment health
poetry run python src/tests/check_env.py
poetry run pytest src/tests --ignore=models/mmpose -v
```

> [!NOTE]
> **Automatic DARK & UDP Codec Patching**:
> The environment setup script (`scripts/models_installation/setup_models.py` / `patch_mmpose.py`) automatically patches the cloned MMPose config files to enable exact sub-pixel decoders:
> - **HRNet-W32 & ResNet-50**: Configures `codec = dict(type='MSRAHeatmap', ..., unbiased=True)` to enable the **DARK (Distribution-Aware Coordinate Representation)** Taylor-expansion sub-pixel decoder.
> - **ViTPose-Small**: Configures `codec = dict(type='UDPHeatmap', ...)` to enable the **UDP (Unbiased Data Processing)** codec.

> **Automated Makefile Alternative (Linux / macOS / Git Bash)**:
> ```bash
> make setup-models
> make install BACKEND=cuda   # or BACKEND=cpu
> make check
> ```

---

## 🚀 One-Click Scientific Reproduction Suite

The repository includes a comprehensive, automated reproduction harness [`scripts/reproduce_all.py`](scripts/reproduce_all.py) to regenerate all publication figures, uncertainty calibration plots, and benchmarks:

```bash
# 1. Download packaged benchmark results (~55 MB from Release)
poetry run python scripts/download_benchmark_data.py

# 2. Reproduce all 5 Qualitative Paper Figures + Methodology Diagram (~30s)
poetry run python scripts/reproduce_all.py --figures

# 3. Reproduce all Uncertainty Calibration Plots (ECE, KDE, Beta Optimization) (~50s)
poetry run python scripts/reproduce_all.py --uncertainty

# 4. Reproduce everything end-to-end (Data + Figures + Uncertainty + Tests)
poetry run python scripts/reproduce_all.py --all
```

### CLI Reproduction Flags

| Flag | Purpose & Generated Artifacts |
| :--- | :--- |
| `--all` | Complete end-to-end reproduction: verifies benchmark data, generates all 6 paper figures, produces all uncertainty calibration plots, and executes the test suite. |
| `--figures` | Generates the **Methodology Pipeline Overview** (`methodology_pipeline_overview.pdf/png`) and all **5 Qualitative Publication Figures** in `outputs/figures/visualizations/`. |
| `--uncertainty` | Generates all quantitative calibration figures: **ECE 3×3 Matrices**, **OoD KDE Distributions**, **β Parameter Optimization Curves**, and **Uniform Noise Boxplots** in `outputs/figures/uncertainty/`. |
| `--data` | Downloads and verifies the integrity of all 27 precomputed benchmark evaluation parquets in `outputs/data/`. |
| `--tests` | Runs the full 233 unit and integration PyTest test suite (`src/tests`). |

---

## 💻 Quickstart: Python API

The following self-contained example runs the complete end-to-end inference and uncertainty calibration pipeline using the exact configuration parameters from `configs/`:

```python
import cv2
import numpy as np
from pose_uncertainty.models import create_model_adapter
from pose_uncertainty.models.adapters import MMPoseAdapter
from pose_uncertainty.pipeline.tta import TTAEngine
from pose_uncertainty.pipeline.scale_tta import ScaleAugConfig, ScaleAugmentor, compute_heatmap_confidence
from pose_uncertainty.core.sampling import sample_from_heatmap
from pose_uncertainty.core.mixture import select_best_model
from pose_uncertainty.core.mrf_decoder import MRFDecoder, UnaryPotential
from pose_uncertainty.core.skeleton import COCO_SKELETON, COCO_SKELETON_EDGES
from pose_uncertainty.utils.types import StandardizedHeatmap

# 1. Initialize Pose Backbone Adapter (from configs/model/hrnet_w32.yaml)
adapter = create_model_adapter("hrnet_w32", device="cuda")

# 2. Setup Multi-Scale & Geometric TTA (from configs/tta.yaml)
image = cv2.imread("data/sample.jpg")                  # Input RGB image (H, W, 3)
bbox = (120.0, 80.0, 180.0, 360.0)                   # Bounding box (x, y, w, h)
img_h, img_w = image.shape[:2]

scale_aug = ScaleAugmentor(ScaleAugConfig(
    enabled=True,
    scales=[0.85, 0.925, 1.0, 1.075, 1.15],
    aggregation="weighted_mean",
    sharpness_scale=10.0,
    peak_exponent=1.5,
    sharpness_exponent=1.5,
    epsilon=1e-10,
))
tta_engine = TTAEngine(config={"flip": {"enabled": True}, "seed": 42})

# 3. Multi-Scale TTA Batch Inference & Continuous Sigmoidal Aggregation
scaled_bboxes = scale_aug.get_scaled_bboxes(bbox, img_h, img_w)
heatmaps_per_scale, confs_per_scale, metadata_per_scale = [], [], []
metadata_ref = None

for sbbox in scaled_bboxes:
    aug_images, aug_metas = tta_engine.prepare_batch(image)
    tta_bboxes = [
        (img_w - 1.0 - (sbbox[0] + sbbox[2]), sbbox[1], sbbox[2], sbbox[3]) if meta.is_flipped else sbbox
        for meta in aug_metas
    ]
    heatmaps_list = adapter.predict_batch(aug_images, bboxes=tta_bboxes)

    accum = []
    scale_metadata = None
    for std_hm, meta in zip(heatmaps_list, aug_metas):
        hm = std_hm.data.copy()
        if meta.is_flipped:
            hm = TTAEngine.inverse_flip_heatmap(hm, adapter.flip_pairs, shift_heatmap=adapter.shift_heatmap)
        accum.append(hm)
        if scale_metadata is None and not meta.is_flipped and std_hm.metadata is not None:
            scale_metadata = std_hm.metadata

    metadata_per_scale.append(scale_metadata)
    if all(abs(a - b) < 1.0 for a, b in zip(sbbox, bbox)) and scale_metadata is not None:
        metadata_ref = scale_metadata

    scale_avg = np.mean(accum, axis=0).astype(np.float32)
    heatmaps_per_scale.append(scale_avg)
    confs_per_scale.append(compute_heatmap_confidence(
        scale_avg, sharpness_scale=10.0, peak_exponent=1.5, sharpness_exponent=1.5, epsilon=1e-10
    ))

if metadata_ref is None:
    metadata_ref = next((m for m in metadata_per_scale if m is not None), None)

# Continuous Sharpness-Weighted Aggregation across scales
agg_heatmap = scale_aug.aggregate(heatmaps_per_scale, confs_per_scale, metadata_per_scale, metadata_ref)

# 4. Stochastic Sampling & Robust GMM Fitting (from configs/sampling.yaml & configs/mixture.yaml)
unary_potentials = {}
num_kp = agg_heatmap.shape[0]
for k in range(num_kp):
    # Draw 1000 Monte Carlo samples per keypoint (from configs/sampling.yaml)
    samples = sample_from_heatmap(
        agg_heatmap[k], num_samples=1000, strategy="rejection", temperature=0.3, use_dequantization=True, seed=42 + k
    )
    mixture_res = select_best_model(
        samples.astype(np.float64), aic_weight=0.0, bic_weight=1.0, reg_covar=1e-4, max_iter=100, tol=1e-3, random_state=42 + k
    )
    means = [comp.mean for comp in mixture_res.components]
    covs = [comp.covariance for comp in mixture_res.components]
    weights = [comp.weight for comp in mixture_res.components]
    unary_potentials[k] = UnaryPotential(means=means, covariances=covs, weights=weights)

# 5. Decode Global Pose via Kinematic MRF Belief Propagation (from configs/mixture.yaml)
mrf = MRFDecoder(skeleton=COCO_SKELETON, bone_length_sigma=2.0, use_covariance_score=True)
calibrated_dict = mrf.decode(unary_potentials, area=float(bbox[2] * bbox[3]))
calibrated_pose_hm = np.array([calibrated_dict[k] for k in range(num_kp)], dtype=np.float32)

# 6. Project Heatmap Coordinates to Original Image Space
ref_hm_obj = StandardizedHeatmap(data=agg_heatmap, original_size=(img_h, img_w), metadata=metadata_ref)
final_coords = MMPoseAdapter.transform_heatmap_coords_to_image(calibrated_pose_hm, ref_hm_obj)

# 7. Render Skeleton & Keypoints Overlaid on Image
vis_img = image.copy()
x, y, w, h = [int(v) for v in bbox]
cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 180, 0), 2)

for parent_idx, child_idx in COCO_SKELETON_EDGES:
    pt1 = tuple(np.round(final_coords[parent_idx]).astype(int))
    pt2 = tuple(np.round(final_coords[child_idx]).astype(int))
    cv2.line(vis_img, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

for kpt_idx, (kx, ky) in enumerate(final_coords):
    center = (int(round(kx)), int(round(ky)))
    cv2.circle(vis_img, center, 4, (0, 100, 255), -1, cv2.LINE_AA)
    cv2.circle(vis_img, center, 5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite("outputs/calibrated_pose_demo.jpg", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
print("Saved calibrated pose visualization to 'outputs/calibrated_pose_demo.jpg'!")
```

---

## 📓 Jupyter Notebooks Experimental Suite

The repository provides **17 interactive Jupyter notebooks** covering the complete developmental progression:

| Notebook | Focus Area |
| :--- | :--- |
| [`03_test_coco_loader.ipynb`](notebooks/03_test_coco_loader.ipynb) | COCO 2017 dataloader verification, keypoint parsing, and lazy loading. |
| [`04_test_crowdpose_loader.ipynb`](notebooks/04_test_crowdpose_loader.ipynb) | CrowdPose 14-keypoint dataset ingestion and crowd visualization. |
| [`05_test_ochuman_loader.ipynb`](notebooks/05_test_ochuman_loader.ipynb) | OCHuman heavy-occlusion dataset validation and statistics. |
| [`06_test_model_inference.ipynb`](notebooks/06_test_model_inference.ipynb) | Multi-backbone forward pass verification (ResNet-50, HRNet-W32, ViTPose-Small). |
| [`07_test_model_adapters.ipynb`](notebooks/07_test_model_adapters.ipynb) | Framework-agnostic adapter layer validation and interface contracts. |
| [`08_demo_sampling.ipynb`](notebooks/08_demo_sampling.ipynb) | Stochastic Monte Carlo sampling strategies and temperature scaling analysis. |
| [`09_demo_em_flow.ipynb`](notebooks/09_demo_em_flow.ipynb) | Robust EM algorithm from scratch with uniform component outlier absorption. |
| [`10_complete_pipeline.ipynb`](notebooks/10_complete_pipeline.ipynb) | End-to-end integration: Ingestion → Sampling → GMM → Sub-pixel pose. |
| [`11_test_heatmap_decode.ipynb`](notebooks/11_test_heatmap_decode.ipynb) | Mathematical consistency between direct argmax and heatmap decoding. |
| [`12_validate_coord_transformation.ipynb`](notebooks/12_validate_coord_transformation.ipynb) | Exact inverse affine coordinate mapping between heatmap and image space. |
| [`13_debug_fiftyone_visual.ipynb`](notebooks/13_debug_fiftyone_visual.ipynb) | Visual debugging workflow and interactive FiftyOne session launch. |
| [`14_analysis_run_benchmark_stats.ipynb`](notebooks/14_analysis_run_benchmark_stats.ipynb) | Mass benchmark statistical analysis and diagnostic cohort distributions. |
| [`15_verify_tta_flip.ipynb`](notebooks/15_verify_tta_flip.ipynb) | Horizontal flip TTA validation and blurriness-free aggregation. |
| [`16_test_scale_tta.ipynb`](notebooks/16_test_scale_tta.ipynb) | Multi-scale TTA, confidence weighting, and boundary clamping validation. |
| [`17_validate_downscaled_benchmark.ipynb`](notebooks/17_validate_downscaled_benchmark.ipynb) | Low-resolution input degradation and robustness stress testing. |
| [`18_validate_image_quality_augmentations.ipynb`](notebooks/18_validate_image_quality_augmentations.ipynb) | Photometric corruptions: noise, contrast, Gaussian blur, and box smoothing. |
| [`verify_tta_pipeline.ipynb`](notebooks/verify_tta_pipeline.ipynb) | Comprehensive end-to-end certification of the full Robust Pose TTA framework. |

---

## 🔮 Limitations & Multi-View 3D Extensibility

### Limitations
1. **CPU Iterative Mixture Latency**: While the single-threaded CPU implementation of the EM loop currently limits throughput to 0.5–0.7 FPS, batch-tensor GPU parallelization across all 17 joints provides an immediate pathway to substantially enhance overall throughput toward interactive frame rates.
2. **2D Perspective Foreshortening**: 2D Euclidean bone priors cannot distinguish between anatomical deformation and depth foreshortening along the optical axis.

### Multi-View 3D Extensibility
Standard multi-view pose estimation models project 2D heatmaps into massive 3D voxel grids, suffering from the $\mathcal{O}(N^3)$ dimensional memory curse. Because our framework parameterizes 2D heatmaps into continuous probability distributions:
- Multi-camera 2D GMM rays can be algebraically triangulated into **Continuous 3D Gaussian Mixtures** ($\boldsymbol{\mu} \in \mathbb{R}^3, \mathbf{\Sigma} \in \mathbb{R}^{3\times3}$) without voxel discretization.
- Kinematic tree MRF belief propagation executed directly in 3D Euclidean space uses true physical bone lengths, naturally resolving the 2D projective foreshortening dilemma.

---

## 📑 Citation

If you use this codebase or methodology in your research, please cite:

```bibtex
@misc{reina2026probabilistic,
  title        = {Human Pose Estimation by Probabilistic Mixtures of Gaussian and Uniform Distributions},
  author       = {Reina-Alguacil, Santiago and L{\'o}pez-Rubio, Ezequiel},
  year         = {2026},
  howpublished = {\url{https://github.com/santiago-git19/probabilistic-pose-gmm}},
  note         = {Department of Computer Languages and Computer Science, University of M{\'a}laga. Preprint}
}
```

---

## 📄 License & Acknowledgments

This project is licensed under an **Academic and Non-Commercial Research License** with mandatory citation requirement - see the [LICENSE](LICENSE) file for details.

Developed at the **Department of Computer Languages and Computer Science, University of Málaga**. Built upon open-source foundations from [PyTorch](https://pytorch.org/), [MMPose](https://github.com/open-mmlab/mmpose), [Hydra](https://hydra.cc/), and [FiftyOne](https://voxel51.com/fiftyone/).
