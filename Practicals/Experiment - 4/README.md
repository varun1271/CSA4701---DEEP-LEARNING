# Experiment 4: Categorical Cross-Entropy, Softmax Probabilities, and Bias-Variance Tradeoff in Deep Image Classification

**Course:** Deep Learning (`CSA4701`)  
**Student Name:** Y. Varun Kumar Reddy  
**Register Number:** `192424230`  
**Academic Session:** MS2 - October – 2026  

---

## 1. Problem Scenario

> **Manufacturing & Vision System Scenario:**  
> An image-classification model predicts probabilities for multiple object categories. Calculate categorical cross-entropy between the true class labels and predicted probabilities.  
> **Core Mathematical & Learning Dynamics Focus:** Bias–Variance, Overfitting and Underfitting Analysis.

---

## 2. Directory Structure & Architecture

```
Experiment - 4/
├── Experiment 4 Report.docx                           # Formal academic Word report (strictly 10 pages)
├── Experiment 4 Report.pdf                            # Compiled formal PDF report (verified exact 10 pages)
├── generate_report.py                                 # Automated Python report generator & Word COM pipeline
├── README.md                                          # Master documentation for Experiment 4
├── Data Collection & Preprocessing/                   # Module 1
│   ├── 01_data_collection_preprocessing.py            # Feature synthesis, Z-score scaling & EDA
│   ├── raw_image_classification_features_1000.csv     # Raw synthesized image feature matrix (N = 1,000, C = 5)
│   ├── preprocessed_image_features_1000.csv           # Z-score standardized image feature matrix
│   ├── class_metadata_and_labels_1000.csv             # Statistical profile and metadata per class
│   ├── 01_class_distribution_and_feature_embeddings.png # Figure 1: Class balance & 2D PCA latent projection
│   ├── 02_multiclass_feature_separability_and_distributions.png # Figure 2: Violin plots & correlation matrix
│   ├── 03_image_feature_standardization_and_outliers.png # Figure 3: Standardization densities & boxplots
│   └── README.md                                      # Detailed technical documentation for Module 1
├── Data Splitting & Model Design/                     # Module 2
│   ├── 02_data_splitting_model_design.py              # Stratified split, Stable Softmax, CCE & architectures
│   ├── train_image_features.csv                       # Stratified Training partition (70% - 700 samples)
│   ├── val_image_features.csv                         # Stratified Validation partition (15% - 150 samples)
│   ├── test_image_features.csv                        # Stratified Test partition (15% - 150 samples)
│   ├── trained_classifier_architectures.npz           # Serialized neural weights across capacity regimes
│   ├── 04_stratified_dataset_partitions_and_balance.png # Figure 4: Stratified partition radar chart
│   ├── 05_softmax_probabilities_and_temperature_scaling.png # Figure 5: Softmax calibration & entropy vs. T
│   ├── 06_cross_entropy_loss_surface_and_gradients.png # Figure 6: Convex CCE loss surface & gradient vectors
│   └── README.md                                      # Detailed technical documentation for Module 2
└── Training & Evaluation/                             # Module 3
    ├── 03_training_and_evaluation.py                  # Training dynamics, Bootstrap Bias-Variance & BLAS speedup
    ├── model_complexity_and_generalization_gap.csv     # 100-epoch training & validation trajectories
    ├── bias_variance_decomposition_analysis.csv       # Bootstrap decomposition across capacities (H in [0, 256])
    ├── test_predictions_and_cross_entropy_breakdown.csv # Test partition predictions, probabilities & sample losses
    ├── computational_vectorization_benchmark.csv      # Scalar Loop vs. Vectorized NumPy BLAS latency & speedups
    ├── 07_learning_curves_underfitting_vs_overfitting.png # Figure 7: Loss learning curves & accuracy trajectories
    ├── 08_bias_variance_decomposition_curve.png       # Figure 8: Classic U-shaped Bias-Variance curves
    ├── 09_test_confusion_matrix_and_computational_speedup.png # Figure 9: Confusion matrix & speedup bar chart
    └── README.md                                      # Detailed technical documentation for Module 3
```

---

## 3. Mathematical Foundations

### 3.1 Softmax Mapping with Temperature Scaling
Given continuous feature representation $\mathbf{x}_i \in \mathbb{R}^D$ and model weights $\mathbf{W} \in \mathbb{R}^{C \times D}$, bias $\mathbf{b} \in \mathbb{R}^C$, unnormalized logits are computed as:
$$\mathbf{z}_i = \mathbf{W}\mathbf{x}_i + \mathbf{b} \in \mathbb{R}^C$$
The temperature-scaled Softmax activation function maps logits to normalized class probabilities:
$$P(\hat{Y}_i = c \mid \mathbf{x}_i, T) = \hat{y}_{i,c} = \frac{\exp(z_{i,c} / T)}{\sum_{j=0}^{C-1} \exp(z_{i,j} / T)}$$
- **Numerical Overflow Protection:** Implemented using the Log-Sum-Exp identity: $m = \max_k z_k$, $\ln \sum_j \exp(z_j) = m + \ln \sum_j \exp(z_j - m)$.

### 3.2 Categorical Cross-Entropy (CCE) & Analytic Gradients
For one-hot target vectors $\mathbf{y}_i \in \{0, 1\}^C$ where $\sum_{c=0}^{C-1} y_{i,c} = 1$:
$$\mathcal{L}_{CCE} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^{C-1} y_{i,c} \ln(\hat{y}_{i,c} + \epsilon) = -\frac{1}{N} \sum_{i=1}^N \ln(\hat{y}_{i, c^*_i} + \epsilon)$$
The analytic gradient with respect to logit $z_{i,c}$ evaluates to:
$$\frac{\partial \mathcal{L}_{CCE}}{\partial z_{i,c}} = \hat{y}_{i,c} - y_{i,c} \implies \nabla_{\mathbf{Z}} \mathcal{L}_{CCE} = \frac{1}{N} (\hat{\mathbf{Y}} - \mathbf{Y})$$

### 3.3 Bias-Variance Decomposition Theorem
For ensemble bootstrap expectations $\bar{\mathbf{f}}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[\hat{\mathbf{f}}(\mathbf{x}; \mathcal{D})]$:
$$\mathbb{E}_{\mathcal{D}} \left[ \|\mathbf{y} - \hat{\mathbf{f}}(\mathbf{x}; \mathcal{D})\|_2^2 \right] = \text{Bias}^2[\hat{\mathbf{f}}(\mathbf{x})] + \text{Variance}[\hat{\mathbf{f}}(\mathbf{x})] + \sigma_{\text{noise}}^2$$
- **Squared Bias:** $\text{Bias}^2 = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \|\bar{\mathbf{f}}(\mathbf{x}_i) - \mathbf{y}_i\|_2^2$ (Underfitting error)
- **Model Variance:** $\text{Variance} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \frac{1}{B} \sum_{b=1}^B \|\hat{\mathbf{f}}_b(\mathbf{x}_i) - \bar{\mathbf{f}}(\mathbf{x}_i)\|_2^2$ (Sensitivity to training perturbations)
- **Irreducible Noise:** $\sigma_{\text{noise}}^2 = 0.0450$ (Bayes error boundary floor)

---

## 4. Key Quantitative Results

### 4.1 Empirical Bias-Variance Decomposition Summary
| Capacity ($H$) | Architecture | Squared Bias | Model Variance | Noise Floor | Total Generalization Error | Learning Regime |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **$H = 0$** | Linear Softmax | $0.6854$ | $0.0942$ | $0.0450$ | $0.8246$ | Severe Underfitting (High Bias) |
| **$H = 4$** | MLP ($8 \to 4 \to 5$) | $0.6711$ | $0.0372$ | $0.0450$ | $0.7533$ | Underfitting (Inadequate Capacity) |
| **$H = 8$** | MLP ($8 \to 8 \to 5$) | $0.5847$ | $0.0466$ | $0.0450$ | $0.6763$ | Suboptimal Capacity |
| **$H = 16$** | MLP ($8 \to 16 \to 5$) | $0.5308$ | $0.0381$ | $0.0450$ | $0.6139$ | Transition Boundary |
| **$H = 32$** | MLP ($8 \to 32 \to 5$) | $0.3688$ | $0.0218$ | $0.0450$ | $0.4356$ | Optimal Tradeoff (Sweet Spot) |
| **$H = 64$** | MLP ($8 \to 64 \to 5$) | $0.2370$ | $0.0294$ | $0.0450$ | $0.3114$ | Balanced Generalization |
| **$H = 128$** | MLP ($8 \to 128 \to 5$) | $0.1916$ | $0.0635$ | $0.0450$ | **$0.3001$** | **Global Minimum of Expected Error** |
| **$H = 256$** | MLP ($8 \to 256 \to 5$) | $0.1904$ | $0.1388$ | $0.0450$ | $0.3743$ | Overfitting (Variance Explosion) |

### 4.2 Computational Hardware Speedup (Scalar vs. BLAS)
| Batch Size ($N$) | Evaluation Elements ($N \times C$) | Scalar Python Loop | Vectorized NumPy BLAS | Speedup Factor |
| :---: | :---: | :---: | :---: | :---: |
| $N = 10$ | $50$ | $0.222\text{ ms}$ | $0.164\text{ ms}$ | $1.4\times$ |
| $N = 50$ | $250$ | $1.041\text{ ms}$ | $0.191\text{ ms}$ | $5.4\times$ |
| $N = 100$ | $500$ | $2.061\text{ ms}$ | $0.237\text{ ms}$ | $8.7\times$ |
| $N = 500$ | $2,500$ | $10.576\text{ ms}$ | $0.541\text{ ms}$ | $19.5\times$ |
| $N = 1,000$ | $5,000$ | $20.768\text{ ms}$ | $0.889\text{ ms}$ | $23.4\times$ |
| $N = 5,000$ | $25,000$ | $104.759\text{ ms}$ | $3.448\text{ ms}$ | **$30.4\times$ (Peak)** |
| $N = 10,000$ | $50,000$ | $175.470\text{ ms}$ | $6.825\text{ ms}$ | $25.7\times$ |

---

## 5. Execution Instructions

To reproduce all datasets, figures, and compiled formal reports:

```powershell
# 1. Execute Module 1: Data Collection & Preprocessing
python "Experiment - 4/Data Collection & Preprocessing/01_data_collection_preprocessing.py"

# 2. Execute Module 2: Data Splitting & Model Design
python "Experiment - 4/Data Splitting & Model Design/02_data_splitting_model_design.py"

# 3. Execute Module 3: Training & Evaluation
python "Experiment - 4/Training & Evaluation/03_training_and_evaluation.py"

# 4. Generate Formal 10-Page Word & PDF Reports
python "Experiment - 4/generate_report.py"
```

---

## 6. Verification and Compliance

- **Course:** `Deep Learning` (`CSA4701`)
- **Student:** `Y. Varun Kumar Reddy` (`192424230`)
- **Format:** Strictly matching the academic styling and visual layout of `Experiment - 1`, `Experiment - 2`, and `Experiment - 3`.
- **Report Verification:** `Experiment 4 Report.pdf` is verified at **exactly 10 pages** with zero overflow or orphan sections.
