# Experiment 6: Neural Network Oscillation Mitigation via Gradient Descent with Momentum

**Course:** Deep Learning (`CSA4701`)  
**Student Name:** Y. Varun Kumar Reddy  
**Register Number:** `192424230`  
**Academic Session:** MS2 - October – 2026  

---

## 1. Problem Scenario

> **Neural Network Optimization Scenario:**  
> A neural network experiences oscillations while optimizing its loss function. Implement gradient descent with momentum and compare its convergence with standard gradient descent.  
> **Theoretical Core:** Pathological Ravines, Condition Number $\kappa$, Analytical Backpropagation, Polyak Classical Momentum, and Hardware Tensor Vectorization.

---

## 2. Directory Structure & Architecture

```
Experiment - 6/
├── Experiment 6 Report.docx                           # Formal academic Word report (strictly 10 pages)
├── Experiment 6 Report.pdf                            # Compiled formal PDF report (verified exact 10 pages)
├── generate_report.py                                 # Automated Python report generator & Word COM pipeline
├── README.md                                          # Master documentation for Experiment 6
├── Data Collection & Preprocessing/                   # Module 1
│   ├── 01_data_collection_preprocessing.py            # Signal synthesis, pathological curvature & scaling
│   ├── raw_ill_conditioned_signals_1000.csv           # Raw synthesized sensor records (N = 1,000, D = 8)
│   ├── preprocessed_optimization_signals_1000.csv     # Standardized features (Z01-Z08) and scaled target
│   ├── signal_feature_metadata_1000.csv               # Feature statistics, units, and curvature significance
│   ├── 01_ill_conditioned_feature_distributions.png   # Figure 1: Density profiles & extreme scale disparities
│   ├── 02_feature_correlation_and_eigenvalue_spectrum.png # Figure 2: Pearson correlation & log-eigenvalues (kappa)
│   ├── 03_feature_standardization_and_outlier_bounds.png # Figure 3: Standardized boxplots & +/-3.0 sigma bounds
│   └── README.md                                      # Detailed technical documentation for Module 1
├── Data Splitting & Model Design/                     # Module 2
│   ├── 02_data_splitting_model_design.py              # Stratified split, MLP formulation & ravine simulation
│   ├── train_signals.csv                              # Stratified Training partition (70% - 700 records)
│   ├── val_signals.csv                                # Stratified Validation partition (15% - 150 records)
│   ├── test_signals.csv                               # Stratified Testing partition (15% - 150 records)
│   ├── trained_neural_network_weights.npz             # Serialized Xavier weights and biases (W1, b1, W2, b2)
│   ├── 04_dataset_split_proportions_and_response_invariance.png # Figure 4: Stratified split proportions & target KDEs
│   ├── 05_neural_network_architecture_and_forward_mapping.png # Figure 5: 8 -> 16 -> 1 MLP topology & forward activations
│   ├── 06_ill_conditioned_loss_surface_and_momentum_path.png # Figure 6: 2D contour & 3D ravine surface trajectories
│   └── README.md                                      # Detailed technical documentation for Module 2
└── Training & Evaluation/                             # Module 3
    ├── 03_training_and_evaluation.py                  # 150-epoch training, beta sweep, BLAS benchmark & KDEs
    ├── neural_optimizer_convergence_and_oscillations.csv # 150-epoch loss curves, gradient norms & oscillation indices
    ├── momentum_beta_sensitivity_analysis.csv         # Beta sweep results across beta in [0.0, 0.99]
    ├── computational_vectorization_benchmark.csv      # Pure-Python scalar loop vs. NumPy BLAS speedups
    ├── test_predictions_standard_vs_momentum.csv      # Sample-level test predictions and residual errors
    ├── 07_neural_loss_convergence_standard_vs_momentum.png # Figure 7: Multi-regime MSE loss decay & gradient norms
    ├── 08_momentum_beta_sensitivity_and_oscillation_metrics.png # Figure 8: Beta sensitivity U-curve & oscillation curves
    ├── 09_test_prediction_residuals_and_blas_speedup.png # Figure 9: Residual distributions & hardware speedup bar chart
    └── README.md                                      # Detailed technical documentation for Module 3
```

---

## 3. Key Quantitative Findings

### 3.1 Pathological Loss Landscape & Condition Number
- **Eigenvalue Spectrum:** The synthesized feature covariance matrix revealed an extreme condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{1692.57}{0.0279} \approx 60,772.7 \gg 1$.
- **Pathology:** This severe conditioning creates elongated, steep-walled anisotropic ravines where the gradient points almost orthogonally to the true minimizer along the transverse coordinate axes.

### 3.2 Optimization Dynamics & Oscillation Damping
- **Standard Gradient Descent (Low LR = 0.015):** Experiences monotonic but glacial longitudinal progress along the ravine floor, failing to reach the minimum within 150 epochs ($\text{Final Loss} = 0.1254$, $\text{Test MSE} = 0.1170$, $\text{Test } R^2 = 0.8818$).
- **Standard Gradient Descent (High LR = 0.045):** Diverges transversely between opposing canyon walls with severe cross-axis jitter ($\text{Test MSE} = 0.0874$).
- **Polyak Momentum ($\beta = 0.90$):** Eliminates transverse cross-axis oscillation via exponential cancellation ($\sum_{\tau=0}^t \beta^{t-\tau} (-1)^\tau \approx 0$) while compounding longitudinal velocity by $\frac{1}{1 - 0.9} = 10\times$, achieving superior convergence and generalization ($\text{Final Loss} = 0.0345$, $\text{Test MSE} = 0.0494$, $\text{Test } R^2 = 0.9501$ — a **57.8% reduction in test error**).
- **High Momentum ($\beta = 0.98$):** Accumulates excessive kinetic inertia, resulting in overshooting ringing around the global minimum ($\text{Test MSE} = 0.0804$, $\text{Test } R^2 = 0.9188$).

### 3.3 Momentum Hyperparameter Sensitivity Sweep
- $\beta = 0.00$ (Standard GD): $\text{Test MSE} = 0.1084$ | $\text{Speedup} = 1.0\times$
- $\beta = 0.60$: $\text{Test MSE} = 0.0853$ | $\text{Speedup} = 2.5\times$
- $\beta = 0.80$: $\text{Test MSE} = 0.0720$ | $\text{Speedup} = 5.0\times$
- $\beta = 0.90$ (Optimal): $\text{Test MSE} = 0.0598$ | $\text{Speedup} = 10.0\times$ (Apex stability & accuracy)
- $\beta = 0.99$ (Excessive): $\text{Test MSE} = 0.2152$ | $\text{Speedup} = 100.0\times$ (Severe overshooting)

### 3.4 Hardware Vectorization Acceleration (NumPy BLAS)
- Small batches ($N = 100$): BLAS vectorization executes in $0.025\text{ ms}$ vs. $3.929\text{ ms}$ in scalar loops (**$160.3\times$ hardware speedup**).
- Large batches ($N = 10,000$): BLAS executes in $26.2\text{ ms}$ vs. $2,334.0\text{ ms}$ in scalar loops (**$89.1\times$ speedup**).

---

## 4. Execution Instructions

```powershell
# 1. Execute Module 1: Data Collection & Preprocessing
python "Experiment - 6/Data Collection & Preprocessing/01_data_collection_preprocessing.py"

# 2. Execute Module 2: Data Splitting & Model Design
python "Experiment - 6/Data Splitting & Model Design/02_data_splitting_model_design.py"

# 3. Execute Module 3: Training & Evaluation
python "Experiment - 6/Training & Evaluation/03_training_and_evaluation.py"

# 4. Generate Formal 10-Page Word & PDF Reports
python "Experiment - 6/generate_report.py"
```
