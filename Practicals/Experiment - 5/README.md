# Experiment 5: Model Complexity Evaluation and Gradient Optimization (BGD, SGD, Momentum, Adam) in Real-Estate Valuation

**Course:** Deep Learning (`CSA4701`)  
**Student Name:** Y. Varun Kumar Reddy  
**Register Number:** `192424230`  
**Academic Session:** MS2 - October – 2026  

---

## 1. Problem Scenario

> **Real-Estate Valuation Scenario:**  
> A real-estate price prediction system compares linear regression, a decision tree, and a high-complexity polynomial model. Evaluate their training and testing errors to identify the most suitable model.  
> **Optimization Focus:** Gradient Descent, SGD, Momentum and Adam Comparison.

---

## 2. Directory Structure & Architecture

```
Experiment - 5/
├── Experiment 5 Report.docx                           # Formal academic Word report (strictly 10 pages)
├── Experiment 5 Report.pdf                            # Compiled formal PDF report (verified exact 10 pages)
├── generate_report.py                                 # Automated Python report generator & Word COM pipeline
├── README.md                                          # Master documentation for Experiment 5
├── Data Collection & Preprocessing/                   # Module 1
│   ├── 01_data_collection_preprocessing.py            # Feature synthesis, Z-score scaling & EDA
│   ├── raw_real_estate_dataset_1000.csv               # Raw synthesized housing records (N = 1,000, D = 8)
│   ├── preprocessed_real_estate_1000.csv             # Standardized feature matrix and kUSD targets
│   ├── real_estate_feature_metadata_1000.csv          # Feature statistics and domain descriptions
│   ├── 01_housing_feature_distributions_and_target.png # Figure 1: Price histogram & SqFt vs Price scatter
│   ├── 02_real_estate_feature_correlation_matrix.png  # Figure 2: Pearson correlation matrix & ranked drivers
│   ├── 03_feature_standardization_and_outliers.png   # Figure 3: Normalization KDEs & boxplot outlier bounds
│   └── README.md                                      # Detailed technical documentation for Module 1
├── Data Splitting & Model Design/                     # Module 2
│   ├── 02_data_splitting_model_design.py              # Stratified split, model formulation & optimizers
│   ├── train_real_estate.csv                          # Stratified Training partition (70% - 700 records)
│   ├── val_real_estate.csv                            # Stratified Validation partition (15% - 150 records)
│   ├── test_real_estate.csv                           # Stratified Testing partition (15% - 150 records)
│   ├── trained_regression_models.npz                  # Serialized weights for linear and polynomial models
│   ├── 04_dataset_split_proportions_and_price_invariance.png # Figure 4: Stratified partition proportions & density
│   ├── 05_model_architecture_predictions_vs_actual.png # Figure 5: Parity plots for Linear, Tree, and Poly
│   ├── 06_loss_surface_and_optimizer_trajectories.png # Figure 6: 2D loss contour & BGD, SGD, Mom, Adam paths
│   └── README.md                                      # Detailed technical documentation for Module 2
└── Training & Evaluation/                             # Module 3
    ├── 03_training_and_evaluation.py                  # Complexity benchmarking, optimizer dynamics & BLAS speedup
    ├── model_complexity_error_comparison.csv          # Train vs Test MSE, RMSE, MAE, R2 across 10 architectures
    ├── optimizer_convergence_and_oscillations.csv     # 100-epoch optimizer loss trajectories & oscillation scores
    ├── computational_vectorization_benchmark.csv      # Scalar Loop vs. Vectorized NumPy BLAS speedups
    ├── test_set_predictions_all_models.csv            # Test set predictions, actual prices & model residuals
    ├── 07_training_vs_testing_error_and_overfitting.png # Figure 7: Train vs Test error curves (overfitting gap)
    ├── 08_optimizer_convergence_loss_curves.png       # Figure 8: Optimizer convergence curves & oscillation bar chart
    ├── 09_residual_distributions_and_speedup_benchmark.png # Figure 9: Residual KDEs & BLAS speedup bar chart
    └── README.md                                      # Detailed technical documentation for Module 3
```

---

## 3. Key Quantitative Findings

### 3.1 Model Complexity & Overfitting Benchmark
- **Linear Regression (Degree 1, 9 weights):** Underfits non-linear diminishing returns ($\text{Test MSE} = 894.2$, $\text{Test } R^2 = 0.9869$).
- **Degree 2 Polynomial (45 weights):** Optimal polynomial model capturing quadratic features ($\text{Test MSE} = 692.6$, $\text{Test } R^2 = 0.9899$).
- **Degree 4 Polynomial (495 weights):** Catastrophic overfitting; training error collapses to $115.7$, but test error explodes to $90,679.6$ ($\text{Test } R^2 = -0.3288$).
- **Decision Tree (Depth = 6, 64 leaves):** Optimal overall generalization; captures piecewise interaction thresholds cleanly ($\text{Test MSE} = 5,313.0$, $\text{Test } R^2 = 0.9221$).

### 3.2 Optimization Dynamics & Oscillation Damping
- **Batch Gradient Descent (BGD):** Smooth, monotonic loss reduction, zero directional oscillation ($0.0008$), but high computation per epoch.
- **Stochastic Gradient Descent (SGD):** High stochastic noise and cross-axis jitter (Oscillation Index $= 0.9823$).
- **Momentum Gradient Descent ($\beta = 0.9$):** Exponential velocity accumulation dampens SGD oscillations by **$76.4\%$** (Oscillation Index $= 0.2316$), accelerating convergence through ravines.
- **Adam:** Combines momentum and adaptive coordinate scaling, achieving the fastest parameter convergence.
- **BLAS Vectorization:** Accelerates gradient calculations by up to **$987.9\times$** over scalar loops.

---

## 4. Execution Instructions

```powershell
# 1. Execute Module 1: Data Collection & Preprocessing
python "Experiment - 5/Data Collection & Preprocessing/01_data_collection_preprocessing.py"

# 2. Execute Module 2: Data Splitting & Model Design
python "Experiment - 5/Data Splitting & Model Design/02_data_splitting_model_design.py"

# 3. Execute Module 3: Training & Evaluation
python "Experiment - 5/Training & Evaluation/03_training_and_evaluation.py"

# 4. Generate Formal 10-Page Word & PDF Reports
python "Experiment - 5/generate_report.py"
```
