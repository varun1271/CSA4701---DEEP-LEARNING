# Module 3: Training & Evaluation

## 1. Module Overview
Module 3 performs rigorous quantitative model comparison across model complexity tiers (Polynomial Degrees 1 to 4 and Decision Tree Depths 2 to 12), benchmarks optimization convergence and directional gradient oscillations (BGD, SGD, Momentum, Adam), and measures computational hardware speedups achieved via vectorized BLAS matrix operations.

---

## 2. Model Complexity & Generalization Benchmarking

The experimental results definitively demonstrate the classical bias-variance tradeoff:

| Model Architecture | Parameter Count | Training MSE ($k\text{USD}^2$) | Testing MSE ($k\text{USD}^2$) | Test $R^2$ | Generalization Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Deg 1)** | $9$ | $7,461.2$ | $8,024.5$ | $0.8652$ | **Underfitting (High Bias):** Inadequate capacity to model curvature. |
| **Polynomial Reg (Deg 2)** | $45$ | $1,842.1$ | $2,185.4$ | $0.9632$ | **Optimal Polynomial:** Captures quadratic diminishing returns. |
| **Polynomial Reg (Deg 3)** | $165$ | $482.3$ | $14,920.8$ | $0.7488$ | **Severe Overfitting:** Memorizes training fluctuations. |
| **Polynomial Reg (Deg 4)** | $495$ | $112.5$ | $185,420.1$ | $-2.1240$ | **Catastrophic Overfitting:** Test error explodes uncontrollably. |
| **Decision Tree (Depth 2)** | $4$ leaves | $18,240.5$ | $19,510.2$ | $0.6715$ | **Severe Underfitting:** Coarse regional averages. |
| **Decision Tree (Depth 6)** | $64$ leaves | $1,215.4$ | **$2,042.8$** | **$0.9656$** | **Optimal Architecture:** Best overall test generalization. |
| **Decision Tree (Unlimited)** | $420$ leaves | $0.00$ | $4,850.2$ | $0.9183$ | **Overfitting:** Zero training error, but degraded test $R^2$. |

---

## 3. Optimization Algorithms & Oscillation Analysis

1. **Batch Gradient Descent (BGD):**
   - Convergence: Smooth, strictly monotonic loss reduction.
   - Limitation: Requires processing the entire dataset ($N = 700$) per iteration; slow wall-clock latency per epoch.
2. **Stochastic Gradient Descent (SGD):**
   - Convergence: Highly erratic and noisy.
   - Pathology: High gradient variance causes severe directional oscillations (Mean Oscillation Metric $= 0.485$), requiring learning rate decay to prevent divergence.
3. **Momentum Gradient Descent ($\beta = 0.9$):**
   - Convergence: Rapid progress along ravines.
   - Damping: Accumulates velocity $v_t$, averaging out orthogonal cross-axis gradients and reducing oscillation severity to $0.182$ ($62.5\%$ reduction).
4. **Adaptive Moment Estimation (Adam):**
   - Convergence: Fastest parameter convergence velocity.
   - Invariance: Individual coordinate step scaling renders Adam invariant to feature scale disparities, reaching optimal loss within 30 epochs.

---

## 4. Computational Hardware Vectorization Benchmark

- **Scalar Loop Execution:** Nested elementwise loops suffer severe Python interpreter overhead and cache thrashing.
- **Vectorized NumPy BLAS:** Leverages SIMD AVX2 tensor primitives and contiguous memory caching, delivering up to **$28.5\times$** execution acceleration.

---

## 5. Artifacts and Generated Data Files

- **Complexity Metrics:** [model_complexity_error_comparison.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Training%20&%20Evaluation/model_complexity_error_comparison.csv)
- **Optimizer Trajectories:** [optimizer_convergence_and_oscillations.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Training%20&%20Evaluation/optimizer_convergence_and_oscillations.csv)
- **Speedup Benchmark:** [computational_vectorization_benchmark.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Training%20&%20Evaluation/computational_vectorization_benchmark.csv)
- **Test Predictions:** [test_set_predictions_all_models.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Training%20&%20Evaluation/test_set_predictions_all_models.csv)
- **Visualizations:**
  - [07_training_vs_testing_error_and_overfitting.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Training%20&%20Evaluation/07_training_vs_testing_error_and_overfitting.png)
  - [08_optimizer_convergence_loss_curves.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Training%20&%20Evaluation/08_optimizer_convergence_loss_curves.png)
  - [09_residual_distributions_and_speedup_benchmark.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Training%20&%20Evaluation/09_residual_distributions_and_speedup_benchmark.png)
