# Module 3: Training & Evaluation

## 1. Module Overview
Module 3 implements full epoch-by-epoch training dynamics, rigorous empirical Bias-Variance decomposition across neural model capacities, detailed test set performance evaluation across 5 object categories, and computational vectorization benchmarks comparing scalar elementwise loops against optimized NumPy BLAS tensor routines.

---

## 2. Epoch-by-Epoch Training Dynamics Across Regimes

The script tracks training and validation loss ($\mathcal{L}_{CCE}$) and classification accuracy across 100 epochs for three architectural capacities:

1. **Underfitting Regime (High Bias, Low Capacity):**
   - Training Loss: Remains stubbornly high ($\approx 1.25\text{ nats}$).
   - Validation Loss: Follows closely ($\approx 1.30\text{ nats}$), showing negligible generalization gap ($\Delta \mathcal{L} \approx 0.05$), but poor final accuracy ($\approx 52.0\%$).
   - Pathology: Incapable of learning the non-linear feature interactions separating complex visual classes.
2. **Optimal Regulated Regime (Balanced Bias-Variance):**
   - Training Loss: Smoothly converges to $\approx 0.28\text{ nats}$.
   - Validation Loss: Stabilizes at $\approx 0.35\text{ nats}$ with a tight generalization gap ($\Delta \mathcal{L} \approx 0.07$).
   - Generalization: Attains peak test accuracy ($\approx 88.7\%$) without overfitting.
3. **Overfitting Regime (Low Bias, High Variance):**
   - Training Loss: Drops toward zero ($\approx 0.02\text{ nats}$), memorizing individual training sample quirks.
   - Validation Loss: Reaches a minimum around epoch 20, then explodes monotonically beyond $1.65\text{ nats}$.
   - Generalization Gap: Expands to $\Delta \mathcal{L} = 1.62\text{ nats}$, with validation accuracy stagnating around $67\%$.

---

## 3. Mathematical Foundations: Empirical Bias-Variance Decomposition

For multiclass probability predictions, the expected mean squared prediction error decomposes into:
$$\mathbb{E}_{\mathcal{D}} \left[ \|\mathbf{y} - \hat{\mathbf{f}}(\mathbf{x})\|_2^2 \right] = \text{Bias}^2[\hat{\mathbf{f}}(\mathbf{x})] + \text{Variance}[\hat{\mathbf{f}}(\mathbf{x})] + \sigma^2$$
where $\mathcal{D}$ denotes training sets sampled via bootstrap resampling ($B = 25$ runs):

### 3.1 Ensemble Expected Prediction
$$\bar{\mathbf{f}}(\mathbf{x}_i) = \frac{1}{B} \sum_{b=1}^B \hat{\mathbf{y}}_i^{(b)}$$

### 3.2 Squared Bias
Measures how far the average prediction across training sets deviates from the true target vector:
$$\text{Bias}^2 = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \|\bar{\mathbf{f}}(\mathbf{x}_i) - \mathbf{y}_i\|_2^2$$

### 3.3 Model Variance
Measures the sensitivity of the model's predictions to perturbations in the training dataset:
$$\text{Variance} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \frac{1}{B} \sum_{b=1}^B \|\hat{\mathbf{y}}_i^{(b)} - \bar{\mathbf{f}}(\mathbf{x}_i)\|_2^2$$

### 3.4 The Classic U-Shaped Tradeoff
- As hidden layer capacity expands from $H = 0$ (Linear) to $H = 256$:
  - $\text{Bias}^2$ decreases monotonically from $0.285$ to $0.038$.
  - $\text{Variance}$ increases monotonically from $0.012$ to $0.298$.
  - Total expected generalization error forms a convex U-shaped curve, reaching its global minimum at $H = 32$ ($\text{Total Error} \approx 0.145$).

---

## 4. Computational Vectorization Benchmarking

Evaluating $N$ samples across $C = 5$ classes involves $N \times C$ exponentiations, normalizations, and logarithm calculations.
- **Scalar Python Loop:** Suffers significant overhead from CPython bytecode interpretation, object allocation, and cache misses.
- **Vectorized NumPy / BLAS:** Executes SIMD vectorization and cache-aligned matrix blocks, achieving $120\times$ to $480\times$ speedups across batch sizes $N \in [10, 10^4]$.

---

## 5. Artifacts and Generated Data Files

- **Training History & Dynamics:** [model_complexity_and_generalization_gap.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Training%20&%20Evaluation/model_complexity_and_generalization_gap.csv)
- **Bias-Variance Decomposition:** [bias_variance_decomposition_analysis.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Training%20&%20Evaluation/bias_variance_decomposition_analysis.csv)
- **Test Predictions & Sample Breakdown:** [test_predictions_and_cross_entropy_breakdown.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Training%20&%20Evaluation/test_predictions_and_cross_entropy_breakdown.csv)
- **Computational Benchmark:** [computational_vectorization_benchmark.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Training%20&%20Evaluation/computational_vectorization_benchmark.csv)
- **Visualizations:**
  - [07_learning_curves_underfitting_vs_overfitting.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Training%20&%20Evaluation/07_learning_curves_underfitting_vs_overfitting.png)
  - [08_bias_variance_decomposition_curve.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Training%20&%20Evaluation/08_bias_variance_decomposition_curve.png)
  - [09_test_confusion_matrix_and_computational_speedup.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Training%20&%20Evaluation/09_test_confusion_matrix_and_computational_speedup.png)
