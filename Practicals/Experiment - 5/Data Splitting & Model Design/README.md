# Module 2: Data Splitting & Model Design

## 1. Module Overview
Module 2 establishes the stratified data partitioning framework, formulates three contrasting regression model architectures across model complexity tiers, and implements four fundamental gradient-based optimization algorithms from scratch.

---

## 2. Stratified Data Partitioning Architecture

To eliminate sampling bias and preserve identical property price distributions across subsets, the $N = 1,000$ properties are partitioned into quintile price strata and split into:

| Partition Subset | Property Records ($N$) | Percentage ($\%$) | Price Mean ($k\text{USD}$) | Functional Role in Machine Learning Pipeline |
| :--- | :---: | :---: | :---: | :--- |
| **Training Set** | $700$ | $70.0\%$ | $\$1,178.6k$ | Parameter estimation, gradient backpropagation, and tree split learning. |
| **Validation Set** | $150$ | $15.0\%$ | $\$1,179.4k$ | Hyperparameter tuning (tree depth, polynomial degree, learning rates). |
| **Testing Set** | $150$ | $15.0\%$ | $\$1,180.8k$ | Final unbiased generalization evaluation and bias-variance benchmarking. |

---

## 3. Model Complexity Architectures

1. **Multiple Linear Regression (Underfitting Regime):**
   $$\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_0 + \sum_{j=1}^D w_j x_j$$
   - Total Parameters: $D + 1 = 9$ weights.
   - High Bias: Incapable of capturing non-linear diminishing returns and location interactions.
2. **Decision Tree Regressor (Balanced Complexity Regime):**
   - Recursive binary orthogonal splitting minimizing Mean Squared Error impurity:
     $$Q(t) = \frac{1}{N_t} \sum_{i \in D_t} (y_i - \bar{y}_t)^2$$
   - Max depth $= 6$, producing $64$ adaptive regional leaves that capture complex interactions without overparameterizing.
3. **High-Complexity Polynomial Regression (Overfitting Regime):**
   - Degree 3 polynomial feature expansion including all powers and interaction cross-terms:
     $$\mathbf{\phi}(\mathbf{x}) = [1, x_1, \dots, x_D, x_1^2, x_1 x_2, \dots, x_D^3]^T$$
   - Total Parameters: $\binom{D + d}{d} = \binom{8 + 3}{3} = 165$ weights.
   - Severe Overfitting: Memorizes sample-specific market noise, leading to exploding test error.

---

## 4. Optimization Algorithms Formulation

1. **Batch Gradient Descent (BGD):**
   $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{N} \nabla_{\mathbf{w}} \sum_{i=1}^N \mathcal{L}_i(\mathbf{w}_t)$$
   - Deterministic, smooth monotonic descent, but high per-epoch computational cost.
2. **Stochastic Gradient Descent (SGD):**
   $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}_t)$$
   - Evaluates single random instances, introducing high stochastic variance and noisy zig-zagging.
3. **Momentum Gradient Descent:**
   $$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \eta \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}_t), \quad \mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$
   - Accumulates exponential moving average of past gradients ($\beta = 0.9$), suppressing cross-axis oscillations and compounding forward velocity.
4. **Adaptive Moment Estimation (Adam):**
   $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
   $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}, \quad \mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$
   - Automatically scales individual coordinate step sizes according to historical gradient variance.

---

## 5. Artifacts and Generated Data Files

- **Split Datasets:**
  - Training Partition: [train_real_estate.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Splitting%20&%20Model%20Design/train_real_estate.csv)
  - Validation Partition: [val_real_estate.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Splitting%20&%20Model%20Design/val_real_estate.csv)
  - Test Partition: [test_real_estate.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Splitting%20&%20Model%20Design/test_real_estate.csv)
- **Model Parameters:** [trained_regression_models.npz](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Splitting%20&%20Model%20Design/trained_regression_models.npz)
- **Visualizations:**
  - [04_dataset_split_proportions_and_price_invariance.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Splitting%20&%20Model%20Design/04_dataset_split_proportions_and_price_invariance.png)
  - [05_model_architecture_predictions_vs_actual.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Splitting%20&%20Model%20Design/05_model_architecture_predictions_vs_actual.png)
  - [06_loss_surface_and_optimizer_trajectories.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Splitting%20&%20Model%20Design/06_loss_surface_and_optimizer_trajectories.png)
