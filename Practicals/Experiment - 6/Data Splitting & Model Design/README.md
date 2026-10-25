# Module 2: Data Splitting & Model Design

## 1. Module Overview
Module 2 establishes the data partitioning framework, constructs a configurable Multi-Layer Perceptron (MLP: $8 \to 16 \to 1$) with exact backpropagation gradients, and formulates the mathematical mechanics of Momentum Gradient Descent to eliminate pathological ravine oscillations.

---

## 2. Partition Architecture

| Partition Subset | Instance Count ($N$) | Share ($\%$) | Mean Target | Mathematical Role in Optimization Benchmark |
| :--- | :---: | :---: | :---: | :--- |
| **Training Set** | $700$ | $70.0\%$ | $-0.002$ | Objective loss evaluation and gradient backpropagation. |
| **Validation Set** | $150$ | $15.0\%$ | $+0.008$ | Tracking convergence stability and momentum tuning ($\beta$). |
| **Testing Set** | $150$ | $15.0\%$ | $-0.004$ | Final generalization verification and residual analysis. |

---

## 3. Mathematical Foundations: Standard GD vs. Momentum Mechanics

### 3.1 Standard Gradient Descent
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$$
- **Oscillation Pathology:** On an ill-conditioned quadratic surface $f(w_1, w_2) = \frac{1}{2} w_1^2 + \frac{\kappa}{2} w_2^2$ where $\kappa = 50$:
  - Along the steep axis ($w_2$): $\nabla_{w_2} = 50 w_2$. If $\eta > \frac{2}{50} = 0.04$, the update overshoots and diverges exponentially.
  - Along the flat axis ($w_1$): $\nabla_{w_1} = w_1$. With $\eta = 0.038$, progress per step is only $\Delta w_1 = 0.038 w_1$, requiring hundreds of steps to traverse the canyon.

### 3.2 Momentum Gradient Descent (Polyak Momentum)
$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \eta \nabla \mathcal{L}(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$
Expanding the velocity recurrence over past time steps:
$$\mathbf{v}_{t+1} = \eta \sum_{\tau=0}^t \beta^{t - \tau} \nabla \mathcal{L}(\mathbf{w}_\tau)$$
- **Cancellation of High-Frequency Cross-Axis Oscillations:** Along $w_2$, consecutive gradients have alternating signs ($+g, -g, +g, -g$). The exponential moving average cancels these out:
  $$\sum_{\tau=0}^t \beta^{t-\tau} g_{w_2, \tau} \approx 0$$
- **Compounding Longitudinal Velocity:** Along $w_1$, gradients consistently point downhill. The velocity compounds to:
  $$v_{w_1} \to \frac{\eta}{1 - \beta} g_{w_1}$$
  For $\beta = 0.9$, the effective step size is amplified by $\frac{1}{1 - 0.9} = 10\times$, speeding up canyon traversal by an order of magnitude.

---

## 4. Artifacts and Generated Data Files

- **Split Datasets:**
  - Training Partition: [train_signals.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Splitting%20&%20Model%20Design/train_signals.csv)
  - Validation Partition: [val_signals.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Splitting%20&%20Model%20Design/val_signals.csv)
  - Test Partition: [test_signals.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Splitting%20&%20Model%20Design/test_signals.csv)
- **Model Weights:** [trained_neural_network_weights.npz](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Splitting%20&%20Model%20Design/trained_neural_network_weights.npz)
- **Visualizations:**
  - [04_dataset_split_proportions_and_response_invariance.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Splitting%20&%20Model%20Design/04_dataset_split_proportions_and_response_invariance.png)
  - [05_neural_network_architecture_and_forward_mapping.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Splitting%20&%20Model%20Design/05_neural_network_architecture_and_forward_mapping.png)
  - [06_ill_conditioned_loss_surface_and_momentum_path.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%206/Data%20Splitting%20&%20Model%20Design/06_ill_conditioned_loss_surface_and_momentum_path.png)
