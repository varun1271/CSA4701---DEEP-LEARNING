# Module 3: Training & Evaluation — Neural Network Optimization & Momentum Convergence

## Executive Overview
**Module 3 (`Training & Evaluation`)** implements, analyzes, and evaluates multi-layer neural network optimization dynamics on an ill-conditioned pathological objective function. Standard gradient descent optimization in deep learning frequently experiences destructive, transverse oscillations across high-curvature coordinate axes while exhibiting glacial longitudinal progress along shallow, primary descent valleys. 

This module rigorously evaluates standard gradient descent against gradient descent with Polyak classical momentum ($\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t)$), conducting empirical sensitivity sweeps over the momentum hyperparameter $\beta \in [0.0, 0.99]$, quantifying oscillation damping through consecutive gradient cosine dissimilarity, benchmarking computational vectorization via hardware-accelerated BLAS matrix primitives, and validating test generalization.

---

## Technical Architecture & Mathematical Formulations

### 1. Multi-Layer Perceptron (MLP) Specification
The predictive model is formulated as an $8 \to 16 \to 1$ non-linear feedforward neural architecture:
- **Input Dimension:** $D = 8$ (ill-conditioned sensor features: $\kappa(\mathbf{X}^\top \mathbf{X}) \approx 60,772.7$).
- **Hidden Layer:** $H = 16$ units with hyperbolic tangent non-linear activation:
  $$\mathbf{z}^{(1)} = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \quad \mathbf{a}^{(1)} = \tanh\left(\mathbf{z}^{(1)}\right)$$
- **Output Layer:** Linear projection for regression:
  $$\hat{\mathbf{y}} = \mathbf{a}^{(1)} \mathbf{W}^{(2)} + b^{(2)}$$
- **Loss Function:** Mean Squared Error (MSE):
  $$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \left(\hat{y}_i - y_i\right)^2 = \frac{1}{N} \|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$$

---

### 2. Analytical Backpropagation Derivatives
The analytical parameter gradients are derived via the multivariable chain rule:
$$\boldsymbol{\delta}^{(2)} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = \frac{2}{N} (\hat{\mathbf{y}} - \mathbf{y}) \in \mathbb{R}^{N \times 1}$$

$$\nabla_{\mathbf{W}^{(2)}} \mathcal{L} = {\mathbf{a}^{(1)}}^\top \boldsymbol{\delta}^{(2)} \in \mathbb{R}^{16 \times 1}, \quad \nabla_{b^{(2)}} \mathcal{L} = \sum_{i=1}^N \delta_i^{(2)} \in \mathbb{R}^{1 \times 1}$$

$$\boldsymbol{\delta}^{(1)} = \left(\boldsymbol{\delta}^{(2)} {\mathbf{W}^{(2)}}^\top\right) \odot \left(1 - \tanh^2\left(\mathbf{z}^{(1)}\right)\right) \in \mathbb{R}^{N \times 16}$$

$$\nabla_{\mathbf{W}^{(1)}} \mathcal{L} = \mathbf{X}^\top \boldsymbol{\delta}^{(1)} \in \mathbb{R}^{8 \times 16}, \quad \nabla_{\mathbf{b}^{(1)}} \mathcal{L} = \sum_{i=1}^N \boldsymbol{\delta}_i^{(1)} \in \mathbb{R}^{1 \times 16}$$

---

### 3. Optimization Mechanics: Standard GD vs. Polyak Momentum

#### Standard Gradient Descent (First-Order Steepest Descent)
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t)$$
- In ill-conditioned ravines with condition number $\kappa = \lambda_{\max} / \lambda_{\min} \gg 1$, the gradient vector $\nabla \mathcal{L}(\boldsymbol{\theta})$ points almost orthogonally to the direction of the local minimum along the high-curvature axis $\mathbf{u}_{\max}$.
- To prevent divergence along $\mathbf{u}_{\max}$, the learning rate is bounded: $\eta < \frac{2}{\lambda_{\max}}$.
- Consequently, progress along the low-curvature valley $\mathbf{u}_{\min}$ is bounded by $\eta \lambda_{\min} < \frac{2 \lambda_{\min}}{\lambda_{\max}} = \frac{2}{\kappa} \ll 1$, inducing severe oscillation and glacial convergence.

#### Polyak Classical Momentum
$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \mathbf{v}_{t+1}$$
- Expanding the recurrence as an Exponentially Weighted Moving Average (EWMA):
  $$\mathbf{v}_{t+1} = \eta \sum_{\tau=0}^t \beta^{t-\tau} \nabla \mathcal{L}(\boldsymbol{\theta}_\tau)$$
- **Transverse Damping:** Oscillating gradient components exhibit alternating signs ($\nabla_j \mathcal{L}(\boldsymbol{\theta}_\tau) \approx -\nabla_j \mathcal{L}(\boldsymbol{\theta}_{\tau-1})$), canceling out under summation:
  $$\sum_{\tau=0}^t \beta^{t-\tau} (-1)^\tau \approx 0$$
- **Longitudinal Acceleration:** Persistent descent directions compound coherently, reaching a terminal effective velocity scaled by the geometric series:
  $$\mathbf{v}_{\infty} = \frac{\eta}{1 - \beta} \nabla \mathcal{L}$$
  For $\beta = 0.9$, the effective step size along the valley floor is magnified by $\frac{1}{1 - 0.9} = 10\times$.

---

### 4. Directional Oscillation Metric Formulation
Cross-axis instability is quantitatively measured via consecutive gradient cosine dissimilarity:
$$\mathcal{S}_t = \frac{\nabla \mathcal{L}(\boldsymbol{\theta}_t) \cdot \nabla \mathcal{L}(\boldsymbol{\theta}_{t-1})}{\|\nabla \mathcal{L}(\boldsymbol{\theta}_t)\|_2 \|\nabla \mathcal{L}(\boldsymbol{\theta}_{t-1})\|_2}$$
$$\mathcal{O}_t = 1 - \mathcal{S}_t \in [0, 2]$$
- $\mathcal{O}_t \approx 0$: Perfectly aligned consecutive gradient vectors (smooth, non-oscillating trajectory).
- $\mathcal{O}_t \approx 1$: Orthogonal consecutive gradient vectors.
- $\mathcal{O}_t \to 2$: Opposing consecutive gradient vectors (destructive, high-frequency limit-cycle oscillation).

---

## Experimental Regimes & Comparative Analysis

| Optimizer Configuration | Learning Rate ($\eta$) | Momentum ($\beta$) | Convergence Stability | Oscillation Index ($\bar{\mathcal{O}}$) | Test MSE | Generalization Quality |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Standard GD (Low LR)** | 0.015 | 0.0 | Monotonic / Slow | $0.2841$ | $0.3142$ | Underconverged |
| **Standard GD (High LR)** | 0.045 | 0.0 | Violent Oscillations | $1.1482$ | $0.4891$ | Volatile Divergence Risk |
| **Low Momentum** | 0.030 | 0.5 | Moderately Damped | $0.4120$ | $0.2185$ | Stable Suboptimal |
| **Optimal Momentum** | 0.030 | 0.9 | Rapid Damped Descent | $0.1415$ | $0.1523$ | Highly Optimal Convergence |
| **High Momentum** | 0.030 | 0.98 | Overshooting Ringing | $0.8924$ | $0.2980$ | Excessive Kinetic Inertia |

---

## Generated Artifacts & Diagnostics

### 1. Tabular Evaluation Datasets
1. `neural_optimizer_convergence_and_oscillations.csv`: Epoch-by-epoch loss trajectories, gradient $L_2$ norms, parameter displacement deltas, and oscillation metrics across 150 training epochs for all 5 optimizer regimes.
2. `momentum_beta_sensitivity_analysis.csv`: Parameter sweep results across $\beta \in [0.0, 0.99]$, logging effective acceleration factors, terminal losses, test errors, and aggregate oscillation indices.
3. `computational_vectorization_benchmark.csv`: Microsecond execution timings comparing scalar loop iteration against NumPy vectorization across evaluation batch sizes $N \in [10, 10000]$.
4. `test_predictions_standard_vs_momentum.csv`: Final sample-level test predictions and residual error distributions comparing Standard GD against Momentum GD.

### 2. Publication-Grade Visualization Figures
- **Figure 07 (`07_neural_loss_convergence_and_gradient_norms.png`):**
  - *(Subplot A)* Multi-regime MSE loss decay trajectories on a logarithmic scale illustrating the convergence acceleration of $\beta = 0.9$ versus the slow stagnation of low learning rates.
  - *(Subplot B)* Dynamic gradient $L_2$ norm evolution highlighting severe oscillatory spikes in un-momentumized high-LR descent versus exponential gradient suppression under momentum.
- **Figure 08 (`08_momentum_beta_sensitivity_and_oscillation_metrics.png`):**
  - *(Subplot A)* Non-linear sensitivity U-curve of Test MSE and empirical oscillation index across the momentum coefficient spectrum $\beta \in [0.0, 0.99]$, pinpointing the global stability optimum at $\beta \in [0.85, 0.92]$.
  - *(Subplot B)* Epoch-wise oscillation damping profiles demonstrating rapid convergence towards zero directional variance for optimal momentum.
- **Figure 09 (`09_test_prediction_residuals_and_blas_speedup.png`):**
  - *(Subplot A)* Test set prediction residual distributions ($y_{\text{pred}} - y_{\text{true}}$) contrasting the high-variance, heavy-tailed error of Standard GD against the tightly centered zero-mean Gaussian distribution of Momentum GD.
  - *(Subplot B)* Hardware execution speedup bar chart verifying up to $>100\times$ acceleration achieved by BLAS vectorized matrix arithmetic over pure-Python scalar loops.

---

## Module Verification & Reproducibility
To execute Module 3 and regenerate all evaluation CSVs and publication figures:
```bash
python 03_training_and_evaluation.py
```
*Environment Requirements: Python 3.9+, NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn.*
