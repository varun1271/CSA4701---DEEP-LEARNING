# Module 2: Data Splitting & Model Design

## 1. Module Overview
Module 2 executes rigorous stratified data partitioning, mathematical formulation of numerically stable Softmax probability estimation with temperature scaling, analytic derivation of Categorical Cross-Entropy (CCE) gradients, and model architecture design across three distinct capacity regimes (Underfitting, Optimal Regularization, and Overfitting).

---

## 2. Stratified Data Partitioning Architecture

To evaluate the generalization performance of deep classification models without introducing distribution shift or inductive sampling bias, the $N = 1,000$ image observations are split using a strict stratified $70\% / 15\% / 15\%$ scheme:

| Partition Subset | Sample Count ($N$) | Percentage ($\%$) | Class Balance (Samples / Class) | Functional Purpose in Machine Learning Workflow |
| :--- | :---: | :---: | :---: | :--- |
| **Training Set** | $700$ | $70.0\%$ | $140$ samples per class | Parameter optimization via backpropagation and gradient descent. |
| **Validation Set** | $150$ | $15.0\%$ | $30$ samples per class | Hyperparameter tuning, learning rate scheduling, and early stopping. |
| **Test Set** | $150$ | $15.0\%$ | $30$ samples per class | Final unbiased generalization evaluation and bias-variance benchmarking. |

---

## 3. Mathematical Foundations: Stable Softmax & Analytic Gradients

### 3.1 Numerically Stable Softmax (Log-Sum-Exp Trick)
Standard exponentiation of raw logits $z_c$ causes catastrophic numerical overflow when $z_c \gg 0$ ($e^{800} \to \infty$) or underflow when $z_c \ll 0$. The Log-Sum-Exp identity solves this by subtracting the maximum logit:
$$m = \max_{j} z_j$$
$$\sigma(\mathbf{z})_c = \frac{\exp(z_c - m)}{\sum_{j=0}^{C-1} \exp(z_j - m)}$$

### 3.2 Temperature-Scaled Softmax
Introducing thermodynamic temperature $T > 0$ controls the sharpness (entropy) of the output probability distribution:
$$P(\hat{Y} = c \mid \mathbf{x}, T) = \frac{\exp(z_c / T)}{\sum_{j=0}^{C-1} \exp(z_j / T)}$$
- **Low Temperature ($T \to 0^+$):** Collapses into hard one-hot argmax prediction with near-zero predictive entropy ($H(P) \to 0$).
- **Calibrated Temperature ($T = 1.0$):** Standard neural network output probabilities reflecting model confidence.
- **High Temperature ($T \gg 1$):** Approaching maximum entropy uniform distribution ($P(Y=c) \to 1/C$, $H(P) \to \ln C$).

### 3.3 Analytic Gradient Derivation
Let $\mathcal{L}_{CCE} = -\sum_{k} y_k \ln \hat{y}_k$. Differentiating with respect to logit $z_c$:
$$\frac{\partial \mathcal{L}_{CCE}}{\partial z_c} = -\sum_{k} \frac{y_k}{\hat{y}_k} \frac{\partial \hat{y}_k}{\partial z_c}$$
Since $\frac{\partial \hat{y}_k}{\partial z_c} = \hat{y}_k (\delta_{kc} - \hat{y}_c)$, where $\delta_{kc}$ is the Kronecker delta:
$$\frac{\partial \mathcal{L}_{CCE}}{\partial z_c} = -\sum_{k} \frac{y_k}{\hat{y}_k} \hat{y}_k (\delta_{kc} - \hat{y}_c) = -y_c + \hat{y}_c \sum_{k} y_k$$
Given that $\sum_k y_k = 1$ for valid categorical targets:
$$\frac{\partial \mathcal{L}_{CCE}}{\partial z_c} = \hat{y}_c - y_c$$
In tensor notation for batch size $N$:
$$\nabla_{\mathbf{Z}} \mathcal{L} = \frac{1}{N} (\hat{\mathbf{Y}} - \mathbf{Y})$$

---

## 4. Multi-Capacity Neural Architectures

To investigate the Bias-Variance tradeoff, three distinct model capacities are designed:

1. **Underfitting Model (High Bias, Low Variance):**
   - Architecture: Linear Softmax classifier ($8 \to 5$ direct projection, zero hidden layers).
   - Regularization: Heavy $L_2$ weight decay ($\lambda = 0.20$).
   - Capacity: Unable to model non-linear class decision boundaries; exhibits high training and high test cross-entropy loss.
2. **Optimal Regulated Model (Balanced Bias-Variance):**
   - Architecture: Multi-Layer Perceptron ($8 \to 32 \to 5$) with ReLU non-linear activations.
   - Regularization: Optimal weight decay ($\lambda = 10^{-3}$), Dropout ($p = 0.20$).
   - Capacity: Perfectly captures non-linear manifolds while preventing noise memorization; achieves minimum generalization gap.
3. **Overfitting Model (Low Bias, High Variance):**
   - Architecture: Deep overparameterized MLP ($8 \to 128 \to 128 \to 64 \to 5$).
   - Regularization: Zero weight decay ($\lambda = 0.0$), no dropout, unconstrained training.
   - Capacity: Memorizes training data nuances and noise; achieves near-zero training loss while validation loss explodes.

---

## 5. Artifacts and Generated Data Files

- **Split Datasets:**
  - Training Partition: [train_image_features.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Splitting%20&%20Model%20Design/train_image_features.csv)
  - Validation Partition: [val_image_features.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Splitting%20&%20Model%20Design/val_image_features.csv)
  - Test Partition: [test_image_features.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Splitting%20&%20Model%20Design/test_image_features.csv)
- **Model Parameters:** [trained_classifier_architectures.npz](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Splitting%20&%20Model%20Design/trained_classifier_architectures.npz)
- **Visualizations:**
  - [04_stratified_dataset_partitions_and_balance.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Splitting%20&%20Model%20Design/04_stratified_dataset_partitions_and_balance.png)
  - [05_softmax_probabilities_and_temperature_scaling.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Splitting%20&%20Model%20Design/05_softmax_probabilities_and_temperature_scaling.png)
  - [06_cross_entropy_loss_surface_and_gradients.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Splitting%20&%20Model%20Design/06_cross_entropy_loss_surface_and_gradients.png)
