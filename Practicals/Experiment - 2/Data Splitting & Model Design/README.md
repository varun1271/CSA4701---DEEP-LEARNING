# Module 2: Data Splitting & Model Design

## 1. Module Overview
Module 2 focuses on data partitioning, mathematical covariance geometry, spectral decomposition, and the structural design of the Bayes' Theorem customer classification engine:
- **Stratified Dataset Partitioning:** Dividing $1,000$ customer profiles into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets.
- **Covariance Spectral Decomposition:** Computing eigenvalues and eigenvectors of the sample covariance matrix $\mathbf{\Sigma}$ to uncover the principal axes of joint variation.
- **Geometric Covariance Ellipses:** Constructing $1\sigma$, $2\sigma$, and $3\sigma$ confidence ellipses in the feature plane.
- **Bayesian Decision Theory:** Formulating Bayes' Theorem with Multivariate Gaussian class-conditional likelihoods to predict High-Value VIP customers.

---

## 2. Data Splitting Methodology

To evaluate model generalization without data leakage, the 1,000 customer records were partitioned using stratified sampling on the `High_Value_Customer` target:

| Dataset Partition | Sample Count ($N$) | Percentage Share | High-Value VIP Ratio | Primary Functional Role in Experiment |
| :--- | :---: | :---: | :---: | :--- |
| **Training Split** | 699 customers | 70.0% | 37.20% | Fit class priors $P(C_k)$, class conditional means $\mathbf{\mu}_k$, and covariance matrices $\mathbf{\Sigma}_k$. |
| **Validation Split** | 149 customers | 15.0% | 36.91% | Calibrate posterior classification thresholds and tune Bayesian hyperparameters. |
| **Testing Split** | 152 customers | 15.0% | 37.50% | Unseen held-out customer profiles strictly reserved for final performance evaluation. |

---

## 3. Covariance Geometry & Spectral Decomposition

### 3.1 Spectral Decomposition ($\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$)
For the training sample covariance matrix $\mathbf{\Sigma}_{\text{train}} \in \mathbb{R}^{8 \times 8}$, spectral decomposition solves the characteristic equation:
$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
where $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_8 \ge 0$ are the eigenvalues, and $\mathbf{v}_i$ are orthonormal eigenvectors representing the principal directions of customer variation.

| Principal Component | Eigenvalue ($\lambda_i$) | Variance Explained (%) | Cumulative Variance (%) | Dominant Contributing Features |
| :---: | :---: | :---: | :---: | :--- |
| **PC 1** | **2,192.63** | **63.71%** | **63.71%** | Average Order Value ($-0.67$), Spending ($-0.43$), Income ($-0.43$), Recency ($+0.38$) |
| **PC 2** | **886.66** | **25.76%** | **89.47%** | Recency ($+0.88$), Purchase Frequency ($-0.32$), Spending ($-0.26$) |
| **PC 3** | **224.34** | **6.52%** | **95.99%** | Purchase Frequency ($+0.79$), Spending ($+0.41$), Average Order Value ($-0.37$) |
| **PC 4** | 70.05 | 2.04% | 98.03% | Browsing Duration ($+0.98$) |
| **PC 5** | 47.96 | 1.39% | 99.42% | Browsing Duration, Discount Sensitivity |
| **PC 6–8** | 20.00 | 0.58% | 100.00% | Residual noise dimensions |

**Key Geometric Insight:** The first two principal components explain **89.47%** of the entire multi-dimensional customer variance. The principal eigenvector $\mathbf{v}_1$ aligns directly along the joint financial trajectory of **Income + Spending + Average Order Value**, confirming that financial scale is the single largest source of customer dispersion.

### 3.2 Mahalanobis Distance & Covariance Confidence Ellipses
In multi-dimensional feature space, standard Euclidean distance $\|\mathbf{x} - \mathbf{\mu}\|_2$ fails because it assumes isotropic, uncorrelated features. The **Mahalanobis Distance** measures statistical distance accounting for feature covariance:
$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})}$$

The loci of constant Mahalanobis distance form concentric hyper-ellipsoids centered at $\mathbf{\mu}$:
- **$1\sigma$ Confidence Ellipse ($D_M = 1$):** Encloses $39.3\%$ of the probability mass in 2D ($\chi^2_2(p=0.393) = 1.0$).
- **$2\sigma$ Confidence Ellipse ($D_M = 2$):** Encloses $86.5\%$ of probability mass ($\chi^2_2(p=0.865) = 4.0$).
- **$3\sigma$ Confidence Ellipse ($D_M = 3$):** Encloses $98.9\%$ of probability mass ($\chi^2_2(p=0.989) = 9.0$).

---

## 4. Bayes' Theorem Model Architecture

### 4.1 Problem Formulation
Let $Y \in \{C_0, C_1\}$ represent customer status:
- $C_0$: Standard Customer
- $C_1$: High-Value VIP Customer
Given an observed feature vector $\mathbf{x} \in \mathbb{R}^8$ (Income, Spending, Frequency, etc.), Bayes' Theorem computes the exact posterior probability:

$$P(C_1 \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C_1) P(C_1)}{P(\mathbf{x})} = \frac{P(\mathbf{x} \mid C_1) P(C_1)}{P(\mathbf{x} \mid C_0) P(C_0) + P(\mathbf{x} \mid C_1) P(C_1)}$$

### 4.2 Component Definitions
1. **Prior Probabilities $P(C_k)$:** Historical baseline rate of customer tiers estimated from training records:
   $$P(C_0) = 0.6280 \quad (62.8\%), \qquad P(C_1) = 0.3720 \quad (37.2\%)$$
2. **Class-Conditional Likelihood $P(\mathbf{x} \mid C_k)$:** Modeled as a Multivariate Gaussian distribution:
   $$P(\mathbf{x} \mid C_k) = \frac{1}{(2\pi)^{D/2} |\mathbf{\Sigma}_k|^{1/2}} \exp\left( -\frac{1}{2}(\mathbf{x} - \mathbf{\mu}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{x} - \mathbf{\mu}_k) \right)$$
   where $\mathbf{\mu}_k$ is the class mean vector and $\mathbf{\Sigma}_k$ is the class-specific covariance matrix.
3. **Evidence $P(\mathbf{x})$:** Total marginal probability ensuring posterior normalization $\sum_k P(C_k \mid \mathbf{x}) = 1.0$.

### 4.3 Maximum A Posteriori (MAP) Decision Rule
A customer is classified as High-Value VIP if the posterior odds exceed unity:
$$\hat{Y} = \arg\max_{k \in \{0, 1\}} P(C_k \mid \mathbf{x}) \iff \ln \frac{P(C_1 \mid \mathbf{x})}{P(C_0 \mid \mathbf{x})} \ge 0$$

Expanding using the log-likelihood ratio:
$$\ln \frac{P(\mathbf{x} \mid C_1)}{P(\mathbf{x} \mid C_0)} + \ln \frac{P(C_1)}{P(C_0)} \ge 0$$
Because the class covariance matrices differ ($\mathbf{\Sigma}_1 \neq \mathbf{\Sigma}_0$), the quadratic terms $-\frac{1}{2}\mathbf{x}^T(\mathbf{\Sigma}_1^{-1} - \mathbf{\Sigma}_0^{-1})\mathbf{x}$ do not cancel, yielding a **Quadratic Decision Boundary** that naturally adapts to class-specific variance orientations.

---

## 5. Visual Analysis & Generated Graphs

### Graph 4: Data Partitioning Proportions & Distribution Consistency
- **File:** [04_data_splitting_proportions_and_distribution.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/04_data_splitting_proportions_and_distribution.png)
- **Interpretation:** Panel (A) confirms the 70% / 15% / 15% split breakdown. Panel (B) demonstrates that the spending probability density curves for Train, Validation, and Test sets match with near-perfect fidelity, confirming zero dataset shift across splits.

### Graph 5: Covariance Ellipses & Eigenvector Principal Axes
- **File:** [05_covariance_ellipse_and_eigenvectors.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/05_covariance_ellipse_and_eigenvectors.png)
- **Interpretation:** Panel (A) displays the joint scatter of Income vs. Spending with $1\sigma, 2\sigma, 3\sigma$ confidence ellipses and the dominant eigenvector arrow pointing along the diagonal of strong positive co-variation. Panel (B) demonstrates the same elliptical geometry for Purchase Frequency vs. Spending.

### Graph 6: Bayesian Posterior Density & Decision Boundary Architecture
- **File:** [06_bayesian_decision_boundary_and_priors.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/06_bayesian_decision_boundary_and_priors.png)
- **Interpretation:** Panel (A) illustrates how the flat prior $P(C_1) = 0.37$ is updated by the class likelihoods into a sharp sigmoidal posterior probability curve $P(C_1 \mid \text{Spending})$. Panel (B) visualizes the 2D Bayesian MAP decision surface in the Income-Spending plane, with contour gradients indicating posterior confidence.

---

## 6. Artifacts Generated in Module 2
1. [02_data_splitting_model_design.py](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/02_data_splitting_model_design.py): Executable script for partitioning and Bayesian model design.
2. [train_customers.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/train_customers.csv): 699 training customer records.
3. [val_customers.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/val_customers.csv): 149 validation customer records.
4. [test_customers.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/test_customers.csv): 152 held-out test customer records.
5. [covariance_matrix.npy](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/covariance_matrix.npy): Serialized Training Covariance Matrix $\mathbf{\Sigma} \in \mathbb{R}^{8 \times 8}$.
6. [correlation_matrix.npy](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/correlation_matrix.npy): Serialized Training Pearson Correlation Matrix $\mathbf{R} \in \mathbb{R}^{8 \times 8}$.
7. [class_priors_and_params.npz](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Splitting%20&%20Model%20Design/class_priors_and_params.npz): Serialized class priors $P(C_k)$, class conditional means $\mathbf{\mu}_k$, and covariance matrices $\mathbf{\Sigma}_k$.
