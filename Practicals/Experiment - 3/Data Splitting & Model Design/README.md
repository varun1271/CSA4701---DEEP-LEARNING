# Module 2: Data Splitting & Model Design

## 1. Module Overview
Module 2 focuses on dataset partitioning, defect count probability estimation, and the geometric formulation of Information Theory metrics (Shannon Entropy, Cross-Entropy, and KL Divergence):
- **Stratified Partitioning:** Dividing $1,000$ batches into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets stratified by scrap risk to prevent dataset distribution shift.
- **Probability Estimation Architecture:** Formulating point and cumulative probability estimates for critical quality control thresholds ($k = 0, 1, 2, 3, 5, 7, 10, 15+$ defects).
- **Information Geometry:** Deriving the convex relationship between Parameter Estimation, Cross-Entropy Loss, and Kullback-Leibler Divergence.

---

## 2. Data Splitting Methodology

To ensure clean evaluation without data leakage, the 1,000 batches were split using stratified sampling based on the critical batch rejection boundary ($K \ge 10$ defects):

| Dataset Partition | Sample Count ($N$) | Percentage Share | Critical Scrap Rate ($K \ge 10$) | Primary Operational Role in Experiment |
| :--- | :---: | :---: | :---: | :--- |
| **Training Split** | 699 batches | 70.0% | 17.17% | Fit distribution parameters ($r, p$ for NB, $\lambda$ for Poisson, $p$ for Binomial). |
| **Validation Split** | 149 batches | 15.0% | 16.78% | Tune operational acceptance cutoffs and calibrate risk alert thresholds. |
| **Testing Split** | 152 batches | 15.0% | 17.76% | Unseen held-out batches strictly reserved for final information-theoretic evaluation. |

---

## 3. Probability Estimation of Observing Specified Defect Counts

A key engineering requirement of quality assurance is determining the likelihood of observing exactly $k$ defects ($P(K = k)$) and cumulative failure risks ($P(K \ge k_{\text{critical}})$):

| Defect Count ($k$) | Empirical Frequency | Negative Binomial $\text{NB}(r, p)$ | Poisson Model $\text{Pois}(\lambda)$ | Operational Quality Control Category |
| :---: | :---: | :---: | :---: | :--- |
| **$k = 0$ defects** | **4.01%** | **5.21%** | 0.34% | **Pristine Zero-Defect Batch:** High-efficiency production run. |
| **$k = 1$ defect** | **8.73%** | **8.99%** | 1.91% | **Target Optimal Zone:** Fully within standard engineering tolerance. |
| **$k = 2$ defects** | **11.02%** | **10.88%** | 5.45% | **Target Optimal Zone:** Controlled steady-state manufacturing. |
| **$k = 3$ defects** | **13.02%** | **11.30%** | 10.35% | **Target Optimal Zone:** Modal operational performance. |
| **$k = 5$ defects** | **9.30%** | **9.73%** | 16.79% | **Warning Band:** Mild thermal or mechanical drift detected. |
| **$k = 7$ defects** | **7.01%** | **7.13%** | 12.96% | **Warning Band:** Operator inspection required. |
| **$k = 10$ defects** | **3.86%** | **3.82%** | 3.33% | **Critical Scrap:** Batch rejected; root-cause analysis initiated. |
| **$k = 15$ defects** | **1.14%** | **1.08%** | 0.06% | **Extreme Outlier / Catastrophic Failure:** Severe tooling breakdown. |
| **Critical Scrap $P(K \ge 10)$** | **17.17%** | **16.85%** | **6.45%** | **Catastrophic Quality Underestimation:** Poisson underestimates risk by $> 10\%$! |

### Critical Practical Finding:
Under the standard Poisson model, the probability of zero defects ($k = 0$) is calculated as $P(K = 0) = e^{-5.70} = 0.34\%$, drastically underestimating the true empirical occurrence ($4.01\%$). Even more dangerously, the Poisson model estimates the critical scrap risk ($P(K \ge 10)$) as only **$6.45\%$**, whereas the true factory scrap rate is **$17.17\%$**. Relying on Poisson modeling would blind quality managers to nearly two-thirds of defective batches. The **Negative Binomial model** captures this tail risk with sub-percent accuracy ($16.85\%$).

---

## 4. Information Theory Geometry: Why MLE Minimizes KL Divergence

### 4.1 Equivalence of Maximum Likelihood and Minimum KL Divergence
Let $P(k)$ be the true empirical data distribution and $Q_\theta(k)$ be a parametric candidate distribution.
The KL divergence is:
$$D_{KL}(P \parallel Q_\theta) = \sum_{k} P(k) \log_2 P(k) - \sum_{k} P(k) \log_2 Q_\theta(k) = -H(P) + H(P, Q_\theta)$$
Since $H(P)$ is a constant independent of $\theta$:
$$\arg\min_\theta D_{KL}(P \parallel Q_\theta) \equiv \arg\min_\theta H(P, Q_\theta) \equiv \arg\min_\theta \left[ -\sum_{k} P(k) \ln Q_\theta(k) \right]$$
For sample observations $\{k_1, k_2, \dots, k_N\}$ where $P(k) = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(k_i = k)$:
$$-\sum_k P(k) \ln Q_\theta(k) = -\frac{1}{N} \sum_{i=1}^N \ln Q_\theta(k_i) = -\frac{1}{N} \ln \mathcal{L}(\theta)$$
Therefore:
$$\arg\min_\theta D_{KL}(P \parallel Q_\theta) \equiv \arg\max_\theta \ln \mathcal{L}(\theta)$$
**Fundamental Principle:** Finding the Maximum Likelihood Estimate (MLE) in deep learning and statistical modeling is mathematically equivalent to minimizing the Kullback-Leibler Divergence from the true physical process.

### 4.2 Convexity of the Information Surface
- As candidate parameter $\lambda$ varies away from the empirical mean $\bar{K} = 5.70$, Cross-Entropy $H(P, Q_\lambda)$ forms a strictly convex U-shaped curve.
- The global minimum occurs precisely at $\lambda^* = 5.70$, reaching the lower theoretical bound $H(P, Q_{\lambda^*}) = 4.65\text{ bits}$, where $D_{KL}(P \parallel Q_\lambda)$ attains its minimum of $0.75\text{ bits}$.
- For the two-parameter Negative Binomial distribution $\text{NB}(r, p)$, the bivariate information loss surface forms a smooth convex basin whose global minimum achieves $D_{KL} = \mathbf{0.0214 \text{ bits}}$.

---

## 5. Visual Analysis & Generated Graphs

### Graph 4: Data Partitioning Proportions & Defect Consistency
- **File:** [04_data_splitting_proportions_and_distribution.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Splitting%20&%20Model%20Design/04_data_splitting_proportions_and_distribution.png)
- **Interpretation:** Panel (A) confirms the 70% / 15% / 15% split breakdown. Panel (B) verifies that the defect count probability densities for Train, Validation, and Test sets match with near-perfect fidelity, confirming zero dataset shift across splits.

### Graph 5: PMF and Cumulative Distribution Function (CDF) with Quality Thresholds
- **File:** [05_theoretical_vs_empirical_pmf_and_cdf.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Splitting & Model Design/05_theoretical_vs_empirical_pmf_and_cdf.png)
- **Interpretation:** Panel (A) plots the discrete Defect PMF $P(K = k)$ highlighting the precise tracking of the Negative Binomial model. Panel (B) plots the Cumulative Distribution Function $F(k) = P(K \le k)$, showing the $95\%$ compliance bound at $k = 13$ and the critical scrap boundary at $k = 10$.

### Graph 6: Information Theory Surfaces: Entropy, Cross-Entropy & Convexity of KL Divergence
- **File:** [06_information_theory_geometry_entropy_kl.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Splitting & Model Design/06_information_theory_geometry_entropy_kl.png)
- **Interpretation:** Panel (A) visualizes the convex parabolic curve of Cross-Entropy and KL Divergence as a function of Poisson $\lambda$, identifying the unique global minimum at $\lambda^* = 5.70$. Panel (B) plots the bivariate contour landscape $D_{KL}(P \parallel \text{NB}(r, p))$, illustrating the smooth information-theoretic loss surface.

---

## 6. Artifacts Generated in Module 2
1. [02_data_splitting_model_design.py](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Splitting%20&%20Model%20Design/02_data_splitting_model_design.py): Executable script for data partitioning, defect probability modeling, and information theory geometry.
2. [train_batches.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Splitting%20&%20Model%20Design/train_batches.csv): 699 training batch records.
3. [val_batches.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Splitting%20&%20Model%20Design/val_batches.csv): 149 validation batch records.
4. [test_batches.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Splitting%20&%20Model%20Design/test_batches.csv): 152 held-out test batch records.
5. [fitted_distribution_params.npz](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Splitting%20&%20Model%20Design/fitted_distribution_params.npz): Serialized model parameters ($r, p$ for Negative Binomial, $\lambda$ for Poisson, $p$ for Binomial).
