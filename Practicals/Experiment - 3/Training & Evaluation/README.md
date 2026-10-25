# Module 3: Training & Evaluation

## 1. Module Overview
Module 3 benchmarks computational efficiency, evaluates model goodness-of-fit and information divergence on held-out test data, and synthesizes industrial quality control findings for manufacturing systems.

This module addresses the core engineering and theoretical questions:
1. **Computational Speedup:** How much faster is vectorized matrix/tensor evaluation compared to iterative scalar loop calculations as production batches scale from $N = 100$ to $N = 10,000$?
2. **Model Generalization & Information Divergence:** Which distribution minimizes Cross-Entropy $H(P_{\text{test}}, Q)$ and Kullback-Leibler Divergence $D_{KL}(P_{\text{test}} \parallel Q)$ on completely unseen test batches?
3. **Defect Probability Estimation Accuracy:** How accurately do theoretical models predict specific defect counts ($k \in \{0, 1, 2, \dots, 15+\}$) and catastrophic scrap risk ($K \ge 10$)?

---

## 2. Computational Benchmark: Scalar Loops vs. Vectorized Tensor Evaluation

To evaluate throughput for real-time factory line monitoring, wall-clock latency was measured across production batch scales from $N = 100$ to $N = 10,000$:

| Production Batch Scale ($N$) | Sequential Scalar Loops (ms) | Vectorized Tensor Operations (ms) | Measured Speedup Acceleration Factor |
| :---: | :---: | :---: | :---: |
| **100 batches** | 9.64 ms | 0.055 ms | **174.0x faster** |
| **250 batches** | 24.42 ms | 0.095 ms | **255.7x faster** |
| **500 batches** | 43.73 ms | 0.155 ms | **282.7x faster** |
| **1,000 batches** | 90.88 ms | 0.278 ms | **326.5x faster** |
| **2,500 batches** | 258.59 ms | 0.585 ms | **442.3x faster** |
| **5,000 batches** | 638.03 ms | 1.182 ms | **539.8x faster** |
| **10,000 batches** | 5,447.04 ms | 8.775 ms | **620.8x faster** |

### Why is Vectorized Tensor Evaluation up to 621x Faster?
1. **Elimination of Interpreted Function Call Overhead:** The scalar loop executes $N$ discrete function calls into Scipy probability density functions with Python dynamic type-checking, dictionary argument unpacking, and boundary verifications.
2. **SIMD Vectorization & AVX Instructions:** Modern CPUs execute vectorized log-gamma evaluation (`gammaln`) simultaneously across 8 single-precision or 4 double-precision vector registers using Advanced Vector Extensions (AVX-512 / AVX2).
3. **Cache Line Locality:** Vectorized contiguous memory arrays fit entirely within high-speed L1/L2 processor caches, eliminating high-latency RAM access stalls.

---

## 3. Quantitative Information Theory & Goodness-of-Fit (Held-Out Test Set)

Evaluated on 152 completely unseen test batch records:

| Distribution Model | Cross-Entropy $H(P_{\text{test}}, Q)$ | KL Divergence $D_{KL}(P_{\text{test}} \parallel Q)$ | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Test Set Log-Likelihood |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Negative Binomial $\text{NB}(r, p)$** | **3.9797 bits** | **0.0781 bits** | **0.913%** | **1.109%** | **-419.93** |
| **Poisson $\text{Pois}(\lambda)$** | 4.7204 bits | 0.8187 bits | 2.628% | 3.715% | -497.33 |
| **Binomial $\text{Bin}(n=500, p)$** | 4.7409 bits | 0.8392 bits | 2.650% | 3.749% | -499.49 |
| **Uniform Baseline** | 4.5236 bits | 0.6219 bits | 3.462% | 3.893% | -476.60 |

### Key Information-Theoretic Insights:
- **Shannon Entropy Floor:** The intrinsic empirical uncertainty of test batches is $H(P_{\text{test}}) = \mathbf{3.9016 \text{ bits}}$.
- **Minimal Information Penalty:** The Negative Binomial model achieves a Cross-Entropy of $3.9797\text{ bits}$, incurring a negligible KL Divergence penalty of only **$0.0781\text{ bits}$** ($< 2\%$ relative information loss).
- **Log-Likelihood Superiority:** Negative Binomial yields a Log-Likelihood of $-419.93$, outperforming the Poisson model ($-497.33$) by $+77.4$ log units, which corresponds to an astronomical likelihood ratio of $e^{77.4} \approx 4 \times 10^{33}:1$ in favor of the Negative Binomial model.

---

## 4. Defect Count Probability Estimation Table (Test Set Evaluation)

| Defect Count ($k$) | Test Empirical Frequency | Negative Binomial Estimate | Poisson Model Estimate | NB Absolute Error | Poisson Error | Operational Quality Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$k = 0$** | **3.95%** | **5.21%** | 0.34% | **1.26%** | 3.61% | Pristine Zero-Defect Run |
| **$k = 1$** | **7.89%** | **8.99%** | 1.91% | **1.10%** | 5.98% | Target Optimal Tier |
| **$k = 2$** | **10.53%** | **10.88%** | 5.45% | **0.35%** | 5.08% | Target Optimal Tier |
| **$k = 3$** | **11.84%** | **11.30%** | 10.35% | **0.54%** | 1.49% | Target Optimal Tier |
| **$k = 4$** | **11.18%** | **10.74%** | 14.73% | **0.44%** | 3.55% | Acceptable Steady-State |
| **$k = 5$** | **9.21%** | **9.73%** | 16.79% | **0.52%** | 7.58% | Acceptable Steady-State |
| **$k = 6$** | **8.55%** | **8.46%** | 15.93% | **0.09%** | 7.38% | Acceptable Steady-State |
| **$k = 7$** | **7.24%** | **7.13%** | 12.96% | **0.11%** | 5.72% | Elevated Warning Zone |
| **$k = 8$** | **6.58%** | **5.86%** | 9.23% | **0.72%** | 2.65% | Elevated Warning Zone |
| **$k = 9$** | **5.26%** | **4.74%** | 5.84% | **0.52%** | 0.58% | Elevated Warning Zone |
| **$k = 10$** | **3.95%** | **3.82%** | 3.33% | **0.13%** | 0.62% | Critical Batch Scrap ($K \ge 10$) |
| **$k = 12$** | **2.63%** | **2.40%** | 0.81% | **0.23%** | 1.82% | Critical Batch Scrap ($K \ge 10$) |
| **$k = 15$** | **1.32%** | **1.08%** | 0.06% | **0.24%** | 1.26% | Extreme Defect Run ($K \ge 10$) |

---

## 5. Visual Analysis & Generated Graphs

### Graph 7: Information Theory Benchmark (Shannon Entropy, Cross-Entropy & KL Divergence)
- **File:** [07_information_theory_benchmark_and_divergence.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Training%20&%20Evaluation/07_information_theory_benchmark_and_divergence.png)
- **Interpretation:** Panel (A) compares Cross-Entropy against the empirical Shannon floor $H(P) = 3.90\text{ bits}$, showing that Negative Binomial ($3.98\text{ bits}$) sits right at the physical theoretical minimum. Panel (B) visualizes KL Divergence penalties, contrasting the negligible $0.078\text{ bits}$ of Negative Binomial against the massive $> 0.81\text{ bits}$ information loss of Poisson and Binomial models.

### Graph 8: Computational Speedup Benchmark
- **File:** [08_computational_speedup_benchmark.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Training%20&%20Evaluation/08_computational_speedup_benchmark.png)
- **Interpretation:** Panel (A) contrasts the steep execution latency of sequential scalar loops ($5,447\text{ ms}$) against the sub-10 millisecond curve of vectorized tensor evaluation on a log-log scale. Panel (B) demonstrates the continuous growth in acceleration factor, peaking at **$620.8\text{x}$** speedup at $N = 10,000$ batches.

### Graph 9: Probability Estimation Accuracy & Calibration Diagnostic
- **File:** [09_defect_probability_estimation_and_calibration.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Training%20&%20Evaluation/09_defect_probability_estimation_and_calibration.png)
- **Interpretation:** Panel (A) contrasts the low, stable estimation error of Negative Binomial ($< 1.3\%$) against the erratic errors of Poisson ($> 7.5\%$). Panel (B) displays the cumulative probability calibration curve, where Negative Binomial tracks the ideal $45^\circ$ reference line with near-perfect fidelity.

---

## 6. Artifacts Generated in Module 3
1. [03_training_and_evaluation.py](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Training%20&%20Evaluation/03_training_and_evaluation.py): Standalone evaluation and benchmarking pipeline.
2. [computational_benchmark_timing.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Training%20&%20Evaluation/computational_benchmark_timing.csv): Complete execution timing data and speedup acceleration multipliers across batch scales.
3. [distribution_goodness_of_fit_and_information_metrics.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Training%20&%20Evaluation/distribution_goodness_of_fit_and_information_metrics.csv): Full table of Shannon Entropy, Cross-Entropy, KL Divergence, MAE, RMSE, and Log-Likelihoods.
4. [defect_count_probability_estimates_summary.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Training%20&%20Evaluation/defect_count_probability_estimates_summary.csv): Probability estimates $P(K = k)$ and absolute errors across operational defect counts.
5. [test_batch_defect_evaluations.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Training%20&%20Evaluation/test_batch_defect_evaluations.csv): Complete test set batch evaluations with calculated Negative Binomial and Poisson probabilities.
