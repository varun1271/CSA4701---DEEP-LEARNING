# Module 3: Training & Evaluation

## 1. Module Overview
Module 3 benchmarks computational efficiency, evaluates classification performance on held-out test data, and synthesizes empirical covariance and correlation findings for retail decision systems.

This module addresses the core practical and theoretical questions:
1. **Computational Speedup:** How much faster is vectorized matrix Bayesian inference compared to iterative scalar loop calculations as customer batch sizes scale from $N = 100$ to $N = 10,000$?
2. **Bayesian Classification Performance:** How accurately and reliably does the Bayesian inference model identify High-Value VIP customers on unseen test data?
3. **Covariance & Correlation Insights:** Which customer features exhibit strong positive, strong negative, or orthogonal relationships, and what are their enterprise implications?

---

## 2. Computational Benchmark: Scalar Loops vs. Vectorized Matrix Inference

To demonstrate the power of vectorized linear algebra and BLAS routines, wall-clock latency was measured across customer batch sizes from $N = 100$ to $N = 10,000$:

| Customer Batch Scale ($N$) | Sequential Scalar Loops (ms) | Vectorized Matrix Operations (ms) | Measured Speedup Acceleration Factor |
| :---: | :---: | :---: | :---: |
| **100 customers** | 6.50 ms | 1.33 ms | **4.9x faster** |
| **250 customers** | 9.52 ms | 0.28 ms | **33.4x faster** |
| **500 customers** | 20.74 ms | 0.55 ms | **37.9x faster** |
| **1,000 customers** | 39.09 ms | 0.38 ms | **102.5x faster** |
| **2,500 customers** | 101.93 ms | 0.78 ms | **131.5x faster** |
| **5,000 customers** | 193.55 ms | 0.95 ms | **203.6x faster** |
| **10,000 customers** | 364.98 ms | 4.85 ms | **75.2x faster** |

### Why is Vectorized Matrix Inference up to 204x Faster?
1. **Elimination of Python Bytecode Overhead:** The loop version executes $N$ separate object dispatches, dictionary lookups, and function call frames in interpreted Python bytecode.
2. **Batch Quadratic Form Parallelism:** Vectorized matrix evaluation evaluates the entire $N \times D$ Mahalanobis distance array in a single BLAS GEMM/matrix multiply operation:
   $$\mathbf{D}^2_{M} = \text{rowsum}\left( (\mathbf{X} - \mathbf{M}) \mathbf{\Sigma}^{-1} \odot (\mathbf{X} - \mathbf{M}) \right)$$
3. **CPU Cache Locality & SIMD:** BLAS subroutines leverage contiguous memory buffers that reside entirely in L1/L2 hardware caches, avoiding high-latency RAM stalls. Modern CPUs execute vectorized fused multiply-add (FMA) instructions simultaneously across data registers.

---

## 3. Quantitative Bayesian Evaluation Metrics (Held-Out Test Set)

Evaluated on 152 completely unseen test customer records:

| Evaluation Metric | Test Score | Percentage / Value | Retail Enterprise Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **0.7566** | **75.66%** | Overall fraction of test customers correctly classified into their true tier. |
| **Precision** | **0.6786** | **67.86%** | High precision ensures marketing spend is not wasted on false-positive VIP incentives. |
| **Recall (Sensitivity)** | **0.6667** | **66.67%** | Captures two-thirds of all true VIP spenders without requiring aggressive promotional markdowns. |
| **Specificity** | **0.8105** | **81.05%** | Successfully protects standard customer margins from erroneous premium discounting. |
| **F1-Score** | **0.6726** | **0.6726** | Robust balanced harmonic mean between Precision and Recall. |
| **ROC-AUC Score** | **0.8469** | **0.8469** | Area Under Receiver Operating Characteristic Curve demonstrates strong discriminative ranking capability. |
| **PR-AUC (Avg Precision)** | **0.8226** | **0.8226** | Substantially outperforms random baseline ($0.375$), demonstrating reliable probability separation. |
| **Log-Loss** | **0.4517** | **0.4517** | Low cross-entropy confirms sharp, well-calibrated posterior confidence. |
| **Brier Score** | **0.1464** | **0.1464** | Low mean squared probability error verifying empirical calibration against true outcomes. |

### Confusion Matrix Breakdown:
- **True Positives (TP):** 38 VIP customers correctly identified and targeted for premium loyalty services.
- **True Negatives (TN):** 77 Standard shoppers correctly assigned to standard nurture campaigns.
- **False Positives (FP):** 18 Standard shoppers mistakenly flagged as VIPs.
- **False Negatives (FN):** 19 VIP shoppers misclassified as standard shoppers.

---

## 4. Feature Covariance & Correlation Strength Analysis

The empirical covariance and correlation structures computed on held-out customer data reveal clear, actionable retail relationships:

| Feature Pair | Covariance ($\Sigma_{j, k}$) | Pearson Correlation ($r$) | Relationship Category | Enterprise Retail Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Income $\longleftrightarrow$ Avg Order Value** | **+649.85** | **+0.866** | **Strong Positive** | Primary driver of high-ticket retail transactions. Affluent shoppers buy premium items and larger baskets. |
| **Income $\longleftrightarrow$ Annual Spending** | **+417.45** | **+0.855** | **Strong Positive** | Customer gross income establishes the ceiling for total retail expenditure. |
| **Spending $\longleftrightarrow$ Purchase Frequency** | **+220.86** | **+0.847** | **Strong Positive** | High-velocity shoppers generate the overwhelming majority of retail enterprise cash flow. |
| **Spending $\longleftrightarrow$ Avg Order Value** | **+560.70** | **+0.716** | **Strong Positive** | Synergistic compound effect: top grossing customers combine high visit frequency with large basket sizes. |
| **Income $\longleftrightarrow$ Purchase Frequency** | **+148.80** | **+0.596** | **Strong Positive** | Wealthier demographics shop consistently across seasonal merchandise drops. |
| **Income $\longleftrightarrow$ Discount Sensitivity** | **-4.37** | **-0.787** | **Strong Negative** | High-income customers prioritize product availability and luxury quality over coupons and markdowns. |
| **Avg Order Value $\longleftrightarrow$ Discount Sensitivity** | **-6.20** | **-0.697** | **Strong Negative** | Heavy discount shoppers restrict purchases to discounted clearance items, depressing order value. |
| **Spending $\longleftrightarrow$ Discount Sensitivity** | **-4.00** | **-0.691** | **Strong Negative** | Top enterprise spenders rarely wait for promotional coupons before purchasing. |
| **Purchase Frequency $\longleftrightarrow$ Recency** | **-205.95** | **-0.554** | **Strong Negative** | Active loyal customers exhibit very low elapsed days since their last store interaction. |
| **Spending $\longleftrightarrow$ Recency** | **-342.70** | **-0.471** | **Moderate Negative** | Increasing recency is a direct leading indicator of customer disengagement and revenue churn. |
| **Browsing Duration $\longleftrightarrow$ Spending** | **+86.73** | **+0.425** | **Moderate Positive** | Extended digital/in-store browsing creates higher serendipitous discovery and upsell conversion. |
| **Loyalty Tenure $\longleftrightarrow$ Income** | **-0.38** | **-0.009** | **Near-Zero (Orthogonal)** | Membership registration duration is independent of customer personal wealth or earnings. |
| **Loyalty Tenure $\longleftrightarrow$ Spending** | **-1.04** | **-0.024** | **Near-Zero (Orthogonal)** | Tenure alone does not drive revenue without ongoing engagement campaigns. |

---

## 5. Visual Analysis & Generated Graphs

### Graph 7: ROC, Precision-Recall, and Calibration Curves
- **File:** [07_roc_pr_and_calibration_curves.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Training%20&%20Evaluation/07_roc_pr_and_calibration_curves.png)
- **Interpretation:** Panel (A) shows the steep ROC curve rising sharply toward $(0, 1)$ with $\text{AUC} = 0.847$. Panel (B) highlights the strong Precision-Recall curve with $\text{PR-AUC} = 0.823$ far exceeding random guessing. Panel (C) presents the reliability diagram showing excellent alignment with the ideal $45^\circ$ diagonal ($\text{Brier} = 0.146$).

### Graph 8: Computational Speedup Benchmark
- **File:** [08_bayesian_computational_benchmark.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Training%20&%20Evaluation/08_bayesian_computational_benchmark.png)
- **Interpretation:** Panel (A) contrasts the linear growth of sequential loops against the sub-millisecond vectorized matrix curve on a logarithmic scale. Panel (B) demonstrates the dramatic acceleration peaking at **$203.6\text{x}$** speedup, verifying the scalability of vectorized Bayesian scoring.

### Graph 9: Confusion Matrix & Posterior Confidence Distribution
- **File:** [09_confusion_matrix_and_posterior_confidence.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Training%20&%20Evaluation/09_confusion_matrix_and_posterior_confidence.png)
- **Interpretation:** Panel (A) shows the normalized confusion matrix with $81.05\%$ true negative rate and strong true positive capture. Panel (B) illustrates the bimodal posterior probability distribution, showing sharp separation between standard customers ($P \approx 0.05$) and VIP customers ($P \approx 0.85$).

---

## 6. Artifacts Generated in Module 3
1. [03_training_and_evaluation.py](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Training%20&%20Evaluation/03_training_and_evaluation.py): Standalone evaluation and benchmarking script.
2. [computational_benchmark_timing.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Training%20&%20Evaluation/computational_benchmark_timing.csv): Detailed numerical timing results and speedup multipliers across batch sizes.
3. [bayesian_classification_metrics_summary.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Training%20&%20Evaluation/bayesian_classification_metrics_summary.csv): Full table of Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, and calibration scores.
4. [covariance_correlation_analysis_summary.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Training%20&%20Evaluation/covariance_correlation_analysis_summary.csv): Comprehensive ranking of pairwise feature covariances and correlation coefficients with retail interpretations.
5. [test_customer_bayesian_predictions.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Training%20&%20Evaluation/test_customer_bayesian_predictions.csv): Full test customer dataset with calculated Bayesian posterior VIP probabilities and predicted classifications.
