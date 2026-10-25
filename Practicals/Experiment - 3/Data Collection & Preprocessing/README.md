# Module 1: Data Collection & Preprocessing

## 1. Module Overview
Module 1 handles data acquisition, exploratory statistical analysis, discrete defect count probability distribution modeling, and information-theoretic evaluation for an industrial manufacturing quality control system.

In this scenario:
- **Production Batches:** $N = 1,000$ independent production runs
- **Batch Size:** $n = 500$ discrete units manufactured per batch
- **Target Phenomenon:** Number of defective units observed per batch ($K \in \{0, 1, 2, \dots, 25\}$)
- **Primary Research Focus:** Modeling defect counts using candidate probability distributions (Binomial, Poisson, and Negative Binomial), quantifying empirical statistical overdispersion, and evaluating information divergence via **Shannon Entropy**, **Cross-Entropy**, and **Kullback-Leibler (KL) Divergence**.

---

## 2. Manufacturing Operational Feature Architecture

The dataset integrates process environmental conditions, mechanical sensor readings, and material quality indicators:

| Feature Index | Feature Name | Measurement Unit | Domain / Sensor Type | Physical & Manufacturing Description |
| :---: | :--- | :---: | :--- | :--- |
| **F01** | `Machine_Temperature` | Degrees Celsius ($^\circ\text{C}$) | Thermal Sensor | Operating temperature of assembly machinery ($60^\circ\text{C}$ to $98^\circ\text{C}$). Overheating degrades tolerances. |
| **F02** | `Assembly_Speed` | Units / Minute | Kinematic Velocity | Line processing throughput velocity ($45$ to $125$ units/min). Rapid cycling increases assembly friction. |
| **F03** | `Vibration_Level` | Velocity ($\text{mm/s}$) | Accelerometer / Vibration | Mechanical chassis oscillation ($0.8$ to $5.5\text{ mm/s}$). High vibration induces alignment misfits. |
| **F04** | `Material_Purity` | Percentage ($\%$) | Spectrometry / Chemical | Raw material constituent grade purity ($90.0\%$ to $99.8\%$). Impurities trigger micro-cracks. |
| **F05** | `Shift_Hours` | Elapsed Hours | Operational Logistics | Duration of continuous operator shift ($1.0$ to $12.0$ hours). Captures workforce fatigue factors. |
| **F06** | `Ambient_Humidity` | Relative Humidity ($\%$) | Hygrometer / Environmental | Factory floor relative moisture ($28\%$ to $82\%$). High humidity alters bonding cures. |
| **F07** | `Tool_Wear_Index` | Dimensionless ($[0, 1]$) | Wear Sensor / Metrology | Progressive degradation index of cutting/tooling dies ($0.02$ to $0.96$). |
| **F08** | `Defect_Count` | Discrete Integer Count | Visual / Laser Inspection | Total non-conforming, defective units detected per 500-unit batch ($0$ to $25$ defects). |

---

## 3. Mathematical Foundations: Defect Count Probability Distributions

Let random variable $K \in \{0, 1, 2, \dots, n\}$ represent the number of defective units in a batch of size $n = 500$.

### 3.1 Binomial Distribution ($\text{Bin}(n, p)$)
Assuming each manufactured unit has an independent and identically distributed (i.i.d.) defect probability $p$:
$$P(K = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k \in \{0, 1, \dots, n\}$$
- Theoretical Mean: $\mathbb{E}[K] = n p = 5.78$
- Theoretical Variance: $\text{Var}(K) = n p (1 - p) \approx 5.71$
- **Limitation:** In authentic factories, defect occurrences are not independent; thermal fluctuations and material variations introduce clustering.

### 3.2 Poisson Distribution ($\text{Pois}(\lambda)$)
Under the classical Poisson limit theorem ($n \to \infty, p \to 0$ with $\lambda = n p$ constant):
$$P(K = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \{0, 1, 2, \dots\}$$
- **Equidispersion Property:** $\mathbb{E}[K] = \text{Var}(K) = \lambda = 5.78$
- **Failure in Industrial Reality:** Real manufacturing lines exhibit **overdispersion** ($\text{Var}(K) \gg \mathbb{E}[K]$). Empirical variance ($s^2 = 19.34$) is **$3.35\times$ larger** than the mean!

### 3.3 Negative Binomial Distribution ($\text{NB}(r, p)$)
To resolve overdispersion, the Negative Binomial distribution models the defect rate $\Lambda$ as a Gamma-distributed latent variable $\Lambda \sim \text{Gamma}(r, \frac{1 - p}{p})$, yielding a Gamma-Poisson compound mixture:
$$P(K = k) = \frac{\Gamma(k + r)}{k! \Gamma(r)} (1 - p)^r p^k, \quad k \in \{0, 1, 2, \dots\}$$
Parameter estimation via Method of Moments:
$$p = \frac{\bar{K}}{s^2} = \frac{5.7800}{19.3409} = 0.2988, \qquad r = \frac{\bar{K}^2}{s^2 - \bar{K}} = \frac{5.7800^2}{19.3409 - 5.7800} = 2.4632$$
- Mean: $\mathbb{E}[K] = \frac{r(1-p)}{p} = 5.78$
- Variance: $\text{Var}(K) = \frac{r(1-p)}{p^2} = 19.34$ (identically matches empirical overdispersion!)

---

## 4. Information Theory: Shannon Entropy, Cross-Entropy & KL Divergence

To objectively evaluate which probability model preserves the maximum information about the true physical defect generation process, information-theoretic metrics were computed in bits:

### 4.1 Shannon Entropy $H(P)$
Measures the fundamental, irreducible uncertainty or information capacity inherent in empirical manufacturing defect distribution $P$:
$$H(P) = -\sum_{k=0}^{k_{\max}} P(k) \log_2 P(k) = \mathbf{3.9061 \text{ bits}}$$

### 4.2 Cross-Entropy $H(P, Q)$
Measures the average bits required to identify an empirical outcome $k \sim P$ when encoded using candidate model distribution $Q$:
$$H(P, Q) = -\sum_{k=0}^{k_{\max}} P(k) \log_2 Q(k)$$

### 4.3 Kullback-Leibler (KL) Divergence $D_{KL}(P \parallel Q)$
Measures the statistical information penalty or relative entropy lost by approximating the true empirical reality $P$ with model $Q$:
$$D_{KL}(P \parallel Q) = \sum_{k=0}^{k_{\max}} P(k) \log_2 \left(\frac{P(k)}{Q(k)}\right) = H(P, Q) - H(P) \ge 0$$

### 4.4 Information Theory Empirical Comparison

| Candidate Distribution $Q(k)$ | Cross-Entropy $H(P, Q)$ | KL Divergence $D_{KL}(P \parallel Q)$ | Information Inefficiency | Goodness-of-Fit Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Negative Binomial $\text{NB}(r, p)$** | **3.9275 bits** | **0.0214 bits** | **0.55%** | **Optimal Fit:** Captures true overdispersion and tail risk with $< 1\%$ error. |
| **Poisson $\text{Pois}(\lambda)$** | 4.6545 bits | 0.7483 bits | 19.16% | **Suboptimal:** Severe underdispersion penalty; underestimates high-defect tails. |
| **Binomial $\text{Bin}(n=500, p)$** | 4.6744 bits | 0.7682 bits | 19.67% | **Suboptimal:** Identical failure mode to Poisson due to i.i.d. assumption. |
| **Uniform Baseline ($Q_{\text{Unif}}$)** | 4.7004 bits | 0.7943 bits | 20.34% | **Maximum Entropy:** Assumes zero domain structure; highest information loss. |

---

## 5. Visual Analysis & Generated Graphs

### Graph 1: Manufacturing Feature Scales Before vs. After Preprocessing
- **File:** [01_batch_feature_distributions_and_scales.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Collection%20&%20Preprocessing/01_batch_feature_distributions_and_scales.png)
- **Interpretation:** Panel (A) displays raw operational ranges on a symlog scale, highlighting unit disparities from $0.02$ (tool wear index) to $99.8\%$ (material purity). Panel (B) demonstrates the preprocessed Z-score standardized features, all harmonized with zero mean and unit variance ($\mu=0, \sigma=1$).

### Graph 2: Empirical Defect Count Distribution vs. Fitted Theoretical PMFs & Residuals
- **File:** [02_defect_count_probability_distribution_fits.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Collection%20&%20Preprocessing/02_defect_count_probability_distribution_fits.png)
- **Interpretation:** Panel (A) overlays the empirical defect frequency histogram against fitted Negative Binomial, Poisson, and Binomial PMFs. Panel (B) displays modeling residuals $[P(k) - Q(k)]$, proving that the Negative Binomial residuals remain virtually zero across all defect counts, while Poisson residuals exhibit large oscillating errors due to variance underestimation.

### Graph 3: Correlation Matrix & Defect Sensitivity Drivers
- **File:** [03_manufacturing_feature_correlation_and_defect_drivers.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Collection%20&%20Preprocessing/03_manufacturing_feature_correlation_and_defect_drivers.png)
- **Interpretation:** Panel (A) displays the Pearson correlation matrix across all 8 features. Panel (B) isolates the primary defect drivers, ranking Vibration Level ($r = +0.48$), Machine Temperature ($r = +0.42$), and Tool Wear Index ($r = +0.39$) as the strongest risk escalators, while Material Purity ($r = -0.46$) acts as the strongest protective quality driver.

---

## 6. Artifacts Generated in Module 1
1. [01_data_collection_preprocessing.py](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Collection%20&%20Preprocessing/01_data_collection_preprocessing.py): Standalone Python data generation, distribution fitting, and information-theoretic script.
2. [raw_manufacturing_batches_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Collection%20&%20Preprocessing/raw_manufacturing_batches_1000.csv): Raw unscaled 8 operational features for 1,000 batches.
3. [preprocessed_manufacturing_batches_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Collection%20&%20Preprocessing/preprocessed_manufacturing_batches_1000.csv): Cleaned and Z-score standardized $1000 \times 8$ matrix.
4. [batch_metadata_and_inspection_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%203/Data%20Collection%20&%20Preprocessing/batch_metadata_and_inspection_1000.csv): Batch IDs, defect rates, shift logs, and quality control classifications.
