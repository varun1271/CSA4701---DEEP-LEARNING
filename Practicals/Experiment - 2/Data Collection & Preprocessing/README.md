# Module 1: Data Collection & Preprocessing

## 1. Module Overview
Module 1 handles data synthesis, exploratory data analysis, mathematical covariance formulation, correlation structure analysis, and probability distribution modeling for the retail customer analytics scenario.

In this scenario:
- **Customer Dataset Size:** $N = 1,000$ retail customer profiles
- **Feature Space:** $D = 8$ behavioral, financial, and engagement metrics
- **Primary Research Focus:** Analyzing the Covariance Matrix ($\mathbf{\Sigma}$) and Pearson Correlation Matrix ($\mathbf{R}$) to identify strong positive and negative relationships between customer income, spending, and purchase frequency, alongside theoretical probability distribution fitting.

---

## 2. Customer Feature Architecture

The dataset captures multi-dimensional retail customer behavior:

| Feature Index | Feature Name | Measurement Unit | Domain / Type | Description & Semantic Meaning |
| :---: | :--- | :---: | :--- | :--- |
| **F01** | `Income` | Thousand USD ($k/year) | Continuous Financial | Customer annual gross earnings ($20k to $140k). Primary indicator of purchasing capacity. |
| **F02** | `Spending` | Thousand USD ($k/year) | Continuous Financial | Total annual retail spend ($5k to $125k). Core financial target for revenue generation. |
| **F03** | `Purchase_Frequency` | Transactions / Year | Discrete Count | Total completed purchases or store visits per annum ($3$ to $75$). Key loyalty driver. |
| **F04** | `Average_Order_Value` | USD ($ / order) | Continuous Basket | Mean monetary expenditure per checkout transaction ($46 to $250). |
| **F05** | `Recency` | Elapsed Days | Continuous Behavioral | Days elapsed since most recent transaction ($5$ to $275$ days). Measures immediate churn risk. |
| **F06** | `Discount_Sensitivity` | Ratio ($[0, 1]$) | Continuous Behavioral | Fraction of transactions incentivized by coupons, sales, or promotional markdowns. |
| **F07** | `Browsing_Duration` | Minutes / Session | Continuous Engagement | Average engagement duration per digital/store session ($4$ to $55$ mins). |
| **F08** | `Loyalty_Tenure_Years` | Years | Continuous Demographic | Duration of active registered membership tenure ($1.1$ to $12.0$ years). |

---

## 3. Mathematical Foundations: Covariance vs. Correlation

### 3.1 Sample Covariance Matrix ($\mathbf{\Sigma}$)
For $N$ customer observations across $D$ features, let $\mathbf{x}_i \in \mathbb{R}^D$ denote customer $i$, with sample mean vector $\mathbf{\mu} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i$. The sample covariance between features $j$ and $k$ is given by:

$$\text{Cov}(X_j, X_k) = \Sigma_{j, k} = \frac{1}{N - 1} \sum_{i=1}^N (x_{i, j} - \mu_j)(x_{i, k} - \mu_k)$$

In compact matrix tensor notation, defining centered design matrix $\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\mathbf{\mu}^T \in \mathbb{R}^{N \times D}$:
$$\mathbf{\Sigma} = \frac{1}{N - 1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} \in \mathbb{R}^{D \times D}$$

- The diagonal entries $\Sigma_{j, j} = \text{Var}(X_j) = \sigma_j^2$ represent feature variances.
- The off-diagonal entries $\Sigma_{j, k}$ represent pairwise joint co-variation.

### 3.2 The Scale Dependence Problem of Covariance
Because covariance preserves the product of measurement units:
- $\text{Cov}(\text{Income}, \text{Average\_Order\_Value}) = +649.85$ ($\$k \times \$$)
- $\text{Cov}(\text{Income}, \text{Spending}) = +417.45$ ($\$k \times \$k$)
- $\text{Cov}(\text{Income}, \text{Discount\_Sensitivity}) = -4.37$ ($\$k \times \text{ratio}$)

Although Income and Discount Sensitivity share a very strong relationship, the numerical magnitude of their covariance ($-4.37$) appears tiny compared to $+649.85$ purely because Discount Sensitivity is bounded in $[0, 1]$. Covariance cannot be used alone to compare relationship strengths across disparate units!

### 3.3 Pearson Correlation Matrix ($\mathbf{R}$)
To eliminate measurement scale distortion, the correlation matrix standardizes covariance by the product of individual standard deviations:

$$R_{j, k} = \frac{\Sigma_{j, k}}{\sigma_j \sigma_k} = \frac{\sum_{i=1}^N (x_{i, j} - \mu_j)(x_{i, k} - \mu_k)}{\sqrt{\sum_{i=1}^N (x_{i, j} - \mu_j)^2} \sqrt{\sum_{i=1}^N (x_{i, k} - \mu_k)^2}}$$

In matrix formulation:
$$\mathbf{R} = \mathbf{D}^{-1/2} \mathbf{\Sigma} \mathbf{D}^{-1/2} \in [-1.0, +1.0]^{D \times D}$$
where $\mathbf{D} = \text{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_D^2)$.

---

## 4. Key Empirical Findings: Feature Relationships

From the computed empirical correlation matrix $\mathbf{R}$:

### 4.1 Strong Positive Relationships ($r \ge +0.50$)
1. **Income $\longleftrightarrow$ Average Order Value ($r = +0.866$, $\text{Cov} = +649.85$):** Affluent customers consistently purchase premium product tiers and larger basket volumes per checkout.
2. **Income $\longleftrightarrow$ Annual Spending ($r = +0.855$, $\text{Cov} = +417.45$):** Income serves as the upper financial envelope for total retail capital allocation.
3. **Annual Spending $\longleftrightarrow$ Purchase Frequency ($r = +0.847$, $\text{Cov} = +220.86$):** Repeat store visits and frequent order placements are direct multipliers of top-line revenue.
4. **Annual Spending $\longleftrightarrow$ Average Order Value ($r = +0.716$, $\text{Cov} = +560.70$):** High total spenders combine both high ticket prices and high transaction cadence.
5. **Income $\longleftrightarrow$ Purchase Frequency ($r = +0.596$, $\text{Cov} = +148.80$):** Higher-income shoppers exhibit stable, habitual shopping routines.

### 4.2 Strong Negative Relationships ($r \le -0.40$)
1. **Income $\longleftrightarrow$ Discount Sensitivity ($r = -0.787$, $\text{Cov} = -4.37$):** High earners display low price elasticity; their buying decisions are motivated by convenience and quality rather than sales promotions.
2. **Average Order Value $\longleftrightarrow$ Discount Sensitivity ($r = -0.697$, $\text{Cov} = -6.20$):** Heavy discount shoppers concentrate on discounted, lower-margin items.
3. **Annual Spending $\longleftrightarrow$ Discount Sensitivity ($r = -0.691$, $\text{Cov} = -4.00$):** Top enterprise spenders rarely wait for promotional clearances.
4. **Purchase Frequency $\longleftrightarrow$ Recency ($r = -0.554$, $\text{Cov} = -205.95$):** Customers with high transaction velocity have naturally low recency (they shopped recently).
5. **Annual Spending $\longleftrightarrow$ Recency ($r = -0.471$, $\text{Cov} = -342.70$):** Churning or lapsed customers (high recency) contribute significantly less revenue.

### 4.3 Near-Zero / Orthogonal Relationships ($|r| < 0.15$)
- **Loyalty Tenure $\longleftrightarrow$ Spending ($r = -0.024$) and Income ($r = -0.009$):** Membership length alone does not dictate financial wealth or current spend volume.
- **Recency $\longleftrightarrow$ Browsing Duration ($r = -0.147$):** Session browsing behavior remains independent of purchase intervals.

---

## 5. Probability Distribution Modeling

### 5.1 Annual Income: Gaussian vs. Log-Normal Distribution
Income was modeled parametrically:
- **Gaussian Fit:** $\mathcal{N}(\mu = 67.28, \sigma^2 = 467.72)$, Kolmogorov-Smirnov statistic $= 0.0199$ ($p = 0.816$, failure to reject normality, excellent parametric representation).
- **Log-Normal Fit:** $\ln(X) \sim \mathcal{N}(\mu_L, \sigma_L^2)$, capturing positive domain boundaries and right-tail skewness.

### 5.2 Purchase Frequency: Poisson Count Modeling & Overdispersion
For discrete transaction counts $k \in \{0, 1, 2, \dots\}$:
- Empirical Sample Mean: $\bar{\lambda} = 25.77$
- Empirical Sample Variance: $s^2 = 133.14$
- **Dispersion Ratio:** $\frac{s^2}{\bar{\lambda}} = \frac{133.14}{25.77} = 5.17 \gg 1.0$
- **Theoretical Insight:** While a standard Poisson distribution assumes equidispersion ($\text{Var}(X) = \mathbb{E}[X] = \lambda$), real retail customer transactions exhibit significant **overdispersion** due to latent heterogeneity. Hence, a **Negative Binomial distribution** $\text{NB}(r=6.1, p=0.19)$ provides a superior parametric fit.

---

## 6. Visual Analysis & Generated Graphs

### Graph 1: Feature Scales Before vs. After Standardization
- **File:** [01_customer_feature_distributions_and_scales.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Collection%20&%20Preprocessing/01_customer_feature_distributions_and_scales.png)
- **Interpretation:** Panel (A) displays raw feature ranges on a logarithmic scale, highlighting disparities from $0.02$ to $360.0$. Panel (B) demonstrates the preprocessed Z-score standardized features, all harmonized with zero mean and unit variance ($\mu=0, \sigma=1$).

### Graph 2: Empirical vs. Theoretical Probability Distributions
- **File:** [02_probability_distribution_fits_kde.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Collection%20&%20Preprocessing/02_probability_distribution_fits_kde.png)
- **Interpretation:** Panel (A) contrasts empirical income density with Gaussian and Log-Normal fits. Panel (B) compares discrete purchase frequency against the Poisson PMF and Negative Binomial distribution. Panel (C) displays annual spending density with its Gaussian fit.

### Graph 3: Covariance Matrix vs. Pearson Correlation Matrix Heatmaps
- **File:** [03_covariance_and_correlation_matrices.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Collection%20&%20Preprocessing/03_covariance_and_correlation_matrices.png)
- **Interpretation:** Panel (A) shows the raw covariance matrix $\mathbf{\Sigma}$ with massive scale divergences ($1202.0$ vs $-4.37$). Panel (B) highlights the standardized Pearson correlation matrix $\mathbf{R}$, clearly delineating strong positive ($+0.87, +0.85$), strong negative ($-0.79, -0.69$), and near-zero orthogonal relationships.

---

## 7. Artifacts Generated in Module 1
1. [01_data_collection_preprocessing.py](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Collection%20&%20Preprocessing/01_data_collection_preprocessing.py): Standalone Python data generation and analysis pipeline.
2. [raw_retail_customers_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Collection%20&%20Preprocessing/raw_retail_customers_1000.csv): Raw unscaled 8 behavioral features for 1,000 customers.
3. [preprocessed_retail_customers_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Collection%20&%20Preprocessing/preprocessed_retail_customers_1000.csv): Cleaned and Z-score standardized $1000 \times 8$ matrix.
4. [customer_demographics_metadata_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%202/Data%20Collection%20&%20Preprocessing/customer_demographics_metadata_1000.csv): Descriptive demographic metadata (Age, Gender, City Tier, Loyalty Tier, High-Value Target).
