# Module 1: Data Collection & Preprocessing

## 1. Module Overview
Module 1 handles data synthesis, domain feature modeling, exploratory statistical analysis, and Z-score standardization for an enterprise real-estate automated valuation system (AVM).

In this scenario:
- **Total Properties:** $N = 1,000$ housing records
- **Predictor Dimensions:** $D = 8$ continuous/discrete physical and geographical features
- **Target Variable:** Continuous Property Sale Price in USD ($140,000 to $1,850,000), normalized into thousands of USD ($k\text{USD}$) for numerical stability during gradient-based optimization
- **Primary Research Focus:** Benchmarking model complexity regimes (Linear Regression vs. Decision Tree vs. High-Complexity Polynomial) and comparing fundamental optimization algorithms (**Batch Gradient Descent**, **Stochastic Gradient Descent**, **Momentum Gradient Descent**, and **Adam**).

---

## 2. Real-Estate Feature Architecture

| Feature Index | Feature Name | Measurement Unit | Domain / Source | Computer Vision / Real-Estate Valuation Role |
| :---: | :--- | :---: | :--- | :--- |
| **F01** | `Square_Footage` | Square Feet ($\text{sq ft}$) | Architectural Blueprint | Primary volume metric ($800$ to $5,200\text{ sq ft}$) with non-linear diminishing returns. |
| **F02** | `Bedrooms` | Integer Count | Spatial Layout | Total count of dedicated private bedrooms ($1$ to $6$). |
| **F03** | `Bathrooms` | Continuous Count | Plumbing Inventory | Combined full and half bathroom count ($1.0$ to $5.0$). |
| **F04** | `Age_of_Property_Years` | Elapsed Years | Municipal Assessment | Structural aging index ($0.5$ to $48.0\text{ yrs}$), driving physical depreciation. |
| **F05** | `Distance_to_Metro_KM` | Kilometers ($\text{km}$) | GIS Geospatial Network | Commuting accessibility ($0.5$ to $28.0\text{ km}$); exponential decay in penalty. |
| **F06** | `Location_Desirability_Score` | Index ($[1.0, 10.0]$) | Urban Planning Index | Composite metric measuring neighborhood safety, amenities, and prestige. |
| **F07** | `School_Rating_Index` | Index ($[1.0, 10.0]$) | State Education Board | Local public school district academic performance score. |
| **F08** | `Lot_Area_SqFt` | Square Feet ($\text{sq ft}$) | Cadastral Land Survey | Total parcel land area ($1,500$ to $18,000\text{ sq ft}$). |

---

## 3. Mathematical Valuation Framework

The true data-generating process incorporates physical returns, structural depreciation, non-linear interactions, and market stochasticity:
$$y = f(\mathbf{x}) + \epsilon$$
$$\begin{aligned}
f(\mathbf{x}) = w_0 &+ 185 \cdot x_{\text{sqft}} - 0.012 \cdot (x_{\text{sqft}} - 2000)^2 + 18000 \cdot x_{\text{beds}} + 24000 \cdot x_{\text{baths}} \\
&+ 38000 \cdot (x_{\text{loc}})^{1.25} + 16000 \cdot x_{\text{school}} - 22000 \cdot \ln(1 + x_{\text{metro}}) \\
&- 1800 \cdot x_{\text{age}} + 12 \cdot x_{\text{lot}} + 8.5 \cdot (x_{\text{loc}} - 5.0) \cdot (x_{\text{sqft}} - 2000)
\end{aligned}$$
where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ represents irreducible market variance ($\sigma_{\text{noise}} = \$22,000$).

---

## 4. Artifacts and Generated Data Files

- **Raw Dataset:** [raw_real_estate_dataset_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Collection%20&%20Preprocessing/raw_real_estate_dataset_1000.csv)
- **Preprocessed Dataset:** [preprocessed_real_estate_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Collection%20&%20Preprocessing/preprocessed_real_estate_1000.csv)
- **Feature Metadata & Summary:** [real_estate_feature_metadata_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Collection%20&%20Preprocessing/real_estate_feature_metadata_1000.csv)
- **Visualizations:**
  - [01_housing_feature_distributions_and_target.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Collection%20&%20Preprocessing/01_housing_feature_distributions_and_target.png)
  - [02_real_estate_feature_correlation_matrix.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Collection%20&%20Preprocessing/02_real_estate_feature_correlation_matrix.png)
  - [03_feature_standardization_and_outliers.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%205/Data%20Collection%20&%20Preprocessing/03_feature_standardization_and_outliers.png)
