# Module 1: Data Collection & Preprocessing

## 1. Module Overview
Module 1 handles the end-to-end data acquisition, feature construction, cleaning, and mathematical preprocessing for the online shopping recommendation scenario.

In this scenario:
- **Product Catalog Size:** $N = 500$ e-commerce products
- **Feature Dimensionality:** $D = 20$ numerical features per product
- **Source Dataset:** [hm_ecommerce_products_enriched.parquet](hm_ecommerce_products_enriched.parquet) containing 105,126 fashion products with metadata and 384-dimensional dense transformer embeddings.

---

## 2. Feature Architecture (20 Numerical Dimensions)
To capture both explicit catalog attributes and latent visual/semantic stylistic nuances, the 20 numerical features are structured as follows:

| Feature Index | Feature Name | Source / Type | Description & Semantic Meaning |
| :--- | :--- | :--- | :--- |
| **F01** | `F01_Product_Type` | Metadata (Categorical Code) | Garment classification (e.g., Shirt, Trousers, Dress) |
| **F02** | `F02_Department` | Metadata (Catalog Hierarchy) | Store department code (e.g., Men, Women, Divided) |
| **F03** | `F03_Section` | Metadata (Store Placement) | Store section index |
| **F04** | `F04_Garment_Group` | Metadata (Fabric Category) | Garment manufacturing grouping |
| **F05** | `F05_Colour_Group` | Metadata (Color Palette) | Fine-grained color category code (1–93) |
| **F06** | `F06_Colour_Master` | Metadata (Perceived Color) | High-level color family ID (e.g., Blue, Black, Red) |
| **F07** | `F07_Appearance_Pattern` | Metadata (Visual Print) | Graphical appearance (e.g., Solid, Striped, Dot, Floral) |
| **F08** | `F08_Desc_Word_Count` | Extracted (Textual Density) | Word count of product description |
| **F09** | `F09_Semantic_Style` | Latent Embedding (PCA 1) | Primary stylistic orientation (Casual vs Formal) |
| **F10** | `F10_Semantic_Seasonality` | Latent Embedding (PCA 2) | Seasonal weight and warmth (Summer vs Winter) |
| **F11** | `F11_Semantic_Silhouette` | Latent Embedding (PCA 3) | Garment fit silhouette (Relaxed vs Slim/Structured) |
| **F12** | `F12_Semantic_Texture` | Latent Embedding (PCA 4) | Surface texture and weave characteristics |
| **F13** | `F13_Semantic_Occasion` | Latent Embedding (PCA 5) | Usage context (Sportswear vs Evening wear) |
| **F14** | `F14_Semantic_Formality` | Latent Embedding (PCA 6) | Professionalism / office dress code affinity |
| **F15** | `F15_Semantic_Trendiness` | Latent Embedding (PCA 7) | Trend sensitivity (Classic staple vs Youth trend) |
| **F16** | `F16_Semantic_Fit` | Latent Embedding (PCA 8) | Tailoring structure and cut details |
| **F17** | `F17_Semantic_Fabric` | Latent Embedding (PCA 9) | Material composition (Denim/Cotton vs Knit/Silk) |
| **F18** | `F18_Semantic_Detailing` | Latent Embedding (PCA 10) | Complexity of embellishment / hardware |
| **F19** | `F19_Semantic_Versatility` | Latent Embedding (PCA 11) | Multi-outfit compatibility index |
| **F20** | `F20_Semantic_Tone` | Latent Embedding (PCA 12) | Aesthetic mood and brightness tone |

---

## 3. Preprocessing Pipeline & Mathematical Formulation

### 3.1 The Problem with Raw Features
In raw e-commerce catalogs:
- `F07_Appearance_Pattern` has values around $1,010,000$.
- `F02_Department` spans from $1,201$ to $9,989$.
- `F08_Desc_Word_Count` ranges from $2$ to $111$.
- `F09`–`F20` (latent embeddings) range from $-0.4$ to $+0.4$.

If raw features were fed directly into vector dot products:
$$\mathbf{u} \cdot \mathbf{p}_{raw} = \sum_{k=1}^{20} u_k p_{k, raw}$$
The single feature `F07_Appearance_Pattern` ($\approx 10^6$) would account for $>99.99\%$ of the entire dot product value! Colors, styles, silhouettes, and descriptions would be completely ignored.

### 3.2 Standardization (Z-Score Normalization)
Each feature $k$ across all $N$ products is centered to mean 0 and scaled to unit variance:
$$\tilde{p}_{i, k} = \frac{p_{i, k} - \mu_k}{\sigma_k}$$
where $\mu_k = \frac{1}{N}\sum_{i=1}^N p_{i, k}$ and $\sigma_k = \sqrt{\frac{1}{N}\sum_{i=1}^N (p_{i, k} - \mu_k)^2}$.

### 3.3 L2 Unit-Norm Normalization
To enable similarity calculation via vector dot products, each product feature vector is normalized to have unit Euclidean norm ($\|\mathbf{p}_i\|_2 = 1$):
$$\hat{\mathbf{p}}_i = \frac{\tilde{\mathbf{p}}_i}{\|\tilde{\mathbf{p}}_i\|_2} = \frac{\tilde{\mathbf{p}}_i}{\sqrt{\sum_{k=1}^{20} \tilde{p}_{i, k}^2}}$$
**Key Consequence:** For any customer vector $\hat{\mathbf{u}}$ and product vector $\hat{\mathbf{p}}$, their vector dot product is identically equal to their **Cosine Similarity**:
$$\hat{\mathbf{u}} \cdot \hat{\mathbf{p}} = \|\hat{\mathbf{u}}\|_2 \|\hat{\mathbf{p}}\|_2 \cos(\theta) = 1 \cdot 1 \cdot \cos(\theta) = \cos(\theta) \in [-1, 1]$$
This guarantees that similarity scores are strictly bounded, calibrated, and invariant to arbitrary feature magnitudes.

---

## 4. Visual Analysis & Generated Graphs

### Graph 1: Before vs After Feature Scales
- **File:** [01_before_after_feature_scales.png](01_before_after_feature_scales.png)
- **Interpretation:** Panel (A) displays raw feature ranges on a logarithmic scale, highlighting severe disparities from $10^{-1}$ to $10^6$. Panel (B) demonstrates the preprocessed and L2-normalized feature values, all harmonized within $[-0.6, +0.6]$, ensuring equal representation across all 20 dimensions.

### Graph 2: Detailed Probability Density Distributions (KDE)
- **File:** [02_before_after_distribution_kde.png](02_before_after_distribution_kde.png)
- **Interpretation:** Compares four key features (`F01`, `F02`, `F04`, `F08`). Before preprocessing, densities are separated across thousands of integer units. After standardization and L2 normalization, all features align into smooth, well-conditioned distributions centered around zero.

### Graph 3: Feature Correlation Matrix Heatmap
- **File:** [03_before_after_correlation_heatmap.png](03_before_after_correlation_heatmap.png)
- **Interpretation:** Demonstrates the inter-correlation matrix among features before vs after preprocessing. PCA latent features remain orthogonal ($r \approx 0$), avoiding multicollinearity while preserving critical catalog affinity signals.

---

## 5. Artifacts Generated in Module 1
1. [01_data_collection_preprocessing.py](01_data_collection_preprocessing.py): Standalone executable Python pipeline.
2. [raw_products_500.csv](raw_products_500.csv): Raw unscaled 20 features for 500 products.
3. [preprocessed_products_500.csv](preprocessed_products_500.csv): Cleaned, standardized, and L2-normalized $500 \times 20$ matrix.
4. [products_metadata_500.csv](products_metadata_500.csv): Descriptive attributes (product name, department, type) for recommendation explainability.
