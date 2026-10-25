# Module 1: Data Collection & Preprocessing

## 1. Module Overview
Module 1 handles dataset synthesis, feature extraction, ground-truth one-hot encoding, exploratory statistical profiling, and Z-score standardization for an enterprise image-classification system.

In this scenario:
- **Total Image Samples:** $N = 1,000$ image instances
- **Object Categories:** $C = 5$ distinct classes (`Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`)
- **Visual Embedding Dimension:** $D = 8$ engineered continuous feature descriptors representing deep visual properties
- **Target Formulation:** Multi-class categorical indicators and one-hot ground-truth target vectors $\mathbf{y}_i \in \{0, 1\}^5$ where $\sum_{c=1}^5 y_{i,c} = 1$
- **Primary Research Focus:** Formulating the probabilistic basis for Categorical Cross-Entropy (CCE) loss, analyzing intraclass compactness versus interclass separability in latent space, and establishing the groundwork for bias-variance decomposition.

---

## 2. Visual Feature Representation Architecture

The feature space captures structural geometry, texture semantics, spectral distributions, and background context:

| Feature Index | Feature Name | Representation / Scale | Extractor Source | Computer Vision & Semantic Significance |
| :---: | :--- | :---: | :--- | :--- |
| **F01** | `Edge_Contrast_Index` | $[0.1, 1.5]$ | Sobel / Canny Edge Density | Measures sharpness and structural boundary intensity. High in human-made rigid objects (airplanes, cars). |
| **F02** | `Texture_Energy_Score` | $[0.05, 1.4]$ | Gabor Filter / GLCM Energy | Captures local surface repetition and organic texture. Elevated in biological classes (cat fur, deer coats). |
| **F03** | `Color_Saturation_Mean` | $[0.05, 1.2]$ | HSV Color Space Chrominance | Quantifies chromatic vibrancy versus grayscale neutrality. Distinguishes natural wildlife from metallic vehicles. |
| **F04** | `Spatial_Symmetry_Ratio` | $[0.2, 1.0]$ | Bilateral Axis Alignment | Ratio of left-right reflectional invariance. Airplanes and frontal automobiles exhibit near-perfect bilateral symmetry. |
| **F05** | `Aspect_Ratio_Proxy` | $[0.5, 2.5]$ | Bounding Box Length / Height | Elongation metric. Automobiles are horizontally wide; deer and standing birds exhibit vertical elongation. |
| **F06** | `High_Freq_Detail` | $[0.1, 1.4]$ | 2D Laplacian Variance | Measures fine-grained micro-structure versus smooth macro-regions. High in textured fauna. |
| **F07** | `Dominant_Hue_Intensity` | $[0.05, 1.0]$ | Hue Peak Histogram Area | Spectral purity at dominant wavelength. Distinguishes blue sky / ocean backdrops from earth tones. |
| **F08** | `Background_Uniformity` | $[0.1, 1.1]$ | Spatial Inverse Variance | Measures peripheral texture flatness. High for sky/runway backgrounds; low for complex forest clutter. |

---

## 3. Mathematical Foundations: Categorical One-Hot Targets & Softmax Probabilities

Let $\mathbf{x}_i \in \mathbb{R}^D$ denote the visual feature vector for image sample $i \in \{1, 2, \dots, N\}$, and let $c^*_i \in \{0, 1, \dots, C-1\}$ be its true class index.

### 3.1 One-Hot Indicator Formulation
The target distribution is modeled as a Dirac delta (one-hot) categorical vector:
$$\mathbf{y}_i = [y_{i,0}, y_{i,1}, \dots, y_{i,C-1}]^T, \quad \text{where } y_{i,c} = \begin{cases} 1 & \text{if } c = c^*_i \\ 0 & \text{otherwise} \end{cases}$$
subject to the categorical simplex constraint:
$$\sum_{c=0}^{C-1} y_{i,c} = 1, \quad y_{i,c} \in \{0, 1\}$$

### 3.2 Model Logits and Softmax Mapping
Given unnormalized model logits $\mathbf{z}_i = \mathbf{W}\mathbf{x}_i + \mathbf{b} \in \mathbb{R}^C$, the Softmax activation maps $\mathbf{z}_i$ onto the probability simplex $\Delta^{C-1}$:
$$\hat{y}_{i,c} = P(\hat{Y}_i = c \mid \mathbf{x}_i) = \sigma(\mathbf{z}_i)_c = \frac{\exp(z_{i,c})}{\sum_{j=0}^{C-1} \exp(z_{i,j})}$$

### 3.3 Categorical Cross-Entropy (CCE) Loss
The sample-level cross-entropy loss between the true distribution $\mathbf{y}_i$ and the predicted distribution $\hat{\mathbf{y}}_i$ is defined as:
$$\mathcal{L}_{CCE}(\mathbf{y}_i, \hat{\mathbf{y}}_i) = -\sum_{c=0}^{C-1} y_{i,c} \ln(\hat{y}_{i,c} + \epsilon) = -\ln(\hat{y}_{i, c^*_i} + \epsilon)$$
where $\epsilon = 10^{-15}$ guarantees numerical stability against $\ln(0)$ singularity.

For the entire dataset of $N$ observations:
$$\mathcal{L}_{CCE} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^{C-1} y_{i,c} \ln(\hat{y}_{i,c} + \epsilon)$$

---

## 4. Preprocessing & Data Validation Pipeline

1. **Integrity Validation:** Zero null values, no negative infinity or undefined values detected across all 1,000 samples.
2. **Balanced Class Partitioning:** 200 samples per object category ($20.0\%$ per class), ensuring zero inductive class-frequency bias.
3. **Z-Score Normalization:**
   $$\tilde{x}_{i,d} = \frac{x_{i,d} - \mu_d}{\sigma_d}$$
   Transforms raw feature dimensions with disparate physical scales into zero-mean, unit-variance coordinates ($\mu = 0.00, \sigma = 1.00$).

---

## 5. Artifacts and Generated Data Files

- **Raw Dataset:** [raw_image_classification_features_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Collection%20&%20Preprocessing/raw_image_classification_features_1000.csv)
- **Preprocessed Dataset:** [preprocessed_image_features_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Collection%20&%20Preprocessing/preprocessed_image_features_1000.csv)
- **Class Metadata & Summary:** [class_metadata_and_labels_1000.csv](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Collection%20&%20Preprocessing/class_metadata_and_labels_1000.csv)
- **Visualizations:**
  - [01_class_distribution_and_feature_embeddings.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Collection%20&%20Preprocessing/01_class_distribution_and_feature_embeddings.png)
  - [02_multiclass_feature_separability_and_distributions.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Collection%20&%20Preprocessing/02_multiclass_feature_separability_and_distributions.png)
  - [03_image_feature_standardization_and_outliers.png](file:///c:/Users/sampa/OneDrive/Documents/CSA0471%20DEEP%20LEARNING/Practicals/Experiment%20-%204/Data%20Collection%20&%20Preprocessing/03_image_feature_standardization_and_outliers.png)
