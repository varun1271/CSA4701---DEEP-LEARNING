# Module 2: Data Splitting & Model Design

## 1. Module Overview
Module 2 focuses on the mathematical formulation and structural design of the recommendation models:
- Defining **Vector Dot Products** for single-pair similarity scoring.
- Formulating **Matrix Multiplication** for batch parallel retrieval across 500 products.
- Constructing **Product-to-Product Similarity Matrices** for instant catalog-wide collaborative filtering.
- Partitioning interaction data into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets.

---

## 2. Mathematical Modeling: Vector Dot Products vs Matrix Multiplication

### 2.1 Vector Dot Product for a Single Customer-Product Pair
Let customer $c$ have preference vector $\mathbf{u}_c \in \mathbb{R}^{20}$ and product $i$ have feature vector $\mathbf{p}_i \in \mathbb{R}^{20}$.
The similarity score is computed as their algebraic inner product:
$$\text{Score}(c, i) = \mathbf{u}_c \cdot \mathbf{p}_i = \sum_{k=1}^{20} u_{c, k} \cdot p_{i, k} = u_{c, 1} p_{i, 1} + u_{c, 2} p_{i, 2} + \dots + u_{c, 20} p_{i, 20}$$

Because vectors were L2-normalized in Module 1 ($\|\mathbf{u}_c\|_2 = 1$ and $\|\mathbf{p}_i\|_2 = 1$):
$$\mathbf{u}_c \cdot \mathbf{p}_i = \|\mathbf{u}_c\|_2 \|\mathbf{p}_i\|_2 \cos(\theta) = \cos(\theta) \in [-1, 1]$$
- $\cos(\theta) \to +1.0$: Strong positive alignment (Customer loves this item).
- $\cos(\theta) \approx 0.0$: Orthogonal / Neutral alignment (No shared interest).
- $\cos(\theta) \to -1.0$: Direct conflict / Negative affinity.

### 2.2 Matrix Multiplication for Multi-Customer Batch Recommendations
When an e-commerce platform serves $M$ concurrent customers (e.g., $M = 100$) against $N = 500$ products:
- Customer Matrix: $\mathbf{U} \in \mathbb{R}^{M \times 20}$
- Product Matrix: $\mathbf{P} \in \mathbb{R}^{500 \times 20} \implies \mathbf{P}^T \in \mathbb{R}^{20 \times 500}$

Instead of evaluating $M \times N = 50,000$ discrete scalar loops, the entire scoring table is computed in a single tensor contraction:
$$\mathbf{S} = \mathbf{U} \times \mathbf{P}^T \in \mathbb{R}^{M \times 500}$$
where the $(c, i)$-th entry is:
$$\mathbf{S}_{c, i} = (\mathbf{U} \mathbf{P}^T)_{c, i} = \sum_{k=1}^{20} U_{c, k} P_{i, k} = \mathbf{u}_c \cdot \mathbf{p}_i$$

### 2.3 Product-to-Product Similarity Matrix ("Customers Who Bought This Also Liked")
Matrix multiplication also computes item-item similarities across the entire catalog in one step:
$$\mathbf{S}_{item} = \mathbf{P} \mathbf{P}^T \in \mathbb{R}^{500 \times 500}$$
- Main diagonal $\mathbf{S}_{item}[i, i] = \mathbf{p}_i \cdot \mathbf{p}_i = \|\mathbf{p}_i\|_2^2 = 1.0$ (Identity self-similarity).
- Off-diagonal values $\mathbf{S}_{item}[i, j] = \mathbf{p}_i \cdot \mathbf{p}_j$ reflect stylistic similarity between products $i$ and $j$.

---

## 3. Data Splitting Methodology

A realistic interaction history of 2,000 customer-product interactions ($M = 100$ customers $\times$ 20 top interactions) was split into three subsets:
- **Training Set (70% - 1,400 interactions):** Used to fit customer baseline preferences and latent factor representations.
- **Validation Set (15% - 300 interactions):** Used for threshold calibration and tuning cutoff $K$.
- **Test Set (15% - 300 interactions):** Held-out interactions used to evaluate real-world recommendation accuracy.

---

## 4. Visual Analysis & Generated Graphs

### Graph 4: Data Splitting Proportions & Split Consistency
- **File:** [04_data_splitting_proportions.png](04_data_splitting_proportions.png)
- **Interpretation:** Panel (A) shows the clean 70% / 15% / 15% donut breakdown. Panel (B) verifies that the affinity score distributions across Train, Validation, and Test sets match, ensuring no data distribution shift.

### Graph 5: Product Similarity Matrix Heatmap ($\mathbf{P}\mathbf{P}^T$)
- **File:** [05_product_similarity_matrix_heatmap.png](05_product_similarity_matrix_heatmap.png)
- **Interpretation:** Heatmap of the matrix multiplication $\mathbf{P} \mathbf{P}^T$. The bright yellow diagonal ($\text{score} = 1.0$) confirms self-similarity. Off-diagonal clusters highlight product groupings (e.g. shirts, dresses, sportswear) with high internal similarity scores ($0.4$ to $0.7$).

### Graph 6: Model Design: Vector Dot Product vs Matrix Multiplication Architecture
- **File:** [06_vector_dot_vs_matrix_mult_architecture.png](06_vector_dot_vs_matrix_mult_architecture.png)
- **Interpretation:** Panel (A) displays the bell-shaped similarity score distribution across all 50,000 pairs (mean $\approx 0.00$), showing how Top-$K$ items stand out beyond the 95th percentile ($\ge 0.37$). Panel (B) compares the operations profile, showing that while both methods perform $1,000,000$ multiply-adds, matrix multiplication packages them into a vectorized tensor operation.

---

## 5. Artifacts Generated in Module 2
1. [02_data_splitting_model_design.py](02_data_splitting_model_design.py): Executable script for splitting and matrix design.
2. [train_interactions.csv](train_interactions.csv): 1,400 training interactions.
3. [val_interactions.csv](val_interactions.csv): 300 validation interactions.
4. [test_interactions.csv](test_interactions.csv): 300 held-out test interactions.
5. [customer_preference_matrix.npy](customer_preference_matrix.npy): Serialized Customer Preference Matrix $\mathbf{U} \in \mathbb{R}^{100 \times 20}$.
6. [product_catalog_matrix.npy](product_catalog_matrix.npy): Serialized Product Catalog Matrix $\mathbf{P} \in \mathbb{R}^{500 \times 20}$.
