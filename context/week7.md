---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "week7"
day: 7
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/week7.html"
---

# Linear Algebra for Machine Learning

## Week 7: Singular values and low rank

Every matrix is a sum of rank-one layers, largest first

#### [week7#title] Singular values and low rank

*SDAIA Academy · Linear Algebra for Machine Learning*

Week 7: Every matrix is a sum of rank-one layers, largest first

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-101

#### [week7#agenda] Singular values and low rank

*This week*

- **Session 1 · The SVD and principal components**: The singular value decomposition, low-rank approximation and its error, and principal component analysis through the SVD

#### [week7#review-week6] What does power iteration converge to?

*Review · week 6*

**Question:** You repeat x ← A x / ‖A x‖ for a symmetric A with eigenvalues 7, 3 and −2, from a random start. What does x approach?

- A. The eigenvector of −2
- B. The eigenvector of 7, up to sign
- C. An average of the three eigenvectors
- D. Nothing: dividing by the norm each step prevents convergence

### Session 1: The SVD and principal components

#### [week7#s1] The SVD and principal components

*Session 1*

- Read a singular value decomposition and state what each factor holds
- Build a rank-k approximation and predict its error from the singular values
- Compute principal components of centred data and interpret the leading ones

#### [week7#svd-idea] Every matrix is a rotation, a stretch along the axes, and another rotation

*Block A · the SVD*

1. **1 · Vᵀ** turn the input onto the axes
2. **2 · Σ** stretch each axis by σᵢ ≥ 0
3. **3 · U** turn onto the output directions

**A = U Σ Vᵀ**, for any m × n matrix. U and V have orthonormal columns; the singular values σ₁ ≥ σ₂ ≥ … ≥ 0.

#### [week7#svd-by-hand] The singular values are the square roots of the eigenvalues of AᵀA

*By hand*

A = [[2, 2], [−1, 1]]

AᵀA = [[5, 3], [3, 5]], eigenvalues 8 and 2

**σ₁ = √8 ≈ 2.828, σ₂ = √2 ≈ 1.414**

```python
# file: numpy
U, s, Vt = np.linalg.svd(A)
s                     # [2.8284, 1.4142]
np.allclose(U @ np.diag(s) @ Vt, A)
```

#### [week7#what-svd-reveals] The singular values give rank, size and conditioning at a glance

*What it reveals*

| Read off Σ | Meaning | Week |
| --- | --- | --- |
| Number of non-zero σ | The rank | 3 |
| σ₁ | The largest stretch: the matrix's 2-norm | 7 |
| σ₁ / σₙ | The condition number | 5 |
| How fast σ falls | How well a few layers summarise the matrix | 7, 8 |

#### [week7#check-svd] Reading singular values

*Checkpoint*

**Question:** A 500 × 40 feature matrix has singular values 310, 52, 9, … , 0.004, 0.000002. What does that tell you?

- A. All 40 directions are equally informative
- B. The condition number is 310
- C. Full rank but nearly rank-deficient: cond ≈ 1.6 × 10⁸, unsafe for the normal equations
- D. The rank is 3

#### [week7#layers] A is a sum of rank-one layers, largest first: keep the first k

*Block B · low rank*

**A = σ₁ u₁v₁ᵀ + σ₂ u₂v₂ᵀ + …**

**A_k** keeps the first k layers.

Each layer is an outer product: one column pattern times one row pattern.

- **Eckart-Young.** No rank-k matrix is closer to A than A_k, and ‖A − A_k‖²_F = σ²ₖ₊₁ + … + σ²ₙ. The error is known before you build A_k.

#### [week7#low-rank-example] Two layers rebuild this 4 × 3 matrix to within 8%

*A small case*

| Rows of M |
| --- |
| (5, 4, 1) |
| (4, 5, 1) |
| (1, 1, 3) |
| (5, 5, 1) |

| k | Relative error ‖M − M_k‖ / ‖M‖ |
| --- | --- |
| 1 | 0.2398 |
| 2 | 0.0828 |
| 3 | 0 |

Singular values 11.73, 2.72 and 1.00: the error of each row is the tail of that list.

#### [week7#storage] A rank-k copy stores k(m + n + 1) numbers instead of m × n

*The trade*

**4.5%** storage of a rank-20 copy of a 1,000 × 800 matrix: 36,020 numbers of 800,000

- Small k saves space only when the σ fall fast
- Past some k, the copy is larger than the matrix
- Choose k from the error curve, not by habit

#### [week7#check-low-rank] Predict the error before computing it

*Checkpoint*

**Question:** A matrix has singular values 10, 6, 3 and 1. What is ‖A − A₂‖_F, the error of its best rank-2 approximation?

- A. 3
- B. 4
- C. √136 ≈ 11.66
- D. √10 ≈ 3.16

#### [week7#pca-through-svd] Centre the data, take its SVD: the rows of Vᵀ are the principal components

*Block C · PCA through the SVD*

```python
# file: numpy
Xc = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
components = Vt                 # one per row
share = s**2 / np.sum(s**2)     # variance share
scores = Xc @ Vt[:2].T          # 2-D view
```

- Same as week 6's eigenvectors of the covariance
- Without forming the covariance, so better rounding
- σ² / (n − 1) are the variances along each component

#### [week7#reading-components] A component is a pattern of weights over the original features

*Interpreting*

- For images: each component is itself an image; large weights mark pixels that vary together
- The sign is arbitrary: flip a component and its scores flip with it
- Look at which items score highest and lowest on it, then name what separates them
- A component explains variance, not meaning: interpretation is a claim to check

#### [week7#check-pca-svd] A component that points at the mean

*Checkpoint*

**Question:** A colleague takes the SVD of a ratings matrix without centring it. The first singular value is huge and the first right singular vector has nearly equal entries. What happened?

- A. The first layer captures the average rating, not a pattern of variation
- B. Every item is rated the same by every person
- C. The matrix has rank 1
- D. The SVD routine has failed

#### [week7#lab-7-1] SVD image compression and PCA on public handwritten digits

Lab 7.1 · 105 min

Compress a public CC0 test image by truncated SVD with error and storage by rank, then find the principal components of the public handwritten digits.

[W7_1.ipynb](https://sdaiaacademy.github.io/fnd-mth-101/labs/W7_1.ipynb)

- [ ] Set up; an SVD of a two-by-two by hand (15 min)
- [ ] The image and its singular values (10 min)
- [ ] Rank-k approximations: error and storage (20 min)
- [ ] The error is the tail of the singular values (10 min)
- [ ] Principal components of the digits (20 min)
- [ ] Explained variance and the leading components (20 min)
- [ ] Record what the leading components show (10 min)
- [ ] ✱ Stretch: Rebuild one digit from 5, 10 and 20 components and say which details come back first

#### [week7#debrief-image] Rank 50 kept the image to within 3% at a fifth of the storage

*Debrief · lab 7.1*

| k | Relative error | Storage |
| --- | --- | --- |
| 1 | 0.2113 | 0.4% |
| 10 | 0.1332 | 3.9% |
| 20 | 0.0871 | 7.8% |
| 50 | 0.0313 | 19.6% |
| 100 | 0.0137 | 39.1% |

The measured errors matched the tail of the singular values to four decimals. σ₁ = 57,279.5: the average brightness.

#### [week7#debrief-digits] Two components hold 28.5% of the digits' variance; 21 hold 90%

*Debrief · lab 7.1*

- Shares: 14.9%, 13.6%, 11.8%, 8.4%, 5.8%
- 13 components for 80%, 21 for 90%, of 64
- First component: 4 (−22.3) and 6 (−16.2) at one end, 3 (+18.7) and 9 (+13.1) at the other

- **A careful reading.** The first component weighs ink on the left of the middle rows against ink higher up and to the right: it sorts digits by where ink falls, not by which digit they are.

#### [week7#wrap-up] Three things to keep

*Week 7 · wrap-up*

- **Every matrix has an SVD.** Rotate, stretch, rotate. Σ gives rank, norm and condition number.
- **Truncate with a known error.** The error of A_k is the tail of the singular values, and no rank-k matrix does better.
- **PCA is a centred SVD.** Centre, decompose, read the components as patterns; interpret with care.

#### [week7#next] Week 8: Low rank in practice, and working numerically

*Next*

Problem set 7 (60 minutes, AI off for the first attempt): labs/problem-sets/PS7.ipynb. Shortlist two public datasets for the capstone.
