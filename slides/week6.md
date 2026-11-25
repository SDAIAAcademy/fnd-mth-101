---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "week6"
day: 6
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/week6.html"
---

# Linear Algebra for Machine Learning

## Week 6: Eigenvectors and link ranking

Directions a matrix only stretches, and the page a random reader ends on

#### [week6#title] Eigenvectors and link ranking

*SDAIA Academy · Linear Algebra for Machine Learning*

Week 6: Directions a matrix only stretches, and the page a random reader ends on

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-101

#### [week6#agenda] Eigenvectors and link ranking

*This week*

- **Session 1 · Eigenvectors and link ranking**: Eigenvectors of symmetric matrices, covariance and the idea of principal components, and power iteration for link ranking

#### [week6#review-week5] Normal equations or QR?

*Review · week 5*

**Question:** A model fitted by the normal equations and by QR gives coefficients that differ in the third significant digit. The residual norms agree to eight digits. What is the most likely cause?

- A. Nearly collinear columns: cond(X) is high and the normal equations square it
- B. A bug in the QR routine of the library
- C. Nothing: equal residuals mean the coefficients must be equal too
- D. Too few rows; more data would make the methods agree

### Session 1: Eigenvectors and link ranking

#### [week6#s1] Eigenvectors and link ranking

*Session 1*

- Find the eigenvalues and eigenvectors of a two-by-two symmetric matrix by hand
- Explain what the leading eigenvector of a covariance matrix shows
- Rank pages by power iteration with damping, and explain dangling pages and islands

#### [week6#eigen-idea] An eigenvector is a direction the matrix only stretches

*Block A · eigenvectors*

A = [[4, 1], [1, 4]]

A (1, 0) = (4, 1): turned, not an eigenvector

A (1, 1) = (5, 5) = **5 × (1, 1)**: an eigenvector, eigenvalue 5

- **A v = λ v.** λ is the stretch factor. A negative λ flips the direction; λ = 0 squashes it: the matrix is singular.

#### [week6#characteristic] Eigenvalues solve det(A − λI) = 0

*By hand*

det([[4 − λ, 1], [1, 4 − λ]]) = (4 − λ)² − 1 = 0

4 − λ = ±1, so **λ = 5 or 3**

For a 2 × 2: λ² − (trace) λ + det = 0. Here trace 8, det 15.

```python
# file: numpy
w, V = np.linalg.eigh(A)
w          # [3., 5.]  ascending
V[:, 1]    # [0.7071, 0.7071]
```

#### [week6#symmetric-spectral] A symmetric matrix has real eigenvalues and orthogonal eigenvectors

*Symmetric matrices*

- A = Q Λ Qᵀ: rotate into the eigenvector basis, stretch each axis, rotate back
- (1, 1)/√2 and (1, −1)/√2 are orthogonal, as week 4 would want
- Covariance matrices and XᵀX are symmetric: everything this week applies to them

> **Machine learning use.** The eigenvalues of XᵀX are the squares of X's singular values; their ratio is cond(X)², week 5's number.

#### [week6#check-eigen] Reading eigenvalues

*Checkpoint*

**Question:** A symmetric 3 × 3 matrix has eigenvalues 9, 4 and 0. Which statement is right?

- A. Its diagonal entries are 9, 4 and 0
- B. It is invertible, because two eigenvalues are positive
- C. It has rank 2 and squashes one direction to zero
- D. Its eigenvectors have lengths 9, 4 and 0

#### [week6#covariance] A covariance matrix records how features vary together

*Block B · covariance*

| Learner | Quiz 1 | Quiz 2 |
| --- | --- | --- |
| 1 | 60 | 65 |
| 2 | 70 | 72 |
| 3 | 80 | 79 |
| 4 | 90 | 88 |
| 5 | 50 | 56 |

Centre the columns (means 70 and 72), then C = XcᵀXc / (n − 1):

**C = [[250, 195], [195, 152.5]]**

Diagonal: variances. Off the diagonal: how the two move together.

#### [week6#pca-idea] The leading eigenvector of the covariance is the direction of greatest spread

*Principal components*

**99.94%** of the total variance lies along (0.79, 0.62)

- Eigenvalues 402.25 and 0.25
- The two quizzes are almost one measurement: "overall score"
- That direction is the **first principal component**

> **Machine learning use.** Principal component analysis (PCA) keeps the few directions that hold most of the variance. Week 7 computes it through the SVD.

#### [week6#check-pca] What does the first component tell you?

*Checkpoint*

**Question:** Four features of course pages have a covariance matrix with eigenvalues 30, 5, 1 and 0.5. What can you say?

- A. All four features carry about the same information
- B. About 82% of the variance lies along one direction, a combination of the features
- C. The first feature explains 82% of the variance
- D. The fourth feature is an error and must be removed

#### [week6#power-iteration] Multiply by the matrix again and again, and the leading eigenvector wins

*Block C · power iteration*

- Write the start as a mix of eigenvectors
- Each multiplication scales each part by its eigenvalue
- The part with the largest |λ| outgrows the rest; normalise to keep the numbers in range
- The error shrinks by about |λ₂ / λ₁| per step

#### [week6#random-reader] A page's rank is the chance a random reader is on it, in the long run

*Link ranking*

Pages P, Q, R. P links to Q and R, Q to R, R to P. Column-stochastic M; start (1/3, 1/3, 1/3).

| Step | P, Q, R |
| --- | --- |
| 1 | 0.333, 0.167, 0.500 |
| 2 | 0.500, 0.167, 0.333 |
| limit | 0.4, 0.2, 0.4 |

- **An eigenvector.** The limit satisfies M x = x: the eigenvector of eigenvalue 1, scaled to add up to 1.

#### [week6#damping] Dead ends, closed loops and link farms break the plain walk; damping repairs it

*What goes wrong*

| Problem | Effect | Repair |
| --- | --- | --- |
| A page with no links out | Probability leaks away | Jump anywhere from it |
| A closed group of pages | The walk is trapped, may cycle | Damping: jump anywhere with probability 1 − d |
| Pages that link to each other to lift one | The group gains rank | Smaller d limits it; it does not remove it |

#### [week6#check-power] Why does damping speed up convergence?

*Checkpoint*

**Question:** With damping d = 0.85, power iteration converges in about a hundred steps; at d = 0.99 it takes many more. Why?

- A. A larger d removes pages from the graph
- B. A larger d makes the leading eigenvalue larger than 1
- C. A larger d needs a better starting vector
- D. The second-largest eigenvalue is at most d, so the error shrinks more slowly when d is near 1

#### [week6#lab-6-1] Power iteration and link ranking on a synthetic web graph

Lab 6.1 · 105 min

Rank the 40 help-centre pages by power iteration, see what a dangling page, an island and a link farm do, and fix them with damping.

[W6_1.ipynb](https://sdaiaacademy.github.io/fnd-mth-101/labs/W6_1.ipynb)

- [ ] Set up; eigenpairs of a two-by-two by hand (15 min)
- [ ] Build the link matrix of the 40 pages (15 min)
- [ ] Power iteration without damping (15 min)
- [ ] Fix the dangling page and add damping (20 min)
- [ ] Convergence, and a check with the library (15 min)
- [ ] The link farm and a damping sweep (15 min)
- [ ] Record the ranking for the content lead (10 min)
- [ ] ✱ Stretch: Show that the number of steps follows the damping factor

#### [week6#debrief-undamped] Without damping, two thirds of the probability leaked away in 200 steps

*Debrief · lab 6.1*

- **0.3252** total left after 200 steps
- **38%** of what was left sat on the island P36 to P40
- **110** steps to 10⁻¹⁰ with d = 0.85

#### [week6#debrief-farm] A link-farm page ranked first, and a larger d helped the farm more

*Debrief · lab 6.1*

| d | Farm share (P31 to P34) | P31's position | Steps |
| --- | --- | --- | --- |
| 0.50 | 10.7% | 12 | 27 |
| 0.85 | 12.9% | 7 | 110 |
| 0.95 | 18.0% | 2 | 311 |
| 0.99 | 34.8% | 2 | 972 |

At d = 0.85 the top page was P32, a farm page, ahead of the home page P01 in fourth place.

#### [week6#wrap-up] Three things to keep

*Week 6 · wrap-up*

- **Directions that only stretch.** Eigenvectors; symmetric matrices have orthogonal ones and real eigenvalues.
- **Spread has a direction.** The leading eigenvector of a covariance matrix is the direction of greatest variance.
- **Repeat to find it.** Power iteration finds the leading eigenvector by products alone; damping makes it converge.

#### [week6#next] Week 7: Singular values and low rank

*Next*

Problem set 6 (60 minutes, AI off for the first attempt): labs/problem-sets/PS6.ipynb.
