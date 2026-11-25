---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/index.html"
---

# Linear Algebra for Machine Learning

الجبر الخطي لتعلم الآلة

Data & AI Foundations Track · 8 weeks + capstone · 45 hours · Level 100

Compute vectors, projections, least squares and the SVD by hand and in array code, and judge when a numerical result can be trusted

Participants compute each idea of linear algebra twice: by hand on small cases and in array code on real data. They treat data as vectors, matrices as transformations, least squares as projection, and eigen and singular value decompositions as ways to find structure. Labs use public embeddings and images and a synthetic link graph. The capstone is a low-rank analysis of a public dataset, defended orally.

**Who it is for:** Beginners. Career changers and non-computing graduates closing bridging gaps before advanced study, and computing graduates refreshing the mathematics behind models; a placement test lets participants skip any course whose content they already command.

**Format:** Extended: 8 weeks part-time, one session and one lab a week, weekly graded problem sets and a capstone

## The case study

**The online learning platform (fictional).** An unnamed online platform of short courses wants better search, cheaper scoring, honest fits and a fair ranking of its help centre. Its synthetic data (60 help articles, 400 learners, 120 courses, a 40-page link graph, 3,596 ratings) is generated with a fixed seed. The labs also use public GloVe word vectors, downloaded at run time, and two datasets bundled with the libraries: the handwritten digits in scikit-learn and a CC0 test image in scikit-image. Nothing public is copied into the course.

## The course

### Week 1: Vectors and similarity

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/week1.md

- **Session 1: Vectors, dot products and similarity**
  - Compute a dot product, a norm and a cosine by hand and in NumPy
  - Explain why cosine compares directions and ignores length
  - Describe how distances behave as the dimension grows
  - [Lab 1.1: Cosine-similarity search over public sentence embeddings (105 min)](https://sdaiaacademy.github.io/fnd-mth-101/labs/W1_1.ipynb)

### Week 2: Matrices as maps, and their cost

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/week2.md

- **Session 1: Matrix products, attention scores and FLOPs**
  - Compute a matrix product by hand and state the shape of every product
  - Explain a matrix product as one map followed by another
  - Count the FLOPs and bytes of a product and compare loop and vectorised code
  - [Lab 2.1: Loop and vectorised matrix products, timed and costed in FLOPs (105 min)](https://sdaiaacademy.github.io/fnd-mth-101/labs/W2_1.ipynb)

### Week 3: Span, rank and linear systems

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/week3.md

- **Session 1: Span, rank and linear systems**
  - Decide whether a set of vectors is independent and state the rank of a matrix
  - Solve a three-by-three system by elimination by hand and with a solver
  - Explain what a rank-deficient feature table does to a solve
  - [Lab 3.1: Rank and linear systems on a feature table (105 min)](https://sdaiaacademy.github.io/fnd-mth-101/labs/W3_1.ipynb)

### Week 4: Orthogonal projection and QR

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/week4.md

- **Session 1: Orthogonal projection and QR**
  - Project a vector onto a line and onto a plane by hand and in code
  - Check the properties of a projection matrix
  - Build a QR factorisation by Gram-Schmidt and check it against the library
  - [Lab 4.1: Projections and QR by Gram-Schmidt (105 min)](https://sdaiaacademy.github.io/fnd-mth-101/labs/W4_1.ipynb)

### Week 5: Least squares and conditioning

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/week5.md

- **Session 1: Least squares and conditioning**
  - Fit a line by the normal equations by hand
  - Fit by QR and compare residuals and coefficients with the normal equations
  - Diagnose an ill-conditioned or rank-deficient system from its numbers
  - [Lab 5.1: Least squares by normal equations and by QR on nearly collinear data (105 min)](https://sdaiaacademy.github.io/fnd-mth-101/labs/W5_1.ipynb)

### Week 6: Eigenvectors and link ranking

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/week6.md

- **Session 1: Eigenvectors and link ranking**
  - Find the eigenvalues and eigenvectors of a two-by-two symmetric matrix by hand
  - Explain what the leading eigenvector of a covariance matrix shows
  - Rank pages by power iteration with damping, and explain dangling pages and islands
  - [Lab 6.1: Power iteration and link ranking on a synthetic web graph (105 min)](https://sdaiaacademy.github.io/fnd-mth-101/labs/W6_1.ipynb)

### Week 7: Singular values and low rank

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/week7.md

- **Session 1: The SVD and principal components**
  - Read a singular value decomposition and state what each factor holds
  - Build a rank-k approximation and predict its error from the singular values
  - Compute principal components of centred data and interpret the leading ones
  - [Lab 7.1: SVD image compression and PCA on public handwritten digits (105 min)](https://sdaiaacademy.github.io/fnd-mth-101/labs/W7_1.ipynb)

### Week 8: Low rank in practice, and working numerically

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/week8.md

- **Session 1: Low rank in practice**
  - Predict held-out ratings with a truncated SVD and choose the rank
  - Count the parameters of a low-rank weight update
  - Find a broadcasting or rounding error by checking code against a hand case
  - [Lab 8.1: Low-rank ratings and adapter updates, checked by hand (105 min)](https://sdaiaacademy.github.io/fnd-mth-101/labs/W8_1.ipynb)

### Capstone: Capstone oral checks

Slides: https://sdaiaacademy.github.io/fnd-mth-101/slides/capstone.md

## Capstone

A low-rank analysis of a public dataset of the participant's choice, submitted as a notebook and defended in a ten-minute oral check in which the participant reworks one step with new numbers

**What you submit:** A link to a public GitHub repository with the capstone notebook, its data source and licence, and the notebook's outputs; then a ten-minute oral check in which the participant walks through the notebook and reworks one step with new numbers the assessor gives

**Rubric:**

- Data matrix stated: source, licence, shape, centring and scaling, each choice justified: 15%
- SVD computed and checked: a small case agrees with hand work; reconstruction error reported by rank: 25%
- Rank chosen from the error curve and the singular values, with the trade-off stated: 20%
- Leading components interpreted in the dataset's own terms, with their limits: 25%
- Notebook runs from top to bottom; seed, data source and licence stated; outputs honest: 15%

## Outcomes

- **LO1** Explain the geometric meaning of vectors, dot products, norms, projections and matrix products, given two- and three-dimensional examples (Awareness) Evidence: supervised AI-off test, at least 70% correct
- **LO2** Describe what rank, eigenvalues and singular values reveal about a matrix and name one machine learning use of each, given covariance and data matrices (Awareness) Evidence: written answers marked against a rubric
- **LO3** Identify ill-conditioned or rank-deficient systems and explain their effect on a least-squares fit, given five numerical cases (Awareness) Evidence: a diagnosis sheet correct on four of five
- **LO4** Compute products, inverses, projections and decompositions by hand for small matrices and in array code for large ones, given the weekly problem sets (Applied) Evidence: notebooks in which hand and code results agree within tolerance
- **LO5** Implement least squares through QR factorisation and compare it with the normal equations, given synthetic data with nearly collinear features (Applied) Evidence: a notebook comparing residuals and condition numbers, with the difference explained
- **LO6** Apply the singular value decomposition to build low-rank approximations and principal components, given a public image, ratings or embedding matrix (Applied) Evidence: the capstone notebook with error by rank and an interpretation of the leading components

## Tools

- **Array computing:** NumPy, SciPy, JAX
- **Notebook environment:** JupyterLab, marimo, a free hosted notebook service
- **Problem-set checking:** nbgrader, pytest
- **Hint-only tutor:** an open-weight model in tutor mode, an approved assistant with a tutor instruction

## Prerequisites

- School algebra: solving equations and reading graphs of functions
- Writes short Python programs with lists, loops and functions (FND-PRG-101 or equivalent)

## Sources

### Data

- [GloVe: Global Vectors for Word Representation (Stanford NLP), the 6B 50-dimensional vectors, Public Domain Dedication and License](https://github.com/stanfordnlp/GloVe)
- [gensim-data: the release the labs download the GloVe vectors from](https://github.com/piskvorky/gensim-data)
- [scikit-learn: load_digits, the bundled handwritten digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [UCI Machine Learning Repository: Optical Recognition of Handwritten Digits (CC BY 4.0)](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [scikit-image: the bundled test images (skimage.data.brick, CC0)](https://scikit-image.org/docs/stable/api/skimage.data.html)

### Methods and tools

- [NumPy: linear algebra (numpy.linalg)](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [SciPy: linear algebra (scipy.linalg)](https://docs.scipy.org/doc/scipy/reference/linalg.html)
- [Hu et al. (2021), LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

SDAIA Academy · Data & AI Foundations Track · Linear Algebra for Machine Learning

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-25 · commit f5792fb
