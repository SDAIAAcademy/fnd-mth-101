<!-- academy:title -->
# Linear Algebra for Machine Learning

الجبر الخطي لتعلم الآلة

Data & AI Foundations Track · SDAIA Academy · FND-MTH-101 · Level 100

REPLACE: one line under the title: what participants can do after the course

Participants compute each idea of linear algebra twice: by hand on small cases and in array code on real data. They treat data as vectors, matrices as transformations, least squares as projection, and eigen and singular value decompositions as ways to find structure. Labs use public embeddings and images and a synthetic link graph. The capstone is a low-rank analysis of a public dataset, defended orally.

**Format:** Extended: 8 weeks part-time, one session and one lab a week, weekly graded problem sets and a capstone

**Course page:** https://sdaiaacademy.github.io/fnd-mth-101/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Explain the geometric meaning of vectors, dot products, norms, projections and matrix products, given two- and three-dimensional examples | Awareness | supervised AI-off test, at least 70% correct |
| LO2 | Describe what rank, eigenvalues and singular values reveal about a matrix and name one machine learning use of each, given covariance and data matrices | Awareness | written answers marked against a rubric |
| LO3 | Identify ill-conditioned or rank-deficient systems and explain their effect on a least-squares fit, given five numerical cases | Awareness | a diagnosis sheet correct on four of five |
| LO4 | Compute products, inverses, projections and decompositions by hand for small matrices and in array code for large ones, given the weekly problem sets | Applied | notebooks in which hand and code results agree within tolerance |
| LO5 | Implement least squares through QR factorisation and compare it with the normal equations, given synthetic data with nearly collinear features | Applied | a notebook comparing residuals and condition numbers, with the difference explained |
| LO6 | Apply the singular value decomposition to build low-rank approximations and principal components, given a public image, ratings or embedding matrix | Applied | the capstone notebook with error by rank and an interpretation of the leading components |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Week 1 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week1.html`](slides/week1.html) |
| Week 2 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week2.html`](slides/week2.html) |
| Week 3 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week3.html`](slides/week3.html) |
| Week 4 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week4.html`](slides/week4.html) |
| Week 5 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week5.html`](slides/week5.html) |
| Week 6 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week6.html`](slides/week6.html) |
| Week 7 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week7.html`](slides/week7.html) |
| Week 8 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week8.html`](slides/week8.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Week 1 · Session 1 | REPLACE: what participants build in lab 1.1 | [`W1_1.ipynb`](labs/W1_1.ipynb) |
| 2.1 | Week 2 · Session 1 | REPLACE: what participants build in lab 2.1 | [`W2_1.ipynb`](labs/W2_1.ipynb) |
| 3.1 | Week 3 · Session 1 | REPLACE: what participants build in lab 3.1 | [`W3_1.ipynb`](labs/W3_1.ipynb) |
| 4.1 | Week 4 · Session 1 | REPLACE: what participants build in lab 4.1 | [`W4_1.ipynb`](labs/W4_1.ipynb) |
| 5.1 | Week 5 · Session 1 | REPLACE: what participants build in lab 5.1 | [`W5_1.ipynb`](labs/W5_1.ipynb) |
| 6.1 | Week 6 · Session 1 | REPLACE: what participants build in lab 6.1 | [`W6_1.ipynb`](labs/W6_1.ipynb) |
| 7.1 | Week 7 · Session 1 | REPLACE: what participants build in lab 7.1 | [`W7_1.ipynb`](labs/W7_1.ipynb) |
| 8.1 | Week 8 · Session 1 | REPLACE: what participants build in lab 8.1 | [`W8_1.ipynb`](labs/W8_1.ipynb) |
<!-- /academy:labs -->

## Capstone

A low-rank analysis of a public dataset of the participant's choice, submitted as a notebook and defended in a ten-minute oral check in which the participant reworks one step with new numbers

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- School algebra: solving equations and reading graphs of functions
- Writes short Python programs with lists, loops and functions (FND-PRG-101 or equivalent)

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Array computing:** NumPy, SciPy, JAX
- **Notebook environment:** JupyterLab, marimo, a free hosted notebook service
- **Problem-set checking:** nbgrader, pytest
- **Hint-only tutor:** an open-weight model in tutor mode, an approved assistant with a tutor instruction
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- SDAIA Data & AI Academic Programs Standards Framework (2026): bridging unit MSF-Linear at Explain level: REPLACE: the address of the official text
