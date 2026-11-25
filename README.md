<!-- academy:title -->
# Linear Algebra for Machine Learning

الجبر الخطي لتعلم الآلة

Data & AI Foundations Track · SDAIA Academy · FND-MTH-101 · Level 100

Compute vectors, projections, least squares and the SVD by hand and in array code, and judge when a numerical result can be trusted

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
| Week 1 | Vectors and similarity | Vectors, dot products and similarity | [`slides/week1.html`](slides/week1.html) |
| Week 2 | Matrices as maps, and their cost | Matrix products, attention scores and FLOPs | [`slides/week2.html`](slides/week2.html) |
| Week 3 | Span, rank and linear systems | Span, rank and linear systems | [`slides/week3.html`](slides/week3.html) |
| Week 4 | Orthogonal projection and QR | Orthogonal projection and QR | [`slides/week4.html`](slides/week4.html) |
| Week 5 | Least squares and conditioning | Least squares and conditioning | [`slides/week5.html`](slides/week5.html) |
| Week 6 | Eigenvectors and link ranking | Eigenvectors and link ranking | [`slides/week6.html`](slides/week6.html) |
| Week 7 | Singular values and low rank | The SVD and principal components | [`slides/week7.html`](slides/week7.html) |
| Week 8 | Low rank in practice, and working numerically | Low rank in practice | [`slides/week8.html`](slides/week8.html) |
| Capstone | Capstone oral checks |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**The online learning platform (fictional).** An unnamed platform of short courses wants better search, cheaper scoring, honest fits and a fair ranking of its help centre; the labs speak to its search lead, its help-centre content lead and a data analyst, by role only. Its data (60 help articles with judged search questions, 400 learners, 120 courses, a 40-page link graph, 3,596 ratings) is synthetic, written by [`labs/prepare_data.py`](labs/prepare_data.py) with seed 101.

The labs also use three public datasets, none copied into this repository:

| Dataset | Licence | How the labs get it |
|---|---|---|
| GloVe 6B word vectors, 50 dimensions (Stanford NLP) | Public Domain Dedication and License | Downloaded at run time from the gensim-data release (about 69 MB, checked by MD5) |
| Handwritten digits, 8 × 8 (UCI) | CC BY 4.0 | Bundled with scikit-learn (`load_digits`) |
| Brick-wall test image | CC0 | Bundled with scikit-image (`skimage.data.brick`) |

The "sentence embeddings" of lab 1.1 are the mean of the GloVe vectors of a sentence's words, a simple public stand-in that the lab and the slides name as such.

## Plan and minutes

The entry's 45 hours, in minutes:

| Catalog hours | Minutes | How they are spent |
|---|---:|---|
| Instruction 14 | 840 | 8 weekly sessions × 105 minutes of teaching, checkpoints and lab debrief |
| Lab 14 | 840 | 8 labs × 105 minutes, each with steps timed to the minute and one stretch |
| Guided self-study 8 | 480 | 8 problem sets × 60 minutes ([`labs/problem-sets/`](labs/problem-sets/)) |
| Assessment 3 | 180 | 2 supervised tests × 90 minutes (45 on paper, 45 at the keyboard), after week 4 and after week 8 |
| Capstone 6 | 360 | dataset clinic 30, independent analysis 280, peer dry run 25, oral check 10, changeover 5, written feedback 10 |
| **Total** | **2,700** | 45 hours |

## Assessment

| Assessed work | Weight |
|---|---:|
| Weekly problem sets, auto-checked | 20% |
| Two supervised AI-off tests, on paper and at the keyboard | 40% |
| Capstone notebook, rubric-scored | 30% |
| Oral check | 10% |

**Passing:** 60% overall and at least 50% on each supervised test.

**Authenticity:** Proctored tests without assistance, and an oral check in which the participant reworks a capstone step with new numbers.

**Certificate:** completion with assessed work.

The problem sets are scored by [`labs/problem-sets/pscheck.py`](labs/problem-sets/pscheck.py), which stores no answers: it checks that each hand answer agrees with the participant's own code, and checks the code against properties and the library. The capstone rubric scores the notebook only; the oral check is scored on its own, so nothing counts twice.

**Held outside this repository**, in the trainer pack: the worked answers to the problem sets, the two tests' blueprint and items, the five-case diagnosis sheet for LO3 with its key, and the oral check's new-number items with their answers. Publishing them would defeat the assessment. The lab solutions are public, in [`labs/solutions/`](labs/solutions/).

## AI policy

AI-off for first attempts at problem sets and in both tests. AI-tutor in labs, giving hints without answers. AI-full in the capstone, which is defended orally.

The tutor's instruction is [`labs/tutor/TUTOR.md`](labs/tutor/TUTOR.md).

## Responsible use

Covers bias in learned representations: in the embeddings lab participants measure how closely occupation words align with gendered words and record what that means for a search built on them. Data is public or synthetic.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Week 1 · Session 1 | Cosine search over public sentence embeddings | [`W1_1.ipynb`](labs/W1_1.ipynb) |
| 2.1 | Week 2 · Session 1 | Loop and vectorised matrix products, costed in FLOPs | [`W2_1.ipynb`](labs/W2_1.ipynb) |
| 3.1 | Week 3 · Session 1 | Rank and linear systems on a feature table | [`W3_1.ipynb`](labs/W3_1.ipynb) |
| 4.1 | Week 4 · Session 1 | Projections and QR by Gram-Schmidt | [`W4_1.ipynb`](labs/W4_1.ipynb) |
| 5.1 | Week 5 · Session 1 | Least squares by normal equations and by QR | [`W5_1.ipynb`](labs/W5_1.ipynb) |
| 6.1 | Week 6 · Session 1 | Power iteration and link ranking on a synthetic web graph | [`W6_1.ipynb`](labs/W6_1.ipynb) |
| 7.1 | Week 7 · Session 1 | SVD image compression and PCA on public handwritten digits | [`W7_1.ipynb`](labs/W7_1.ipynb) |
| 8.1 | Week 8 · Session 1 | Low-rank ratings and adapter updates, checked by hand | [`W8_1.ipynb`](labs/W8_1.ipynb) |
<!-- /academy:labs -->

## Capstone

A low-rank analysis of a public dataset of the participant's choice, submitted as a notebook and defended in a ten-minute oral check in which the participant reworks one step with new numbers

**Submission:** a link to a public GitHub repository with the capstone notebook, its data source and licence, and the notebook's outputs; then a ten-minute oral check: four minutes walking through the notebook, six reworking one step with new numbers the assessor gives.

| Rubric criterion (the notebook) | Weight |
|---|---:|
| Data matrix stated: source, licence, shape, centring and scaling, each choice justified | 15 |
| SVD computed and checked: a small case agrees with hand work; reconstruction error reported by rank | 25 |
| Rank chosen from the error curve and the singular values, with the trade-off stated | 20 |
| Leading components interpreted in the dataset's own terms, with their limits | 25 |
| Notebook runs from top to bottom; seed, data source and licence stated; outputs honest | 15 |

## Repository

```
course.json      the course manifest: outcomes, units, sessions, labs, capstone, tools, sources
slides/          one deck per week and the capstone deck; the engine builds them into the course site
labs/            the lab notebooks, their solutions, the problem sets and their checker, the helper
                 module (mth.py), the data generator (prepare_data.py), the tutor instruction
assessments/     the public practice questions (bank.json); exam items are never kept here
```

Build and check with the Academy engine: `node academy.mjs check`, `node academy.mjs build`.

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
<!-- /academy:sources -->

### National references

- SDAIA Data & AI Academic Programs Standards Framework (2026): bridging unit MSF-Linear at Explain level (one of the framework's post-bachelor bridging units, section 3.6): https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/Academic-Programs-Standards-Framework-Q3.pdf
