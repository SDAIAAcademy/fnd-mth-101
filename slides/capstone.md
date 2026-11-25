---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "capstone"
day: 9
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/capstone.html"
---

# Linear Algebra for Machine Learning

## Capstone: Capstone oral checks

Ten minutes each: four to walk through the notebook, six to rework one step with new numbers

#### [capstone#title] Capstone oral checks

*SDAIA Academy · Linear Algebra for Machine Learning*

Capstone: Ten minutes each: four to walk through the notebook, six to rework one step with new numbers

Data & AI Foundations Track · 8 weeks + capstone · 4 min talk and demo · 6 min Q&A · FND-MTH-101

#### [capstone#agenda] How the oral checks run

*Capstone*

- **Expectations and final preparation**: the rubric, the format, a last rehearsal
- **Oral checks**: ten minutes each: a walkthrough, then a step reworked with new numbers
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#what-you-submit] A notebook in a public repository, then ten minutes with the assessor

*The submission*

- A public image, ratings or embedding matrix, with its address and licence
- The matrix stated: rows, columns, shape, centring and scaling
- The SVD, checked on a small case by hand
- Relative error by rank, the rank chosen, the leading components interpreted

- **Submission.** The link to a public GitHub repository holding the notebook, its outputs and the data source. Nothing personal, nothing whose licence forbids sharing.

#### [capstone#rubric] The rubric

*How the notebook is scored*

| Item | Value |
| --- | --- |
| Data matrix stated and justified: source, licence, shape, centring, scaling | 15% |
| SVD checked by hand on a small case; error reported by rank | 25% |
| Rank chosen from the error curve, trade-off stated | 20% |
| Leading components interpreted, with their limits | 25% |
| Runs top to bottom; seed and source stated; outputs honest | 15% |

#### [capstone#assessment-overall] How the course is assessed

*The whole course*

| Assessed work | Weight |
| --- | --- |
| Weekly problem sets, auto-checked | 20% |
| Two supervised AI-off tests, on paper and at the keyboard | 40% |
| Capstone notebook, rubric-scored | 30% |
| Oral check | 10% |

Passing: 60% overall and at least 50% on each supervised test. Certificate: completion with assessed work.

#### [capstone#format] Ten minutes each: four to walk through, six to rework a step

*Format*

- **1 · The matrix.** What the rows and columns are, and why you centred or scaled.
- **2 · The result.** The error by rank, the rank chosen, what the leading components show.
- **3 · The rework.** The assessor gives new numbers for one step; you redo it by hand and check it in code.

> **Peer feedback.** After each check, everyone writes one strength, one question and one suggestion.

#### [capstone#rework-example] A rework looks like this: new numbers, the same step, by hand and in code

*Rehearse the rework*

"Your singular values were replaced by 12, 5, 2 and 1. What relative error does rank 2 give?"

By hand: √(2² + 1²) / √(12² + 5² + 2² + 1²) = √5 / √174 ≈ **0.170**

```python
# file: numpy
s = np.array([12, 5, 2, 1.0])
tail = (s[2:]**2).sum()
np.sqrt(tail / (s**2).sum())
# 0.1695
```

#### [capstone#running-order] Who presents when

*Running order*

| # | Participant | Dataset |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final preparation · 20 minutes

*Before the first check*

Run your notebook from the top once more, and rehearse one rework with a neighbour.

#### [capstone#now-presenting] Walkthrough

*Now presenting*

##### Rework with new numbers

#### [capstone#review] What the capstones showed

*Trainer review*

- **Strongest patterns.** What several notebooks got right.
- **Common gaps.** What most notebooks would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#what-next] Keep going

*After the course*

- **Your notebook stays yours:** keep it reproducible, and add the next analysis you named today
- **Next Academy courses:** FND-MTH-102 Calculus and Optimisation for Machine Learning, and FND-AIC-101 How Machine Learning Works, where the least squares and projections of this course return as models

#### [capstone#thanks] Thank you

*SDAIA Academy*

Linear Algebra for Machine Learning
