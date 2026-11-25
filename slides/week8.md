---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "week8"
day: 8
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/week8.html"
---

# Linear Algebra for Machine Learning

## Week 8: Low rank in practice, and working numerically

Low-rank models in recommenders and adapters, and code you can trust

#### [week8#title] Low rank in practice, and working numerically

*SDAIA Academy · Linear Algebra for Machine Learning*

Week 8: Low-rank models in recommenders and adapters, and code you can trust

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-101

#### [week8#agenda] Low rank in practice, and working numerically

*This week*

- **Session 1 · Low rank in practice**: Low-rank structure in recommenders and in model adaptation, vectorisation and broadcasting, floating-point limits, and checking code against hand computation

#### [week8#review-week7] Choosing a rank for an image

*Review · week 7*

**Question:** A 600 × 400 image has singular values that fall slowly: rank 300 still leaves 5% relative error. What should you conclude?

- A. A rank-300 copy is a good way to save space
- B. The SVD was computed wrongly
- C. This image has little low-rank structure; truncation is a poor compressor for it
- D. Centring the image would make rank 10 enough

### Session 1: Low rank in practice

#### [week8#s1] Low rank in practice

*Session 1*

- Predict held-out ratings with a truncated SVD and choose the rank
- Count the parameters of a low-rank weight update
- Find a broadcasting or rounding error by checking code against a hand case

#### [week8#recommenders] A ratings matrix is close to low rank because tastes are few

*Block A · recommenders*

- Rows are people, columns are items; most entries are missing
- A rank-k model: each person is k numbers of taste, each item k numbers of character
- A predicted rating is the dot product of the two
- Truncated SVD after filling the gaps is the simplest version; methods that fit only the observed entries do better

#### [week8#choose-rank-held-out] Choose the rank on ratings the model did not see

*Choosing k*

- **Training error.** Falls with every extra layer, down to zero at full rank. It cannot choose k.
- **Held-out error.** Falls, then rises once extra layers fit noise. Its lowest point is the rank to use.

> **The trap.** Choosing k on the test set and then reporting the test error flatters the model. With enough data, keep a third split for the final report.

#### [week8#adapters] A low-rank update adapts a large model by training two thin matrices

*Model adaptation*

W' = W + **B A**, with B d × r and A r × d, W frozen

d = 1,024, r = 8: **16,384** trainable numbers against 1,048,576: **1.56%**

- **Low-Rank Adaptation (LoRA).** The 2021 paper that introduced the method reports up to 10,000 times fewer trainable parameters on its largest model.

#### [week8#check-rank-choice] Which rank would you report?

*Checkpoint*

**Question:** Held-out errors for ranks 1 to 6 of a film-ratings model: 0.91, 0.86, 0.84, 0.85, 0.87, 0.90. Training errors fall from 0.88 to 0.61. Which rank?

- A. 6, with the lowest training error
- B. 3, with the lowest held-out error
- C. 1, the simplest model
- D. 4, the middle of the range

#### [week8#broadcasting-rules] Broadcasting lines shapes up from the right and stretches dimensions of 1

*Block B · vectorisation and broadcasting*

(3, 1) + (1, 2) gives (3, 2):

```python
# file: numpy
a = np.array([[1], [2], [3]])   # (3, 1)
b = np.array([[10, 20]])        # (1, 2)
a + b   # [[11, 21], [12, 22], [13, 23]]
```

1. Compare shapes from the last dimension
2. Equal sizes match; a size of 1 stretches
3. Anything else is an error

#### [week8#axis-bugs] A square test matrix hides an axis bug

*Silent errors*

- `M - M.mean(axis=1)` on an n × m matrix lines the row means up with the columns
- If n ≠ m it fails with an error; if n = m it runs and is wrong
- Use `keepdims=True`, or a shape that says what you mean: (n, 1)
- Test with rectangular shapes, such as 2 × 3, never only square ones

#### [week8#check-broadcast] Predict the shape

*Checkpoint*

**Question:** X has shape (100, 5). What does `X - X.mean(axis=0)` do?

- A. Subtracts each row's mean from that row
- B. Raises a shape error
- C. Subtracts one overall mean from every entry
- D. Subtracts each column's mean from that column

#### [week8#float-limits] A float keeps a fixed number of significant digits, about 7 or 16

*Block C · floating point*

| Expression | Result | Why |
| --- | --- | --- |
| 1e16 + 1 − 1e16 | 0.0 | At 10¹⁶ the gap between float64 numbers is 2 |
| float32: 1 + 10⁻⁸ == 1 | True | Machine epsilon for float32 is 1.19 × 10⁻⁷ |
| Subtracting nearly equal numbers | Most digits cancel | What is left is mostly rounding |

#### [week8#hand-checks] Trust code that agrees with a hand case and a trusted library, within a stated tolerance

*Checking code*

1. **1 · Hand case** a 2 × 2 or 2 × 3 you worked on paper
2. **2 · Properties** Qᵀ Q = I, A x = b, rows sum to zero
3. **3 · Random cases** against the library, with a tolerance

> **Tolerance.** Compare with `np.allclose(a, b, atol, rtol)`, never with ==. State the tolerance you chose and why.

#### [week8#check-float] Why does this test fail?

*Checkpoint*

**Question:** A test asserts `Q.T @ Q == np.eye(3)` for a correct QR and fails. What is wrong?

- A. Exact equality on floats: compare with a tolerance instead
- B. The QR routine has a bug
- C. The product should be Q @ Q.T
- D. Convert Q to float32 first

#### [week8#lab-8-1] Low-rank ratings and adapter updates, checked by hand

Lab 8.1 · 105 min

Predict held-out course ratings with a truncated SVD, count a low-rank update's parameters, and find a planted bug and a rounding trap against hand work.

[W8_1.ipynb](https://sdaiaacademy.github.io/fnd-mth-101/labs/W8_1.ipynb)

- [ ] Set up; broadcasting shapes by hand (10 min)
- [ ] Ratings: fill, centre, truncate, test (25 min)
- [ ] A low-rank weight update (15 min)
- [ ] Floating-point limits (15 min)
- [ ] Check a function against hand cases (15 min)
- [ ] Find and fix the planted bug (15 min)
- [ ] Write your capstone plan (10 min)
- [ ] ✱ Stretch: Fit the ratings by alternating least squares and compare

#### [week8#debrief-ratings] Rank 3 predicted held-out ratings best: 0.543 against 0.602 for the course mean

*Debrief · lab 8.1*

| Item | Value |
| --- | --- |
| Course mean | 0.6022 |
| Rank 1 | 0.5709 |
| Rank 3 | 0.5432 |
| Rank 6 | 0.5607 |
| Rank 10 | 0.5683 |

RMSE on 665 held-out ratings. The data were generated with three taste dimensions: the curve found them.

#### [week8#capstone-brief] A low-rank analysis of a public dataset of your choice, defended in ten minutes

*The capstone*

- A public image, ratings or embedding matrix, with a licence that allows sharing
- A notebook in a public repository: the data, the SVD checked by hand on a small case, error by rank, the rank you choose, the leading components interpreted
- AI-full: any assistant may help; you must explain and redo every step

- **The oral check.** Four minutes walking through the notebook, six reworking one step with new numbers the assessor gives you, by hand and in code.

#### [week8#wrap-up] Three things to keep

*Week 8 · wrap-up*

- **Low rank is a model.** Ratings and model updates are close to low rank; choose k on held-out data.
- **Shapes, then values.** Broadcasting can run and be wrong. Test with rectangular shapes.
- **Check with tolerance.** A hand case, properties and the library, compared with a stated tolerance.

#### [week8#next] The capstone and the second supervised test

*Next*

Problem set 8 (60 minutes, AI off for the first attempt): labs/problem-sets/PS8.ipynb. Then the second supervised test, and the capstone notebook and oral check.
