---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "week4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/week4.html"
---

# Linear Algebra for Machine Learning

## Week 4: Orthogonal projection and QR

The nearest point in a subspace, and a basis that makes it easy

#### [week4#title] Orthogonal projection and QR

*SDAIA Academy · Linear Algebra for Machine Learning*

Week 4: The nearest point in a subspace, and a basis that makes it easy

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-101

#### [week4#agenda] Orthogonal projection and QR

*This week*

- **Session 1 · Orthogonal projection and QR**: Orthogonal projection onto a line and onto a subspace, projection matrices, and the QR factorisation by Gram-Schmidt

#### [week4#review-week3] Reading a rank

*Review · week 3*

**Question:** A 1,000 × 6 table of features has rank 5. What can you conclude?

- A. Five of the 1,000 rows are independent, the rest repeat them
- B. One column has missing values
- C. One column is a combination of the others; a model cannot separate all six effects
- D. The sixth column must be deleted

### Session 1: Orthogonal projection and QR

#### [week4#s1] Orthogonal projection and QR

*Session 1*

- Project a vector onto a line and onto a plane by hand and in code
- Check the properties of a projection matrix
- Build a QR factorisation by Gram-Schmidt and check it against the library

#### [week4#nearest-point] The projection is the nearest point on the line, and the residual meets it at a right angle

*Block A · projection onto a line*

**p = (b · a / a · a) a**

b = (2, 4), a = (3, 1): b · a = 10, a · a = 10, so **p = (3, 1)**.

Residual b − p = (−1, 3), and (−1, 3) · (3, 1) = 0.

- **Why a right angle.** If the residual leaned along the line, sliding the point that way would bring it closer. The nearest point leaves only the orthogonal part.

#### [week4#projection-onto-plane] Onto the span of several columns, the projection matrix is A (AᵀA)⁻¹ Aᵀ

*Onto a subspace*

A with columns (1, 0, 1) and (0, 1, 1); v = (1, 2, 0)

P v = **(0, 1, 1)**, residual (1, 1, −1)

The residual is orthogonal to both columns: (1, 1, −1) · (1, 0, 1) = 0 and · (0, 1, 1) = 0.

- Condition: **Aᵀ(v − P v) = 0**, the normal equations
- Compute it with a solve, never an inverse

#### [week4#projection-properties] Projecting twice changes nothing, and a projection matrix is symmetric

*Two properties*

- **P² = P.** A point already on the subspace stays where it is.
- **Pᵀ = P.** An orthogonal projection treats both sides alike. A projection that is not symmetric projects at a slant.

> **Use them as checks.** If your P fails either, the code is wrong. The lab and the problem set test both.

#### [week4#check-projection] Where does the projection land?

*Checkpoint*

**Question:** Project b = (4, 2) onto the line through a = (1, 0). Which is the projection, and why?

- A. (0, 2), the part along the second axis
- B. (4, 0), since b · a / a · a = 4
- C. (4), the number b · a
- D. (0.894, 0), the unit vector of b on the axis

#### [week4#orthonormal-bases] With an orthonormal basis, projecting is just dot products

*Block B · orthonormal bases*

- **Orthonormal**: unit length, pairwise orthogonal: QᵀQ = I
- Projection onto their span: **P = Q Qᵀ**, no solve needed
- The coordinates of v are simply the dot products qᵢ · v

> **The plan.** Turn any independent columns into orthonormal ones with the same span. That is Gram-Schmidt, and the bookkeeping is the QR factorisation.

#### [week4#gram-schmidt] Gram-Schmidt removes from each column what earlier columns already explain

*Gram-Schmidt*

| Columns (3, 4) and (1, 0) | Result |
| --- | --- |
| q₁ = (3, 4) / 5 | (0.6, 0.8) |
| Coefficient r₁₂ = q₁ · (1, 0) | 0.6 |
| What is left: (1, 0) − 0.6 q₁ | (0.64, −0.48), length 0.8 |
| q₂ = what is left / 0.8 | (0.8, −0.6) |

#### [week4#qr] A = Q R: Q holds the orthonormal columns, R the coefficients

*The factorisation*

For the example: **R = [[5, 0.6], [0, 0.8]]**

Column 1 of A = 5 q₁; column 2 = 0.6 q₁ + 0.8 q₂.

```python
# file: numpy
Q, R = np.linalg.qr(A)
np.allclose(Q.T @ Q, np.eye(2))   # True
np.allclose(Q @ R, A)             # True
```

> **Signs.** A library may return −q and −r for some columns. Both are correct: QR is unique only up to those signs.

#### [week4#check-qr] Which check proves your QR is right?

*Checkpoint*

**Question:** Your function returns Q (5 × 3) and R (3 × 3) for a 5 × 3 matrix A. Which set of checks is enough?

- A. QᵀQ = I
- B. Q R = A
- C. Q equals np.linalg.qr(A)[0] entry by entry
- D. QᵀQ = I, Q R = A and R upper triangular

#### [week4#rounding-in-gs] When columns are nearly parallel, subtracting shadows leaves mostly rounding

*Block C · rounding*

- Two columns at a tiny angle: almost all of the second is its shadow on the first
- What is left is small, and rounding errors are a large part of it
- **Classical** Gram-Schmidt projects the original column each time; **modified** projects the updated one, and keeps more accuracy
- Libraries use Householder reflections, which keep Q orthonormal to rounding level

#### [week4#why-orthogonality-matters] A Q that is not orthonormal gives wrong projections without any error message

*Why it matters*

- **Silent failure.** Q Qᵀ v is a projection only if QᵀQ = I. Code that skips the check returns numbers that look fine.
- **The habit.** After any factorisation, measure max |QᵀQ − I| and max |QR − A|. Both should be near 10⁻¹⁵ in float64.

#### [week4#check-rounding] A measured loss of orthogonality

*Checkpoint*

**Question:** For a QR of a feature matrix you measure max |QᵀQ − I| = 0.02. What is the soundest next step?

- A. Suspect nearly dependent columns, check the condition number, and use a Householder QR
- B. Accept it: 0.02 is a small number
- C. Use fewer rows so the factorisation is faster and more accurate
- D. Normalise each column of Q again to length 1

#### [week4#lab-4-1] Projections and QR by Gram-Schmidt

Lab 4.1 · 105 min

Project weekly minutes onto two features, build QR by Gram-Schmidt, and watch orthogonality fail on nearly dependent columns.

[W4_1.ipynb](https://sdaiaacademy.github.io/fnd-mth-101/labs/W4_1.ipynb)

- [ ] Set up; project onto a line by hand, then in code (15 min)
- [ ] The projection matrix and its residual (15 min)
- [ ] Project weekly minutes onto two features (15 min)
- [ ] Classical Gram-Schmidt QR (20 min)
- [ ] Check against the library (10 min)
- [ ] Nearly dependent columns: classical against modified (20 min)
- [ ] Record what this means for week 5 (10 min)
- [ ] ✱ Stretch: Explain the columns whose signs differ from the library's Q

#### [week4#debrief-orthogonality] Classical Gram-Schmidt lost half its orthogonality at a condition number of 10⁸

*Debrief · lab 4.1*

| ε | Condition number | Classical | Modified | Householder |
| --- | --- | --- | --- | --- |
| 10⁻⁴ | 1.73 × 10⁴ | 2.3 × 10⁻⁹ | 2.8 × 10⁻¹³ | 2.2 × 10⁻¹⁶ |
| 10⁻⁷ | 1.73 × 10⁷ | 1.3 × 10⁻² | 1.6 × 10⁻⁹ | 4.4 × 10⁻¹⁶ |
| 10⁻⁸ | 1.73 × 10⁸ | 5.0 × 10⁻¹ | 7.1 × 10⁻⁹ | 2.2 × 10⁻¹⁶ |

Loss of orthogonality, max |QᵀQ − I|. On the learner columns (condition number about 553) all three stayed near 10⁻¹⁴.

#### [week4#wrap-up] Three things to keep

*Week 4 · wrap-up*

- **Nearest means orthogonal.** The residual of a projection is orthogonal to the subspace: Aᵀ(v − P v) = 0.
- **QR is projection bookkeeping.** Q spans the same space with orthonormal columns; R records how to rebuild A.
- **Measure the loss.** Check max |QᵀQ − I| after every factorisation; prefer Householder.

#### [week4#next] Week 5: Least squares and conditioning

*Next*

Problem set 4 (60 minutes, AI off for the first attempt): labs/problem-sets/PS4.ipynb. Then the first supervised test, on weeks 1 to 4.
