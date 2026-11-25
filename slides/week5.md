---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "week5"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/week5.html"
---

# Linear Algebra for Machine Learning

## Week 5: Least squares and conditioning

Least squares is a projection; conditioning decides how far to trust it

#### [week5#title] Least squares and conditioning

*SDAIA Academy · Linear Algebra for Machine Learning*

Week 5: Least squares is a projection; conditioning decides how far to trust it

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-101

#### [week5#agenda] Least squares and conditioning

*This week*

- **Session 1 · Least squares and conditioning**: Least squares as projection, the normal equations, the QR route, condition numbers and what nearly collinear features do to a fit

#### [week5#review-week4] A property of every projection

*Review · week 4*

**Question:** P projects onto the span of the columns of A, and r = v − P v. Which statement always holds?

- A. ‖r‖ = ‖P v‖
- B. P has an inverse
- C. P r = r
- D. Aᵀ r = 0

### Session 1: Least squares and conditioning

#### [week5#s1] Least squares and conditioning

*Session 1*

- Fit a line by the normal equations by hand
- Fit by QR and compare residuals and coefficients with the normal equations
- Diagnose an ill-conditioned or rank-deficient system from its numbers

#### [week5#no-exact-solution] With more data points than unknowns, there is usually no exact solution

*Block A · least squares*

- Three points, a line with two coefficients: X β = y has no solution unless the points line up
- **Least squares** chooses β to make ‖y − X β‖ as small as possible
- X β ranges over the span of the columns, so the best X β is the **projection** of y onto it

> **So:** the residual y − X β is orthogonal to every column: **Xᵀ(y − X β) = 0**.

#### [week5#normal-equations] XᵀX β = Xᵀy: least squares by hand

*The normal equations*

Points (−1, 1), (0, 1), (1, 4); X = [1, x]

XᵀX = [[3, 0], [0, 2]], Xᵀy = (6, 3)

**β = (2, 1.5)**: y ≈ 2 + 1.5x

- **Check the residual.** Fitted (0.5, 2, 3.5), residual (0.5, −1, 0.5). It sums to zero and is orthogonal to x: Xᵀr = (0, 0).

#### [week5#qr-route] With X = QR, least squares becomes a triangular solve: R β = Qᵀy

*The QR route*

```python
# file: numpy
Q, R = np.linalg.qr(X)
beta = solve_triangular(R, Q.T @ y)

# the library's own route (SVD)
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

- Q Qᵀ y is the projection of y; Qᵀy its coordinates
- XᵀX is never formed
- Back substitution, as in week 3

#### [week5#check-least-squares] Which fit is the least-squares fit?

*Checkpoint*

**Question:** Two candidate fits for the same data. Fit F has residual (1, −2, 1) and fit G has residual (2, −1, −1). The columns of X are (1, 1, 1) and (−1, 0, 1). Which can be the least-squares fit?

- A. G, because its residual is more even
- B. F, because its residual is orthogonal to both columns
- C. Both, since their residuals have the same length
- D. Neither, because neither residual is zero

#### [week5#condition-number] The condition number says how much a small change in the data can move the answer

*Block B · conditioning*

A = [[1, 1], [1, 1.0001]], cond(A) ≈ **4 × 10⁴**

| b | Solution of A x = b |
| --- | --- |
| (2, 2.0001) | (1, 1) |
| (2, 2.0002) | (0, 2) |

- **Rule of thumb.** About log₁₀ cond decimal digits of accuracy are at risk. Float64 carries about 16.

#### [week5#squaring] Forming XᵀX squares the condition number

*Why QR*

- **4 × 10⁴** cond(A) for the matrix of the last slide
- **1.6 × 10⁹** cond(AᵀA): about 9 of 16 digits at risk

> **The normal equations** work with cond(X)²; QR works with cond(X). For well-conditioned data both are fine. As columns approach collinearity, the normal equations fail first.

#### [week5#collinear-features] Nearly collinear features make coefficients unstable and fitted values stable

*In data*

- Two columns that nearly repeat each other: many β give almost the same X β
- The **fit** barely changes when the data is nudged
- The **split** of the effect between the two columns can swing widely

> **For the report.** Do not read the coefficient of one of two nearly collinear features as "the effect" of that feature.

#### [week5#check-conditioning] Diagnose a fit

*Checkpoint*

**Question:** A fit has cond(X) = 10⁷. Refitting after rounding the data to four decimals changes two coefficients by 30% and the fitted values by 0.01%. What is the diagnosis?

- A. The rounding was too coarse; the model is fine
- B. The model is stable, since its predictions hardly moved
- C. Ill-conditioned: nearly collinear columns; trust the fit, not those two coefficients
- D. Rank-deficient: one column must be removed before any fit

#### [week5#scale-and-centre] Some ill-conditioning is only a matter of units

*Block C · what helps*

| Cause | What helps |
| --- | --- |
| Columns on very different scales (metres and millimetres) | Divide each column by its standard deviation |
| A large mean far from zero, beside the intercept | Centre each column |
| Columns that nearly repeat each other | Drop or combine one; scaling does not help |
| Exactly repeated columns | Drop one: the matrix is rank-deficient |

#### [week5#three-routes] Prefer QR or the library's least-squares solver; check the condition number

*In practice*

- **Normal equations.** Fast and simple; squares the condition number. Fine when cond(X) is small.
- **QR.** The default for full-rank problems: works with cond(X).
- **SVD (lstsq).** Slowest; handles rank deficiency by ignoring tiny singular values. Week 7.

#### [week5#check-route] Which route for this fit?

*Checkpoint*

**Question:** A feature matrix has cond(X) = 10⁶ in float64. You need coefficients accurate to at least six digits. Which route?

- A. The normal equations: they are fastest
- B. QR, which puts about 6 digits at risk and leaves about 10
- C. The normal equations in float32 to save memory
- D. Invert XᵀX and multiply by Xᵀy

#### [week5#lab-5-1] Least squares by normal equations and by QR on nearly collinear data

Lab 5.1 · 105 min

Fit course hours on nearly collinear features two ways, compare residuals and condition numbers, and diagnose five cases.

[W5_1.ipynb](https://sdaiaacademy.github.io/fnd-mth-101/labs/W5_1.ipynb)

- [ ] Set up; a line by hand with the normal equations (15 min)
- [ ] Fit with the normal equations (15 min)
- [ ] Fit with QR and with the library (15 min)
- [ ] Condition numbers of X and of XᵀX (15 min)
- [ ] Nudge the data and watch the coefficients (15 min)
- [ ] Compare the residuals (10 min)
- [ ] Diagnose five practice cases (15 min)
- [ ] Explain the difference (5 min)
- [ ] ✱ Stretch: Centre and scale the columns, then measure again

#### [week5#debrief-agreement] On the course data the two routes agreed to 10⁻¹², because cond(X) was only 2,213

*Debrief · lab 5.1*

| Video minutes = 7.5 × count + noise of | cond(X) | Largest \|β normal − β QR\| |
| --- | --- | --- |
| 0.6 (as in the data) | 2.23 × 10³ | 2.6 × 10⁻¹¹ |
| 0.06 | 2.23 × 10⁴ | 4.2 × 10⁻⁹ |
| 0.006 | 2.23 × 10⁵ | 1.7 × 10⁻⁶ |

Each tenfold rise in cond(X) widened the gap a hundredfold or more: the square at work.

#### [week5#debrief-sensitivity] A 0.01-hour nudge moved the video coefficients, not the fit

*Debrief · lab 5.1*

| Coefficient | Fit | Change after the nudge |
| --- | --- | --- |
| video_count | 0.4228 | +0.0075 |
| video_minutes | 0.0469 | −0.0010 |
| quizzes | 0.9294 | −0.0002 |

- **In float32.** The normal equations gave video_count 0.4308 and video_minutes 0.0458; QR kept 0.4228 and 0.0469. Residual norms differed only in the sixth digit.

#### [week5#wrap-up] Three things to keep

*Week 5 · wrap-up*

- **Least squares is projection.** The residual is orthogonal to the columns: Xᵀ(y − X β) = 0.
- **XᵀX squares trouble.** cond(XᵀX) = cond(X)². Use QR or the library's solver.
- **Fits outlast coefficients.** Near-collinearity leaves predictions stable and individual coefficients unreliable.

#### [week5#next] Week 6: Eigenvectors and link ranking

*Next*

Problem set 5 (60 minutes, AI off for the first attempt): labs/problem-sets/PS5.ipynb.
