---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "week3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/week3.html"
---

# Linear Algebra for Machine Learning

## Week 3: Span, rank and linear systems

When columns repeat each other, and what that does to a system of equations

#### [week3#title] Span, rank and linear systems

*SDAIA Academy · Linear Algebra for Machine Learning*

Week 3: When columns repeat each other, and what that does to a system of equations

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-101

#### [week3#agenda] Span, rank and linear systems

*This week*

- **Session 1 · Span, rank and linear systems**: Span, independence, basis and rank, and solving linear systems by elimination, with the inverse and its cost

#### [week3#review-week2] Cheaper brackets

*Review · week 2*

**Question:** A is 100 × 2,000, B is 2,000 × 2,000 and x is a vector of length 2,000. You need A B x. Which order costs fewer FLOPs?

- A. Either: the answer is the same, so the cost is the same
- B. (A B) x
- C. A (B x)
- D. Whichever order NumPy chooses, since it optimises the brackets

### Session 1: Span, rank and linear systems

#### [week3#s1] Span, rank and linear systems

*Session 1*

- Decide whether a set of vectors is independent and state the rank of a matrix
- Solve a three-by-three system by elimination by hand and with a solver
- Explain what a rank-deficient feature table does to a solve

#### [week3#span] The span of some vectors is everything you can build from them

*Block A · span*

- A **combination** of u and v is a u + b v, for any numbers a and b
- The **span** is the set of all combinations: a line, a plane, or more
- (1, 2) and (2, 4) span only a line: the second is twice the first
- (1, 0) and (0, 1) span the whole plane

> **For data.** The columns of a feature table span the set of predictions any linear model on them can make.

#### [week3#independence] Vectors are independent when none is a combination of the others

*Independence*

- **Independent.** The only combination that gives zero is all coefficients zero. Each vector adds a new direction.
- **Dependent.** Some combination with non-zero coefficients gives zero: 2(1, 2) − (2, 4) = (0, 0). One vector repeats the others.

> **A basis** of a span is an independent set that spans it. Every basis of the same span has the same size: that size is the dimension.

#### [week3#rank] The rank is the number of independent columns, and also of independent rows

*Rank*

| Course | Hours | Minutes | Lessons |
| --- | --- | --- | --- |
| Course 1 | 2 | 120 | 8 |
| Course 2 | 3 | 180 | 10 |
| Course 3 | 1 | 60 | 5 |
| Course 4 | 4 | 240 | 12 |

- Minutes = 60 × hours exactly
- Three columns, **rank 2**
- Singular values 329.18, 2.51 and **0**: one direction is missing

#### [week3#check-rank] How many features are really there?

*Checkpoint*

**Question:** A table has columns: price, tax (15% of price), price with tax, and a discount code that varies freely. What is its rank, if there are many rows?

- A. 4
- B. 2
- C. 3
- D. 1

#### [week3#systems-as-columns] Solving A x = b asks which combination of A's columns makes b

*Block B · linear systems*

- **One solution.** b is in the span and the columns are independent.
- **Infinitely many.** b is in the span and the columns are dependent: one solution plus any null-space vector.
- **None.** b is outside the span. Week 5 finds the closest combination instead.

#### [week3#elimination] Elimination subtracts rows until the system is a staircase

*Elimination*

| Step | Rows of [A \| b] |
| --- | --- |
| Start | [2, 1, 1 \| 4], [4, 3, 3 \| 10], [8, 7, 9 \| 24] |
| Row 2 − 2 × row 1; row 3 − 4 × row 1 | [2, 1, 1 \| 4], [0, 1, 1 \| 2], [0, 3, 5 \| 8] |
| Row 3 − 3 × row 2 | [2, 1, 1 \| 4], [0, 1, 1 \| 2], [0, 0, 2 \| 2] |

Back substitution from the bottom: z = 1, then y = 1, then x = 1.

#### [week3#pivoting] A zero or tiny pivot breaks elimination; swapping rows fixes most cases

*Pivots*

- The **pivot** is the number you divide by at each step
- A zero pivot with nothing to swap in means a dependent column: the matrix is **singular**
- A tiny pivot magnifies rounding; **partial pivoting** swaps in the row with the largest entry
- The number of non-zero pivots is the rank

#### [week3#check-systems] What does a zero pivot tell you?

*Checkpoint*

**Question:** Elimination on a 4 × 4 matrix finds no non-zero pivot in the third column, even after looking for a row to swap. What follows?

- A. The computation needs more precision
- B. The system has exactly one solution, found by skipping the column
- C. The columns are dependent; the system has no solution or infinitely many
- D. The right-hand side b must be zero

#### [week3#inverse] The inverse undoes a map, and exists only when nothing was lost

*Block C · the inverse*

[[4, 7], [2, 6]] has determinant 4 × 6 − 7 × 2 = 10, so

**inverse = [[0.6, −0.7], [−0.2, 0.4]]**

Check: multiply them and get the identity.

- **No inverse.** A singular matrix squashes some direction to zero. Nothing can bring it back: two inputs give the same output.

#### [week3#solve-not-invert] To solve a system, call a solver; do not invert and multiply

*In code*

```python
# file: numpy
x = np.linalg.solve(A, b)      # yes
x = np.linalg.inv(A) @ b       # avoid
```

- Solving does less work than inverting
- It rounds less, and the gap grows with the condition number
- You rarely need the inverse itself

#### [week3#floating-point-first] On a computer, "zero" means "small compared with rounding"

*Rounding*

- A float64 number holds about 16 significant digits
- Exact dependence shows as a singular value of rounding size, about 10⁻¹⁶ times the largest or less, not always exactly 0
- `np.linalg.matrix_rank` counts singular values above a tolerance set from the matrix's size and largest value

> **Near, not exact.** A column that almost repeats another keeps the rank full and still makes a solve unstable. Weeks 4 and 5 measure that.

#### [week3#check-inverse] Can this table be inverted?

*Checkpoint*

**Question:** A 3 × 3 matrix has rows (1, 2, 3), (0, 1, 4) and (1, 3, 7). Does it have an inverse?

- A. Yes: no row is all zeros
- B. Yes: no two rows are equal
- C. It depends on the vector b it is used with
- D. No: the third row is the sum of the first two

#### [week3#lab-3-1] Rank and linear systems on a feature table

Lab 3.1 · 105 min

Solve a small system by elimination, find the repeated column of the learner table from its rank, and compare solving with inverting.

[W3_1.ipynb](https://sdaiaacademy.github.io/fnd-mth-101/labs/W3_1.ipynb)

- [ ] Set up; elimination by hand, then a solver (20 min)
- [ ] Span: is a vector a combination of two others? (15 min)
- [ ] The rank of the table, and the column it repeats (20 min)
- [ ] What the repeated column does to a solve (15 min)
- [ ] A basis: drop the column, check the rank (10 min)
- [ ] Solve against invert: residuals and time (15 min)
- [ ] Record which columns to keep (10 min)
- [ ] ✱ Stretch: Read the repeated column off the null space of the full table

#### [week3#debrief-rank] Five columns, rank 4: total actions repeats the other counts

*Debrief · lab 3.1*

| Quantity | Value |
| --- | --- |
| Singular values of the table | 2951.22, 89.61, 32.87, 6.02, 0 |
| Condition number of XᵀX, five columns | 1.39 × 10¹⁷ |
| Condition number of XᵀX, four columns | 3.06 × 10⁵ |

- total_actions = videos + quizzes + posts, exactly
- On the build laptop, solve and inverse both stopped: "Singular matrix"
- Keep four columns; drop the sum

#### [week3#wrap-up] Three things to keep

*Week 3 · wrap-up*

- **Rank counts directions.** Columns that repeat others add no information, and a model cannot separate their effects.
- **Solutions depend on the span.** One, none or infinitely many: decided by rank and by whether b is reachable.
- **Solve, do not invert.** Call a solver. The inverse is a concept more than a step.

#### [week3#next] Week 4: Orthogonal projection and QR

*Next*

Problem set 3 (60 minutes, AI off for the first attempt): labs/problem-sets/PS3.ipynb. The first supervised test follows week 4.
