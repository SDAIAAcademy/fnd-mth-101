---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/"
---

# Linear Algebra for Machine Learning

Compute vectors, projections, least squares and the SVD by hand and in array code, and judge when a numerical result can be trusted

Participants compute each idea of linear algebra twice: by hand on small cases and in array code on real data. They treat data as vectors, matrices as transformations, least squares as projection, and eigen and singular value decompositions as ways to find structure. Labs use public embeddings and images and a synthetic link graph. The capstone is a low-rank analysis of a public dataset, defended orally.

- **Audience:** Beginners. Career changers and non-computing graduates closing bridging gaps before advanced study, and computing graduates refreshing the mathematics behind models; a placement test lets participants skip any course whose content they already command.
- **Prerequisites:** School algebra: solving equations and reading graphs of functions; Writes short Python programs with lists, loops and functions (FND-PRG-101 or equivalent)
- **Case study:** The online learning platform (fictional): An unnamed online platform of short courses wants better search, cheaper scoring, honest fits and a fair ranking of its help centre. Its synthetic data (60 help articles, 400 learners, 120 courses, a 40-page link graph, 3,596 ratings) is generated with a fixed seed. The labs also use public GloVe word vectors, downloaded at run time, and two datasets bundled with the libraries: the handwritten digits in scikit-learn and a CC0 test image in scikit-image. Nothing public is copied into the course.

## Outcomes

- LO1 (Awareness): Explain the geometric meaning of vectors, dot products, norms, projections and matrix products, given two- and three-dimensional examples
- LO2 (Awareness): Describe what rank, eigenvalues and singular values reveal about a matrix and name one machine learning use of each, given covariance and data matrices
- LO3 (Awareness): Identify ill-conditioned or rank-deficient systems and explain their effect on a least-squares fit, given five numerical cases
- LO4 (Applied): Compute products, inverses, projections and decompositions by hand for small matrices and in array code for large ones, given the weekly problem sets
- LO5 (Applied): Implement least squares through QR factorisation and compare it with the normal equations, given synthetic data with nearly collinear features
- LO6 (Applied): Apply the singular value decomposition to build low-rank approximations and principal components, given a public image, ratings or embedding matrix

## Week 1: Vectors and similarity

Data as vectors, and the angle between them as similarity

#### [week1#title] Vectors and similarity

*SDAIA Academy · Linear Algebra for Machine Learning*

Week 1: Data as vectors, and the angle between them as similarity

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-101

#### [week1#agenda] Vectors and similarity

*This week*

- **Session 1 · Vectors, dot products and similarity**: Vectors as data points and directions, the dot product, norms and cosine similarity, and what distance means in high dimensions

#### [week1#how-the-course-runs] Every idea is computed twice: by hand, then in code

*Eight weeks*

- **By hand.** A two- or three-dimensional case on paper, small enough to see the geometry.
- **In code.** The same operation in NumPy, on data with thousands of numbers.
- **Checked.** The two must agree within a tolerance. When they do not, one of them is wrong.

| Assessed work | Weight |
| --- | --- |
| Weekly problem sets, auto-checked | 20% |
| Two supervised AI-off tests, on paper and at the keyboard | 40% |
| Capstone notebook, rubric-scored | 30% |
| Oral check | 10% |

#### [week1#setup-check] One command proves your setup

*Setup check*

From the course folder, in a terminal:

```bash
# file: terminal
cd labs
uv run python -c "import numpy, scipy, sklearn; print(numpy.__version__)"
```

A version number means you are ready. In a free hosted notebook, the first cell of each lab installs what it needs.

- **What the course assumes.** Python with lists, loops and functions (FND-PRG-101 or equivalent), and school algebra. Nothing of FND-PRG-101 is taught again here.

### Session 1: Vectors, dot products and similarity

#### [week1#s1] Vectors, dot products and similarity

*Session 1*

- Compute a dot product, a norm and a cosine by hand and in NumPy
- Explain why cosine compares directions and ignores length
- Describe how distances behave as the dimension grows

#### [week1#point-and-arrow] A vector is a point and an arrow at the same time

*Block A · vectors*

- **A point: a data record.** A learner's week as (videos, quizzes, posts) = (1, 2, 2). Each coordinate is one measured quantity.
- **An arrow: a direction.** The same numbers drawn from the origin. Its direction says the mix of activity; its length says how much.

> **Order matters.** (1, 2, 2) and (2, 1, 2) hold the same numbers and are different vectors: the coordinates are named positions.

#### [week1#add-and-scale] Adding vectors adds coordinates; scaling stretches the arrow

*Two operations*

| Operation | Example | Result | Geometry |
| --- | --- | --- | --- |
| Sum | (1, 2, 2) + (2, 1, 2) | (3, 3, 4) | Head to tail |
| Scale | 3 × (1, 2, 2) | (3, 6, 6) | Same direction, three times as long |
| Difference | (2, 1, 2) − (1, 2, 2) | (1, −1, 0) | The arrow from the first point to the second |

In NumPy each line is one expression on arrays: `x + y`, `3 * x`, `y - x`.

#### [week1#norm-is-length] The norm is the length of the arrow

*The norm*

**‖x‖ = √(x₁² + x₂² + x₃²)**, Pythagoras in three dimensions.

For x = (1, 2, 2): ‖x‖ = √(1 + 4 + 4) = √9 = **3**.

The distance between two points is the norm of their difference: ‖y − x‖ = ‖(1, −1, 0)‖ = √2 ≈ 1.41.

- **Units.** A norm adds squares of coordinates. If one coordinate is minutes and another is counts, the minutes dominate. Scale before you measure.

#### [week1#check-norm-units] Which learner is nearest?

*Checkpoint*

**Question:** Learners are vectors of (video minutes, quizzes). Learner K is (300, 2). Which is nearest to K by Euclidean distance: M (305, 9), N (330, 2) or P (290, 2)?

- A. P, because it matches K exactly on quizzes
- B. N, because its only difference is in one coordinate
- C. M, at a distance of about 8.6
- D. None: minutes and quizzes cannot be combined in one distance

#### [week1#dot-product] The dot product multiplies matching coordinates and adds them

*Block B · the dot product*

x = (1, 2, 2), y = (2, 1, 2)

**x · y = 1×2 + 2×1 + 2×2 = 8**

And x · x = 1 + 4 + 4 = 9 = ‖x‖²: the norm is a dot product with itself.

```python
# file: numpy
x = np.array([1, 2, 2])
y = np.array([2, 1, 2])
x @ y              # 8
np.sqrt(x @ x)     # 3.0
```

#### [week1#dot-is-angle] x · y = ‖x‖ ‖y‖ cos θ: the dot product measures an angle

*Geometry*

For our pair: cos θ = 8 / (3 × 3) = **0.889**, so θ ≈ **27.3°**.

The **cosine similarity** is that cosine: the dot product of the two vectors after each is scaled to length 1.

**0.889** cosine of (1, 2, 2) and (2, 1, 2)

#### [week1#sign-of-dot] The sign of a dot product says acute, right or obtuse

*Reading the sign*

| Pair | Dot product | Angle | Reading |
| --- | --- | --- | --- |
| (1, 0) and (1, 1) | 1 | 45° | Pointing the same way, partly |
| (1, 0) and (0, 1) | 0 | 90° | Orthogonal: nothing in common |
| (1, 0) and (−1, 1) | −1 | 135° | Pointing partly opposite |

> **Orthogonal** means dot product zero. It comes back in every week from week 4 on.

#### [week1#cosine-ignores-length] Cosine ignores length; the dot product does not

*Length or direction*

- **x and 3x.** x · 3x = 27, larger than x · y = 8. Yet cos(x, 3x) = 1: the same direction exactly.
- **Why a search cares.** A long document has a long vector. Ranking by raw dot product would favour length over meaning. Cosine compares only direction.

#### [week1#many-dots-at-once] A matrix times a vector is many dot products at once

*At scale*

```python
# file: numpy
D = np.array([[1, 2, 2],
              [2, 1, 2],
              [3, 6, 6]], float)
N = D / np.linalg.norm(D, axis=1, keepdims=True)
q = np.array([1, 2, 2]) / 3
N @ q      # [1.0, 0.889, 1.0]
```

- Each row of **D** is one item; normalise every row to length 1
- **N @ q** gives the cosine of the question with every row
- The best match is the largest entry: `np.argmax`

#### [week1#check-dot-or-cosine] Dot product or cosine?

*Checkpoint*

**Question:** A question vector q = (1, 1). Two help articles: A = (4, 4) and B = (1, 1.2). Which statement is right?

- A. A is closer in direction, since q · A = 8 beats q · B = 2.2
- B. A and q differ in direction, since A is four times longer
- C. Both measures rank A first, so the choice never matters
- D. A has cosine 1 with q and B a little less; the dot product also rewards A's length

#### [week1#embeddings] An embedding places words so that related words point the same way

*Block C · embeddings*

- A model reads a large text collection and gives each word a vector
- Words used in similar contexts get vectors at small angles
- The course uses **GloVe**: 400,000 words, 50 numbers each, trained on 6 billion tokens of encyclopedia and news text

- **Licence.** The pre-trained GloVe vectors are released under the Public Domain Dedication and License. The lab downloads them; nothing is copied into the course.

#### [week1#sentence-as-mean] Here a sentence vector is the mean of its word vectors

*From words to sentences*

1. **1 · Split** "video will not play" into four words
2. **2 · Look up** four GloVe vectors of 50 numbers
3. **3 · Average** one vector of 50 numbers for the sentence

> **A baseline, not a sentence model.** The mean ignores word order and is pulled by frequent words. It is the simplest public sentence embedding, and the lab measures how far it gets.

#### [week1#the-cone] When every vector shares a big common part, every cosine looks high

*A shared direction*

| Toy vectors | Pair cosines |
| --- | --- |
| (6, 4), (4, 6), (6, 6) | 0.923, 0.981, 0.981 |
| After subtracting their mean | −0.800, −0.316, −0.316 |

- The three share the part (5, 5); their differences are small
- **Centring** subtracts the mean vector and leaves the differences
- After centring, angles describe how the items differ

#### [week1#high-dimensions] In many dimensions, the nearest and the farthest point come close

*High dimensions*

| Item | Value |
| --- | --- |
| 2 dimensions | 58.69 |
| 10 dimensions | 3.49 |
| 100 dimensions | 0.46 |
| 1,000 dimensions | 0.13 |

Contrast = (farthest − nearest) / nearest, from one random point to 499 others with independent standard normal coordinates.

#### [week1#associations] Vectors learn the associations of the text they were trained on

*What the vectors learnt*

The **lean** of a word w between two word lists:

lean(w) = mean of cos(w, female word) − cos(w, male word), over pairs such as she/he and woman/man.

Positive: closer to the female words. Zero: no lean.

- **Why it matters.** A search that matches a learner's free-text goal to courses through these vectors inherits their leans. The lab measures them; whether a search treats learners differently must be tested on the search.

#### [week1#check-high-dim] Reading a search score

*Checkpoint*

**Question:** A search over mean word vectors returns two articles with cosines 0.951 and 0.949 to a question. What is the soundest reading?

- A. The scores barely differ; check whether a shared direction inflates every cosine before trusting the order
- B. Both are excellent matches, since a cosine above 0.9 means near-identical meaning
- C. The first article is clearly the better answer, since it scores higher
- D. There is a 95% chance that either article answers the question

#### [week1#lab-1-1] Cosine-similarity search over public sentence embeddings

Lab 1.1 · 105 min

Search 60 help articles by meaning with mean GloVe vectors, remove the shared direction, and measure how occupation words lean towards gendered words.

[W1_1.ipynb](https://sdaiaacademy.github.io/fnd-mth-101/labs/W1_1.ipynb)

- [ ] Set up and load the data (10 min)
- [ ] By hand, then in code: dot product, norms and cosine (15 min)
- [ ] Embed the 60 article titles (15 min)
- [ ] Search with one matrix product and score ten questions (20 min)
- [ ] Find the cone, centre it, search again (20 min)
- [ ] Distances in high dimensions (10 min)
- [ ] Measure occupation words against gendered words and record what it means (15 min)
- [ ] ✱ Stretch: Show that Euclidean distance and cosine rank unit vectors in the same order

#### [week1#debrief-search] Centring and dropping frequent words more than doubled the questions answered first

*Debrief · lab 1.1*

| Sentence vectors | Mean pair cosine | Best article first | In the top 3 |
| --- | --- | --- | --- |
| Mean of all word vectors | 0.853 | 2 of 10 | 3 of 10 |
| Content words only, centred | −0.014 | 5 of 10 | 6 of 10 |

Four questions still miss the top 3: a mean of word vectors cannot know that "retake" and "take again" mean the same.

#### [week1#debrief-lean] Nurse, receptionist and housekeeper lean towards the female words

*Debrief · lab 1.1*

| Item | Value |
| --- | --- |
| nurse | +0.130 |
| receptionist | +0.128 |
| housekeeper | +0.120 |
| engineer | −0.066 |
| secretary | −0.076 |

"Secretary" leans male, perhaps because news text often uses it as a title of office. One small model, five word pairs, no error bars: the leans describe the texts' usage, not the jobs, and not yet the search.

#### [week1#wrap-up] Three things to keep

*Week 1 · wrap-up*

- **Length and angle.** The norm is a length; the dot product divided by two lengths is the cosine of an angle.
- **Centre first.** A direction every vector shares makes every cosine high. Subtract the mean before comparing.
- **Vectors carry text.** Embeddings inherit the associations of their training text. Measure them before building on them.

#### [week1#next] Week 2: Matrices as maps, and their cost

*Next*

Problem set 1 (60 minutes, AI off for the first attempt): labs/problem-sets/PS1.ipynb, before the next session. Finish lab 1.1 at home if needed.

## Week 2: Matrices as maps, and their cost

A matrix product is a map, a table of dot products and a bill for arithmetic

#### [week2#title] Matrices as maps, and their cost

*SDAIA Academy · Linear Algebra for Machine Learning*

Week 2: A matrix product is a map, a table of dot products and a bill for arithmetic

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-101

#### [week2#agenda] Matrices as maps, and their cost

*This week*

- **Session 1 · Matrix products, attention scores and FLOPs**: Attention scores as a matrix of dot products, matrix products as composition of maps, and counting floating-point operations and memory

#### [week2#review-week1] Which pair is most similar by cosine?

*Review · week 1*

**Question:** Three vectors: a = (2, 0), b = (10, 1), c = (1, 1). Which pair has the largest cosine similarity?

- A. a and b, because their dot product, 20, is the largest
- B. a and b, at a cosine of about 0.995
- C. a and c, because they have similar lengths
- D. b and c, because both have a second coordinate of 1

### Session 1: Matrix products, attention scores and FLOPs

#### [week2#s1] Matrix products, attention scores and FLOPs

*Session 1*

- Compute a matrix product by hand and state the shape of every product
- Explain a matrix product as one map followed by another
- Count the FLOPs and bytes of a product and compare loop and vectorised code

#### [week2#product-rule] Entry (i, j) of AB is row i of A dotted with column j of B

*Block A · the product*

P = [[2, 1], [1, 3]] (2 × 2), G = [[1, 0, 1], [2, 1, 0]] (2 × 3)

Entry (1, 1): (2, 1) · (1, 2) = 4

**P G = [[4, 1, 2], [7, 3, 1]]**, a 2 × 3 matrix

- **Shapes first.** (n × k)(k × m) gives n × m. The inner sizes must match. G P does not exist here: 3 does not match 2.

#### [week2#scores-as-product] Stacking vectors as rows turns all their dot products into one product

*Many dot products*

- Questions as rows of **Q** (q × d), articles as rows of **A** (a × d)
- **Q Aᵀ** is q × a: entry (i, j) is question i dotted with article j
- Week 1's search for one question was one row of this matrix

> **Why it matters.** One product call replaces q × a separate dot products, and the computer runs it far faster.

#### [week2#attention-scores] Attention scores are a matrix of dot products, scaled and softened

*Attention*

1. **1 · Scores** S = Q Kᵀ / √d
2. **2 · Softmax** each row to positive weights adding to 1
3. **3 · Mix** weights times the value vectors

One query q = (1, 0) against keys (1, 0), (0, 1), (1, 1) with d = 2: scores 0.707, 0, 0.707; weights **0.401, 0.198, 0.401**.

#### [week2#check-attention-shape] The shape of the scores

*Checkpoint*

**Question:** A layer compares 12 queries with 30 keys, each vector of length 64. What is the shape of S = Q Kᵀ?

- A. 30 × 12
- B. 12 × 64
- C. 12 × 30
- D. 64 × 64

#### [week2#matrix-is-a-map] A matrix is a map: it sends each vector to a new vector

*Block B · maps*

- Its columns are where the axis vectors land
- Rot = [[0, −1], [1, 0]] sends (1, 0) to (0, 1): a quarter turn
- Ref = [[1, 0], [0, −1]] flips the second coordinate: a mirror

- **Read the columns.** To know what a 2 × 2 matrix does, draw where (1, 0) and (0, 1) go. Every other vector follows by adding and scaling.

#### [week2#composition] A product of matrices is one map applied after another, read right to left

*Composition*

| Matrix | Means | Sends (1, 1) to |
| --- | --- | --- |
| Ref Rot | turn first, then mirror | (−1, −1) |
| Rot Ref | mirror first, then turn | (1, 1) |

> **Order matters.** Ref Rot ≠ Rot Ref. Matrix products are not commutative, because doing two things in another order is another map.

#### [week2#layers-compose] Stacked linear layers compose into one matrix

*In machine learning*

- Two linear layers W₂(W₁ x) equal one layer (W₂ W₁) x
- Without a non-linear step between them, depth adds nothing
- Shapes chain: (m × h)(h × d) maps d inputs to m outputs

> **Cost still counts.** W₂(W₁ x) and (W₂ W₁) x give the same result with different numbers of operations. Block C counts them.

#### [week2#check-composition] Which matrix does "scale, then turn"?

*Checkpoint*

**Question:** S doubles every vector's first coordinate; T turns the plane a quarter turn. You want one matrix that scales first, then turns. Which is it?

- A. T S
- B. S T
- C. S + T
- D. S T or T S, which are equal

#### [week2#count-flops] An (n × k)(k × m) product costs 2·n·k·m floating-point operations

*Block C · cost*

n·m entries, each a dot product of length k: k multiplications and k additions.

(3 × 4)(4 × 2): 2 × 3 × 4 × 2 = **48 FLOPs**.

| n | FLOPs of n × n times n × n | MB per matrix |
| --- | --- | --- |
| 100 | 0.002 G | 0.08 |
| 1,000 | 2 G | 8 |
| 10,000 | 2,000 G | 800 |

#### [week2#bracket-order] Where you put the brackets changes the cost, not the answer

*Same result, different bill*

| Order | FLOPs |
| --- | --- |
| (W₂ W₁) x, with W₂ 10 × 1,000, W₁ 1,000 × 1,000, x of length 1,000 | 20,020,000 |
| W₂ (W₁ x) | 2,020,000 |

2·10·1,000·1,000 + 2·10·1,000 against 2·1,000·1,000 + 2·10·1,000.

#### [week2#vectorise-first-look] A Python loop pays for every number; a vectorised call pays once

*A first look at vectorisation*

```python
# file: numpy
# loop: one Python step per multiply
for i in range(n):
    for j in range(m):
        C[i][j] = sum(A[i][t] * B[t][j]
                      for t in range(k))

# vectorised: one call
C = A @ B
```

- Same arithmetic, same FLOP count
- The call runs compiled code over whole blocks of memory
- Week 8 returns to vectorisation and broadcasting in full

#### [week2#check-flops] How much does doubling the size cost?

*Checkpoint*

**Question:** Multiplying two 500 × 500 matrices takes 0.25 GFLOPs. About how many FLOPs does a 1,000 × 1,000 product take?

- A. 0.5 GFLOPs
- B. 1 GFLOP
- C. 4 GFLOPs
- D. 2 GFLOPs

#### [week2#lab-2-1] Loop and vectorised matrix products, timed and costed in FLOPs

Lab 2.1 · 105 min

Time triple-loop, row-loop and vectorised matrix products on growing sizes, count their FLOPs and memory, and cost the platform's daily scoring.

[W2_1.ipynb](https://sdaiaacademy.github.io/fnd-mth-101/labs/W2_1.ipynb)

- [ ] Set up; a product by hand, then in code (15 min)
- [ ] Attention scores as one product, with a row softmax (15 min)
- [ ] Composition: two maps, one matrix, and why order matters (10 min)
- [ ] Time a triple-loop product on growing sizes (15 min)
- [ ] Count FLOPs and bytes, and compute the rate achieved (15 min)
- [ ] Time the row-loop and vectorised products (20 min)
- [ ] Cost the platform's daily scoring (15 min)
- [ ] ✱ Stretch: Repeat the timing in 32-bit floats and compare time and memory

#### [week2#debrief-rates] Same FLOPs, rates a thousand times apart

*Debrief · lab 2.1*

| Version | Size | Rate |
| --- | --- | --- |
| Triple Python loop | 32 to 128 | 52 to 56 million FLOPs per second |
| Row loop with NumPy rows | 1,024 | 0.33 s for 2.1 GFLOPs |
| One vectorised call | 1,024 | about 128 GFLOP/s |

One run of the lab's solution on the build laptop; your numbers will differ. At n = 128 the vectorised call was about 595 times faster than the triple loop.

#### [week2#debrief-daily-cost] The planned search costs 640 times today's, and still fits in seconds a day

*Debrief · lab 2.1*

- **0.30 G** FLOPs a day today: 50,000 questions × 60 articles × 50 numbers
- **192 G** FLOPs a day planned: 5,000 articles × 384 numbers

> **For the search lead.** About 1.5 seconds a day at the rate measured on the build laptop; the article matrix grows to 15.36 MB. Cost is not the obstacle; search quality is.

#### [week2#wrap-up] Three things to keep

*Week 2 · wrap-up*

- **Shapes first.** (n × k)(k × m) is n × m. Check shapes before arithmetic.
- **Right to left.** A product of matrices applies the right-hand map first. Order changes the map.
- **Count, then time.** 2·n·k·m FLOPs is fixed by the maths; the rate is set by how the code runs.

#### [week2#next] Week 3: Span, rank and linear systems

*Next*

Problem set 2 (60 minutes, AI off for the first attempt): labs/problem-sets/PS2.ipynb.

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
