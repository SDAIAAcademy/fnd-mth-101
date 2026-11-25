---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "week2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/week2.html"
---

# Linear Algebra for Machine Learning

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
