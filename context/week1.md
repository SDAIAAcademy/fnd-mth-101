---
course: "Linear Algebra for Machine Learning"
code: "FND-MTH-101"
version: "2027.1.0"
deck: "week1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-101/slides/week1.html"
---

# Linear Algebra for Machine Learning

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
