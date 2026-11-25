# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1"]
# ///
# %% [markdown]
# # Lab 1.1 · Cosine search over public sentence embeddings
#
# Week 1 · Session 1 (Vectors, dot products and similarity) · 105 min.
#
# The help centre of the online learning platform answers learners' questions with one of 60 help
# articles. You build its first search by meaning: each article title becomes a vector, a question
# becomes a vector, and the answer is the article at the smallest angle. The vectors are **sentence
# embeddings built as the mean of public GloVe word vectors** (6 billion tokens, 50 dimensions,
# Public Domain Dedication and License): a simple baseline, not the output of a sentence model.
# Then you measure how the same vectors place occupation words between gendered words.
#
# **AI policy.** AI-tutor: an approved assistant may give hints, never answers (`labs/tutor/TUTOR.md`).
# Paste only public course material into it.
#
# **Tasks**
# - [ ] 1 · Set up and load the data (10 min)
# - [ ] 2 · By hand, then in code: dot product, norms and cosine (15 min)
# - [ ] 3 · Embed the 60 article titles (15 min)
# - [ ] 4 · Search with one matrix product and score ten questions (20 min)
# - [ ] 5 · Find the cone, centre it, search again (20 min)
# - [ ] 6 · Distances in high dimensions (10 min)
# - [ ] 7 · Measure occupation words against gendered words and record what it means (15 min)
# - [ ] ✱ Stretch · Show that Euclidean distance and cosine rank unit vectors in the same order

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/fnd-mth-101.git"
    if not Path("/content/fnd-mth-101").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/fnd-mth-101"], check=True)
    os.chdir("/content/fnd-mth-101/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import json
import numpy as np
import mth as U

print("quick mode:", U.QUICK)

# %% [markdown]
# ## 1 · Set up and load the data
#
# `help_articles.csv` holds the 60 synthetic articles (codes A001 to A060), `queries.csv` ten
# learners' questions with the article a content lead judged best. The first call downloads the
# GloVe file once (66 MB) and checks its published checksum.

# %%
articles = U.load_csv("help_articles.csv")
queries = U.load_csv("queries.csv")
bias = json.loads((U.DATA / "bias_words.json").read_text(encoding="utf-8"))

words = set()
for a in articles:
    words |= set(U.tokens(a["title"]))
for q in queries:
    words |= set(U.tokens(q["text"]))
words |= set(bias["occupations"]) | {w for pair in bias["gender_pairs"] for w in pair}
V = U.glove(words, first=100)                 # our words, and the 100 most frequent words
FREQUENT = set(list(U.glove(None, first=100)))
print(len(articles), "articles,", len(queries), "questions,", len(V), "word vectors of length", len(V["course"]))
U.check(words <= V.keys(), "every word of the data has a vector")

# %% [markdown]
# ## 2 · By hand, then in code
#
# Two learners' activity this week as vectors (videos, quizzes, posts): `u = (3, 0, 4)` and
# `v = (4, 4, 2)`. **On paper first**, without a computer: compute `u · v`, `‖u‖`, `‖v‖` and the
# cosine of the angle between them. Write your four numbers below, then write the code.

# %%
# ✏️ Your hand results: replace the None values with your numbers
hand = {"dot": None, "norm_u": None, "norm_v": None, "cos": None}

def cosine(a, b):
    """The cosine of the angle between two vectors."""
    # BEGIN SOLUTION
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    # END SOLUTION

# %%
u, v = np.array([3.0, 0, 4]), np.array([4.0, 4, 2])
try:
    c = cosine(u, v)
except NotImplementedError:
    c = U.reference("w1_cos_uv")
c = U.keep("w1_cos_uv", c)
print("code:", round(float(u @ v), 4), round(float(np.linalg.norm(u)), 4), round(float(np.linalg.norm(v)), 4), round(c, 4))
def reference_hand():
    # BEGIN SOLUTION
    return {"dot": 20, "norm_u": 5, "norm_v": 6, "cos": 20 / 30}  # the solution's hand work
    # END SOLUTION

if None in hand.values():
    try:
        hand = reference_hand()
    except NotImplementedError:
        print("Write your four hand results in `hand` to compare them with the code.")
if None not in hand.values():
    U.check(np.allclose([hand["dot"], hand["norm_u"], hand["norm_v"], hand["cos"]],
                        [u @ v, np.linalg.norm(u), np.linalg.norm(v), c], atol=1e-3), "hand and code agree")

# %% [markdown]
# ## 3 · Embed the 60 article titles
#
# A title's embedding is the mean of its words' vectors: `embed(t) = (1/n) Σ vec(wordᵢ)`.
# Build the 60 × 50 matrix `E`, one row per article, with `U.embed_sentence`.

# %%
def embed_all(texts):
    """A matrix with one row per text: the mean of its word vectors."""
    # BEGIN SOLUTION
    return np.array([U.embed_sentence(t, V) for t in texts])
    # END SOLUTION

# %%
titles = [a["title"] for a in articles]
ids = [a["article"] for a in articles]
try:
    E = embed_all(titles)
except NotImplementedError:
    E = np.array([U.embed_sentence(t, V) for t in titles])
U.check(E.shape == (60, 50), f"E has shape {E.shape}")
print("norms of the first five rows:", np.linalg.norm(E[:5], axis=1).round(3))

# %% [markdown]
# **Look at** the norms: they differ from title to title. Cosine ignores length, which is why the
# search compares angles, not raw dot products.

# %% [markdown]
# ## 4 · Search with one matrix product
#
# Normalise every row to length 1. Then the cosines of one question with all 60 titles are one
# matrix-vector product, `Ê q̂`. Write `search(text, E, k)` returning the `k` best article codes,
# then find the rank of the judged-best article for each of the ten questions.

# %%
def normalise_rows(M):
    # BEGIN SOLUTION
    return M / np.linalg.norm(M, axis=1, keepdims=True)
    # END SOLUTION

def search(qvec, M, k=3):
    """Codes of the k rows of M with the largest cosine to qvec, best first."""
    # BEGIN SOLUTION
    scores = normalise_rows(M) @ (qvec / np.linalg.norm(qvec))
    return [ids[i] for i in np.argsort(-scores)[:k]]
    # END SOLUTION

def rank_of_best(qvecs, M):
    return [search(qv, M, k=len(ids)).index(q["best_article"]) + 1 for qv, q in zip(qvecs, queries)]

# %%
Q = np.array([U.embed_sentence(q["text"], V) for q in queries])
try:
    ranks_raw = rank_of_best(Q, E)
except NotImplementedError:
    ranks_raw = U.reference("w1_ranks_raw")
ranks_raw = U.keep("w1_ranks_raw", ranks_raw)
print("rank of the judged-best article, raw means:", ranks_raw)
print("found first:", sum(r == 1 for r in ranks_raw), "of 10 | in the top 3:", sum(r <= 3 for r in ranks_raw))

# %% [markdown]
# ## 5 · Find the cone, centre it, search again
#
# Compute the mean cosine over all pairs of distinct titles. Close to 1 means every vector points
# into one narrow cone: the frequent words ("my", "the", "i") dominate every mean. Two fixes,
# both linear algebra: **centre** (subtract the mean row from every row and from the question) and
# **drop the 100 most frequent words** before averaging. Measure the cone again, then the ranks.

# %%
def mean_pair_cosine(M):
    """Mean cosine over all pairs of distinct rows."""
    # BEGIN SOLUTION
    N = normalise_rows(M)
    C = N @ N.T
    iu = np.triu_indices(len(M), 1)
    return float(C[iu].mean())
    # END SOLUTION

def embed_content(text):
    ws = [w for w in U.tokens(text) if w not in FREQUENT] or U.tokens(text)
    return U.embed_sentence(" ".join(ws), V)

# %%
Ec = np.array([embed_content(t) for t in titles])
Qc = np.array([embed_content(q["text"]) for q in queries])
mu = Ec.mean(axis=0)
try:
    cones = {"raw": mean_pair_cosine(E), "centred, content words": mean_pair_cosine(Ec - mu)}
except NotImplementedError:
    cones = U.reference("w1_cones")
cones = U.keep("w1_cones", cones)
try:
    ranks_fixed = rank_of_best(Qc - mu, Ec - mu)
except NotImplementedError:
    ranks_fixed = U.reference("w1_ranks_fixed")
ranks_fixed = U.keep("w1_ranks_fixed", ranks_fixed)
print("mean pair cosine:", {k: round(x, 3) for k, x in cones.items()})
print("ranks after the fixes:", ranks_fixed)
print("found first:", sum(r == 1 for r in ranks_fixed), "of 10 | in the top 3:", sum(r <= 3 for r in ranks_fixed))
U.check(cones["raw"] > 0.8 and abs(cones["centred, content words"]) < 0.1, "the cone is found and removed")

# %% [markdown]
# **Look at** the questions that still fail (rank above 3). Which words do they share with their
# best article? A mean of word vectors cannot know that "retake" and "take again" mean the same.

# %% [markdown]
# ## 6 · Distances in high dimensions
#
# Draw 200 random points with independent standard normal coordinates in 2, 50 and 1,000
# dimensions. For one point, compute its distances to the others and the ratio
# `(farthest − nearest) / nearest`. In high dimensions the ratio shrinks: "nearest" means less.

# %%
def contrast(d, n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    # BEGIN SOLUTION
    dist = np.linalg.norm(X[1:] - X[0], axis=1)
    return float((dist.max() - dist.min()) / dist.min())
    # END SOLUTION

# %%
try:
    ratios = {d: contrast(d) for d in (2, 50, 1000)}
except NotImplementedError:
    ratios = {int(k): x for k, x in U.reference("w1_contrast").items()}
U.keep("w1_contrast", {str(k): x for k, x in ratios.items()})
print({d: round(r, 3) for d, r in ratios.items()})
U.check(ratios[2] > ratios[50] > ratios[1000], "the contrast falls as the dimension grows")

# %% [markdown]
# ## 7 · Occupation words and gendered words
#
# For each occupation word `o`, its **lean** is the mean over five pairs (she/he, woman/man,
# her/his, female/male, mother/father) of `cos(o, female word) − cos(o, male word)`. Positive means
# closer to the female words. Compute the lean of the 14 occupations and sort them.

# %%
def lean(word):
    # BEGIN SOLUTION
    return float(np.mean([cosine(V[word], V[f]) - cosine(V[word], V[m]) for f, m in bias["gender_pairs"]]))
    # END SOLUTION

# %%
try:
    leans = {o: lean(o) for o in bias["occupations"]}
except NotImplementedError:
    leans = U.reference("w1_leans")
leans = U.keep("w1_leans", leans)
for o, x in sorted(leans.items(), key=lambda kv: -kv[1]):
    print(f"{o:13s} {x:+.3f}")
U.check(leans["nurse"] > 0.05 > leans["engineer"], "the lean separates some occupations clearly")

# %% [markdown]
# **Record** in two or three sentences, in the cell below: if the platform used these vectors to
# suggest courses from a learner's free-text goal, what could happen to a learner who writes "she"
# or "her" in that goal? What would you check before such a search goes live? (No code: this is
# the note the search lead asked for.)

# %% [markdown]
# *Your note:*

# %% [markdown]
# ## ✱ Stretch
#
# For unit vectors, `‖a − b‖² = 2 − 2 cos(a, b)`. Show on the normalised `E` that ranking by
# Euclidean distance and ranking by cosine give the same order for every question.

# %% [markdown]
# ## Check yourself
#
# <details><summary>1. Why does the raw search find so few best articles?</summary>
# Every mean vector is dominated by the same frequent words, so all titles sit in a narrow cone and
# their cosines with any question are all close to 1. Small differences decide the order.</details>
#
# <details><summary>2. Why centre the vectors before measuring angles?</summary>
# Subtracting the mean removes the direction every vector shares; the angles that are left
# describe how titles differ, which is what a search needs.</details>
#
# <details><summary>3. Does a positive lean prove that the search treats learners unfairly?</summary>
# No. It shows an association learnt from the text the vectors were trained on. Whether a search
# changes its results for a learner because of it has to be tested on the search itself.</details>
#
# ## Carry forward
#
# Week 2 writes the scores of many questions against many articles as one matrix product. Nothing
# from this lab is needed: week 2 starts from its own files.
