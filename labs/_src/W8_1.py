# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1"]
# ///
# %% [markdown]
# # Lab 8.1 · Low-rank ratings and adapter updates, checked by hand
#
# Week 8 · Session 1 (Low rank in practice) · 105 min.
#
# Learners of the online learning platform rate the courses they take (`data/ratings.csv`: 400
# learners, 30 courses, ratings from 1 to 5 in half steps, a training split and a held-out test
# split; synthetic). You predict held-out ratings with a truncated SVD and choose the rank, count
# the parameters of a low-rank weight update of the kind used to adapt large models, meet two
# floating-point traps, and find a planted broadcasting bug by checking code against a hand case.
# The lab ends with your capstone plan.
#
# **AI policy.** AI-tutor: hints, never answers (`labs/tutor/TUTOR.md`). The capstone, from the
# next step on, is AI-full and defended orally.
#
# **Tasks**
# - [ ] 1 · Set up; broadcasting shapes by hand (10 min)
# - [ ] 2 · Ratings: fill, centre, truncate, test (25 min)
# - [ ] 3 · A low-rank weight update (15 min)
# - [ ] 4 · Floating-point limits (15 min)
# - [ ] 5 · Check a function against hand cases (15 min)
# - [ ] 6 · Find and fix the planted bug (15 min)
# - [ ] 7 · Write your capstone plan (10 min)
# - [ ] ✱ Stretch · Fit the ratings by alternating least squares and compare

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

import numpy as np
import mth as U

print("quick mode:", U.QUICK)

# %% [markdown]
# ## 1 · Broadcasting by hand
#
# NumPy lines shapes up from the right; a dimension of 1 stretches to match. **On paper**, give the
# shape of each result, or "error": `(4, 1) + (3,)`, `(4, 3) − (4,)`, `(4, 3) − (4, 1)`,
# `(2, 1, 3) * (5, 1)`. Write them in `hand_shapes`, then check with code.

# %%
hand_shapes = None   # ✏️ a list of four shapes (tuples) or "error"

def reference_hand_shapes():
    # BEGIN SOLUTION
    return [(4, 3), "error", (4, 3), (2, 5, 3)]
    # END SOLUTION

# %%
pairs = [((4, 1), (3,)), ((4, 3), (4,)), ((4, 3), (4, 1)), ((2, 1, 3), (5, 1))]
code_shapes = []
for a, b in pairs:
    try:
        code_shapes.append((np.zeros(a) + np.zeros(b)).shape)
    except ValueError:
        code_shapes.append("error")
print("shapes:", code_shapes)
if hand_shapes is None:
    try:
        hand_shapes = reference_hand_shapes()
    except NotImplementedError:
        pass
if hand_shapes is not None:
    U.check([tuple(s) if s != "error" else s for s in hand_shapes] == code_shapes, "hand and code agree")

# %% [markdown]
# ## 2 · Ratings: fill, centre, truncate, test
#
# Build the 400 × 30 training matrix with NaN where a learner has not rated a course. Fill each
# gap with the course's mean rating, subtract the course means (one broadcast), take the SVD,
# keep k layers, add the means back, and measure the root mean squared error (RMSE) on the
# held-out ratings for k = 1 to 10. Compare with the baseline that predicts the course mean.

# %%
rows = U.load_csv("ratings.csv")
learners = sorted({r["learner"] for r in rows})
courses = sorted({r["course"] for r in rows})
li, ci = {x: i for i, x in enumerate(learners)}, {x: j for j, x in enumerate(courses)}
R = np.full((len(learners), len(courses)), np.nan)
test = []
for r in rows:
    i, j, v = li[r["learner"]], ci[r["course"]], float(r["rating"])
    if r["split"] == "train":
        R[i, j] = v
    else:
        test.append((i, j, v))
ti, tj, tv = (np.array(x) for x in zip(*test))
print(R.shape, "| training ratings:", int(np.isfinite(R).sum()), "| held out:", len(tv))

def predict_rank_k(R, k):
    """Fill gaps with course means, centre, keep k SVD layers, add the means back."""
    # BEGIN SOLUTION
    mu = np.nanmean(R, axis=0)
    F = np.where(np.isnan(R), mu, R) - mu
    Uu, s, Vt = np.linalg.svd(F, full_matrices=False)
    return (Uu[:, :k] * s[:k]) @ Vt[:k] + mu
    # END SOLUTION

def rmse(P):
    return float(np.sqrt(np.mean((P[ti, tj] - tv) ** 2)))

# %%
mu = np.nanmean(R, axis=0)
baseline = rmse(np.tile(mu, (len(learners), 1)))
scores = {}
for k in range(1, 11):
    try:
        scores[k] = rmse(predict_rank_k(R, k))
    except NotImplementedError:
        scores = {int(a): b for a, b in U.reference("w8_rmse").items()}
        break
U.keep("w8_rmse", {str(k): v for k, v in scores.items()})
print(f"baseline (course mean): {baseline:.4f}")
for k, v in scores.items():
    print(f"k = {k:2d}  RMSE {v:.4f}")
best = min(scores, key=scores.get)
print("best rank:", best)
U.check(scores[best] < baseline, "a low-rank model beats the course-mean baseline")

# %% [markdown]
# **Look at** the curve: the error falls, then rises. Extra layers start fitting noise in the
# training ratings. The rank is chosen on held-out data, never on the data the SVD saw.

# %% [markdown]
# ## 3 · A low-rank weight update
#
# Adapting a large model can mean learning a change `ΔW` to a d × d weight matrix. A low-rank
# adapter writes `ΔW = B A` with `B` d × r and `A` r × d. Write `adapter_params(d, r)` and
# compare with `d²` for d = 768 and r = 4, 8, 16. Then check, on a synthetic update made of a
# rank-4 part plus small noise, that its singular values drop after the fourth.

# %%
def adapter_params(d, r):
    # BEGIN SOLUTION
    return 2 * d * r
    # END SOLUTION

# %%
d = 768
for r in (4, 8, 16):
    try:
        p = adapter_params(d, r)
    except NotImplementedError:
        p = 2 * d * r
    print(f"r = {r:2d}: {p:,} parameters, {p / d**2:.2%} of the {d**2:,} in a full update")
rng = np.random.default_rng(8)
dd = U.pick(256, 128)
dW = rng.standard_normal((dd, 4)) @ rng.standard_normal((4, dd)) + 0.01 * rng.standard_normal((dd, dd))
sv = np.linalg.svd(dW, compute_uv=False)
print("first seven singular values of the synthetic update:", sv[:7].round(2))
U.check(sv[3] > 10 * sv[4], "the update is close to rank 4")

# %% [markdown]
# ## 4 · Floating-point limits
#
# (a) Print `0.1 + 0.2 == 0.3`, the gap, and machine epsilon for float32 and float64.
# (b) The variance of five values 10,001 to 10,005 is 2.5 (dividing by n − 1). Compute it in
# float32 two ways: the one-pass formula `(Σx² − n·mean²)/(n − 1)` and the two-pass formula
# `Σ(x − mean)²/(n − 1)`. Which survives, and why?

# %%
print("0.1 + 0.2 == 0.3:", 0.1 + 0.2 == 0.3, "| gap:", abs(0.1 + 0.2 - 0.3))
print("machine epsilon: float32", np.finfo(np.float32).eps, "| float64", np.finfo(np.float64).eps)

def var_one_pass(x):
    # BEGIN SOLUTION
    n = len(x)
    return (np.sum(x * x) - n * np.mean(x) ** 2) / (n - 1)
    # END SOLUTION

def var_two_pass(x):
    # BEGIN SOLUTION
    return np.sum((x - np.mean(x)) ** 2) / (len(x) - 1)
    # END SOLUTION

# %%
x32 = np.arange(10001, 10006, dtype=np.float32)
try:
    one, two = float(var_one_pass(x32)), float(var_two_pass(x32))
except NotImplementedError:
    one, two = U.reference("w8_var")
U.keep("w8_var", [one, two])
print("float32 one-pass:", one, "| two-pass:", two, "| true: 2.5")
U.check(two == 2.5 and one != 2.5, "subtracting two large, nearly equal numbers lost the answer")

# %% [markdown]
# ## 5 · Check a function against hand cases
#
# A check harness runs a function on cases whose answers you worked out by hand, and on random
# cases against a trusted library, with a tolerance. Write `check_function(f, hand_cases,
# random_inputs, reference, tol)`, returning a list of failures. Use it on `sv2`, a function that
# returns the singular values of a 2 × 2 matrix from the formula, with two hand cases from earlier
# weeks: [[3, 0], [4, 5]] gives (√45, √5) and [[2, 0], [0, −3]] gives (3, 2).

# %%
def sv2(M):
    """Singular values of a 2 x 2 matrix, largest first, from the eigenvalues of MᵀM."""
    G = M.T @ M
    tr, det = G[0, 0] + G[1, 1], G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    return np.sqrt([tr / 2 + disc, max(tr / 2 - disc, 0.0)])

def check_function(f, hand_cases, random_inputs, reference, tol=1e-6):
    # BEGIN SOLUTION
    failures = []
    for x, want in hand_cases:
        if not np.allclose(f(x), want, atol=tol, rtol=tol):
            failures.append(("hand", x.tolist(), want))
    for x in random_inputs:
        if not np.allclose(f(x), reference(x), atol=tol, rtol=tol):
            failures.append(("random", x.tolist()))
    return failures
    # END SOLUTION

# %%
hand_cases = [(np.array([[3.0, 0], [4, 5]]), [45 ** 0.5, 5 ** 0.5]), (np.array([[2.0, 0], [0, -3]]), [3, 2])]
randoms = [np.random.default_rng(k).standard_normal((2, 2)) for k in range(U.pick(200, 20))]
try:
    fails = check_function(sv2, hand_cases, randoms, lambda m: np.linalg.svd(m, compute_uv=False))
except NotImplementedError:
    fails = []
print("failures of sv2:", len(fails))
U.check(fails == [], "sv2 agrees with the hand cases and with the library")

# %% [markdown]
# ## 6 · The planted bug
#
# `centre_rows` below should subtract each row's mean from that row. It runs without an error on
# square matrices and gives wrong numbers. Run it on a 2 × 3 hand case first (rows (1, 2, 3) and
# (4, 6, 8): centred rows are (−1, 0, 1) and (−2, 0, 2)), then on a 3 × 3 case. Find the bug from
# the shapes and write `centre_rows_fixed`.

# %%
def centre_rows(M):
    means = M.mean(axis=1)       # shape (n,)
    return M - means             # planted bug: (n, m) − (n,) lines up the means with the columns

def centre_rows_fixed(M):
    # BEGIN SOLUTION
    return M - M.mean(axis=1, keepdims=True)
    # END SOLUTION

# %%
H = np.array([[1.0, 2, 3], [4, 6, 8]])
try:
    print("buggy on 2 x 3:", centre_rows(H))
except ValueError as e:
    print("buggy on 2 x 3 fails loudly:", e)
Sq = np.array([[1.0, 2, 3], [4, 6, 8], [0, 0, 3]])
print("buggy on 3 x 3 (no error, wrong):\n", centre_rows(Sq))
try:
    fixed = centre_rows_fixed(H)
    U.check(np.allclose(fixed, [[-1, 0, 1], [-2, 0, 2]]), "the fixed version matches the hand case")
    U.check(np.allclose(centre_rows_fixed(Sq).sum(axis=1), 0), "every row of the fixed 3 x 3 sums to zero")
except NotImplementedError:
    print("write centre_rows_fixed to continue")

# %% [markdown]
# **Look at** why the 2 × 3 case failed loudly and the 3 × 3 case did not. A square test matrix
# hides axis bugs: test with rectangular shapes.

# %% [markdown]
# ## 7 · Your capstone plan
#
# Fill in the plan below (no code). The trainer approves the dataset before you start.
#
# - **Dataset** (public image, ratings or embedding matrix), its address and its **licence**:
# - **Matrix**: what the rows and columns are, its shape, and whether you will centre or scale it:
# - **A hand check**: the small case you will compute by hand to check your code:
# - **Ranks** you will compare, and how you will choose one:
# - **What the leading components might mean**, as a question to test:
# - **Personal data**: confirm there is none, or say how it is excluded:

# %% [markdown]
# ## ✱ Stretch
#
# Fit the ratings by alternating least squares on the observed entries only (fix B, solve for A
# by least squares per learner; fix A, solve for B per course; repeat) and compare its test RMSE
# by rank with the SVD's.

# %% [markdown]
# ## Check yourself
#
# <details><summary>1. Why not choose the rank with the lowest training error?</summary>
# Training error keeps falling as k grows: more layers always fit the seen ratings better. Only
# held-out error shows when the extra layers fit noise.</details>
#
# <details><summary>2. Why does a low-rank update save so much?</summary>
# 2dr numbers instead of d²: for d = 768 and r = 8, about 2% of the full update.</details>
#
# <details><summary>3. Why did the one-pass variance fail in float32?</summary>
# It subtracts two numbers near 5 × 10⁸ that agree in their first eight or nine digits; float32
# keeps about seven, so the difference is mostly rounding.</details>
#
# ## Carry forward
#
# Your plan in step 7 is the first page of the capstone notebook.
