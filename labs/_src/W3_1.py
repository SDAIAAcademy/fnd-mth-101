# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1"]
# ///
# %% [markdown]
# # Lab 3.1 · Rank and linear systems on a feature table
#
# Week 3 · Session 1 (Span, rank and linear systems) · 105 min.
#
# The data analyst of the online learning platform keeps a feature table of 400 learners
# (`data/learner_features.csv`): videos watched, video minutes, quizzes, forum posts, total actions
# and weekly study minutes. Before anyone fits a model on it, you check which columns carry new
# information. You solve a small system by elimination, test whether a vector lies in a span,
# find the table's rank and the column it repeats, and compare solving with inverting.
#
# **AI policy.** AI-tutor: hints, never answers (`labs/tutor/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Set up; elimination by hand, then a solver (20 min)
# - [ ] 2 · Span: is a vector a combination of two others? (15 min)
# - [ ] 3 · The rank of the table, and the column it repeats (20 min)
# - [ ] 4 · What the repeated column does to a solve (15 min)
# - [ ] 5 · A basis: drop the column, check the rank (10 min)
# - [ ] 6 · Solve against invert: residuals and time (15 min)
# - [ ] 7 · Record which columns to keep (10 min)
# - [ ] ✱ Stretch · Read the repeated column off the null space of the full table

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

import time
import numpy as np
import mth as U

print("quick mode:", U.QUICK)

# %% [markdown]
# ## 1 · Elimination by hand, then a solver
#
# **On paper:** solve by elimination
#
# ```
#  x +  y +  z = 6
# 2x +  y −  z = 1
#  x −  y + 2z = 5
# ```
#
# Write your answer in `hand_xyz`. Then write `solve_elimination(A, b)`: forward elimination with
# partial pivoting (swap in the row with the largest pivot), then back substitution.

# %%
hand_xyz = None   # ✏️ your hand result, (x, y, z)

def solve_elimination(A, b):
    """Solve A x = b for a square A by Gaussian elimination with partial pivoting."""
    # BEGIN SOLUTION
    M = np.column_stack([np.array(A, dtype=float), np.array(b, dtype=float)])
    n = len(M)
    for col in range(n):
        piv = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[piv, col]) < 1e-12:
            raise np.linalg.LinAlgError("singular matrix")
        M[[col, piv]] = M[[piv, col]]
        for r in range(col + 1, n):
            M[r] -= M[r, col] / M[col, col] * M[col]
    x = np.zeros(n)
    for r in range(n - 1, -1, -1):
        x[r] = (M[r, -1] - M[r, r + 1:n] @ x[r + 1:]) / M[r, r]
    return x
    # END SOLUTION

def reference_hand_xyz():
    # BEGIN SOLUTION
    return [1, 2, 3]
    # END SOLUTION

# %%
A3 = np.array([[1.0, 1, 1], [2, 1, -1], [1, -1, 2]])
b3 = np.array([6.0, 1, 5])
try:
    x3 = solve_elimination(A3, b3)
except NotImplementedError:
    x3 = np.linalg.solve(A3, b3)
print("elimination:", x3, "| library:", np.linalg.solve(A3, b3))
U.check(np.allclose(A3 @ x3, b3), "A x = b holds")
if hand_xyz is None:
    try:
        hand_xyz = reference_hand_xyz()
    except NotImplementedError:
        pass
if hand_xyz is not None:
    U.check(np.allclose(hand_xyz, x3), "hand and code agree")

# %% [markdown]
# ## 2 · Span
#
# `w` is in the span of `u` and `v` when `a u + b v = w` for some numbers a and b. Put `u` and `v`
# as the columns of a matrix and compare its rank with the rank after adding `w` as a third
# column: the rank grows only when `w` adds a new direction. Write `in_span(vectors, w)`.

# %%
def in_span(vectors, w):
    # BEGIN SOLUTION
    M = np.column_stack(vectors)
    return bool(np.linalg.matrix_rank(np.column_stack([M, w])) == np.linalg.matrix_rank(M))
    # END SOLUTION

# %%
u, v = np.array([1.0, 0, 1]), np.array([0.0, 1, 1])
w_in, w_out = np.array([2.0, 3, 5]), np.array([1.0, 1, 1])
try:
    spans = (in_span([u, v], w_in), in_span([u, v], w_out))
except NotImplementedError:
    spans = (True, False)
print("(2, 3, 5) in span:", spans[0], "| (1, 1, 1) in span:", spans[1])
U.check(spans == (True, False), "one vector is in the span and one is not")

# %% [markdown]
# ## 3 · The rank of the feature table
#
# Build the 400 × 5 matrix of the columns videos, video_minutes, quizzes, posts, total_actions.
# Its rank is the number of independent columns. Compute it, look at the singular values (they
# come back in week 7: here, a zero singular value means a lost direction), and find which column
# is a combination of the others.

# %%
rows = U.load_csv("learner_features.csv")
cols = ["videos", "video_minutes", "quizzes", "posts", "total_actions"]
X = np.array([[float(r[c]) for c in cols] for r in rows])
y = np.array([float(r["weekly_minutes"]) for r in rows])

def table_rank(M):
    # BEGIN SOLUTION
    return int(np.linalg.matrix_rank(M))
    # END SOLUTION

# %%
try:
    rank = table_rank(X)
except NotImplementedError:
    rank = U.reference("w3_rank")
rank = U.keep("w3_rank", rank)
sv = np.linalg.svd(X, compute_uv=False)
print("shape:", X.shape, "| rank:", rank)
print("singular values:", sv.round(4))
U.check(rank == 4, "five columns, rank 4: one column repeats the others")
combo = X[:, 0] + X[:, 2] + X[:, 3] - X[:, 4]
print("largest |videos + quizzes + posts − total_actions| over 400 learners:", np.abs(combo).max())

# %% [markdown]
# **Look at** the last singular value: it is zero to about 13 decimals, the size of rounding.
# `total_actions` is exactly `videos + quizzes + posts`.

# %% [markdown]
# ## 4 · What the repeated column does
#
# A fit of weekly minutes on the five columns needs the 5 × 5 matrix `XᵀX`. Try to solve
# `XᵀX β = Xᵀy` and to invert `XᵀX`. Print the condition number of `XᵀX` (week 5 explains it:
# the larger, the less you can trust a solve).

# %%
G = X.T @ X
print("condition number of XᵀX:", f"{np.linalg.cond(G):.3g}")
try:
    beta = np.linalg.solve(G, X.T @ y)
    print("solve returned:", beta.round(3), "| check XᵀX β − Xᵀy:", np.abs(G @ beta - X.T @ y).max().round(6))
except np.linalg.LinAlgError as e:
    print("solve failed:", e)
try:
    Ginv = np.linalg.inv(G)
    print("inverse: largest entry", f"{np.abs(Ginv).max():.3g}", "| G Ginv − I largest:", f"{np.abs(G @ Ginv - np.eye(5)).max():.3g}")
except np.linalg.LinAlgError as e:
    print("inverse failed:", e)

# %% [markdown]
# **Look at** what happened on your computer. Rounding may let the solver return numbers, but
# they are not to be trusted: with a repeated column, infinitely many β fit equally well, and
# the solver's choice is an accident of rounding.

# %% [markdown]
# ## 5 · A basis
#
# Drop `total_actions`. Check that the remaining four columns are independent (rank equals the
# number of columns) and that the condition number of `XᵀX` falls.

# %%
X4 = X[:, :4]
try:
    r4 = table_rank(X4)
except NotImplementedError:
    r4 = 4
print("rank of the four columns:", r4, "| condition number of X4ᵀX4:", f"{np.linalg.cond(X4.T @ X4):.3g}")
U.check(r4 == X4.shape[1], "the four columns are a basis of the column space")

# %% [markdown]
# ## 6 · Solve against invert
#
# On an n × n system whose condition number is 10¹⁰ (built on purpose), compare
# `np.linalg.solve(A, b)` with `np.linalg.inv(A) @ b`: the residual `‖A x − b‖` and the best of
# three timings. Write `compare(n)`.

# %%
def best_time(f, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best

def make_system(n, seed=3):
    rng = np.random.default_rng(seed)
    Qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Qm @ np.diag(np.logspace(0, -10, n)) @ Qm.T   # singular values from 1 down to 1e-10
    return A, rng.standard_normal(n)

def compare(n):
    A, b = make_system(n)
    # BEGIN SOLUTION
    xs = np.linalg.solve(A, b)
    xi = np.linalg.inv(A) @ b
    return {"n": n, "resid_solve": float(np.linalg.norm(A @ xs - b)), "resid_inv": float(np.linalg.norm(A @ xi - b)),
            "time_solve": best_time(lambda: np.linalg.solve(A, b)), "time_inv": best_time(lambda: np.linalg.inv(A) @ b)}
    # END SOLUTION

# %%
for n in U.pick([200, 800, 1600], [100, 200]):
    try:
        r = compare(n)
        print(f"n = {n:5d}  residual solve {r['resid_solve']:.2e}  inverse {r['resid_inv']:.2e}  "
              f"time solve {r['time_solve']:.4f} s  inverse {r['time_inv']:.4f} s")
    except NotImplementedError:
        print("compare(n) is not written yet")
        break

# %% [markdown]
# **Look at** both columns. The difference is modest here; it grows with the condition number.
# Solving never does worse, and you never need the inverse itself to apply it to one vector.

# %% [markdown]
# ## 7 · Record
#
# Write two or three sentences for the data analyst: which columns of the table to keep, which to
# drop and why, and what a model fitted on all five would report. (No code.)

# %% [markdown]
# *Your note:*

# %% [markdown]
# ## ✱ Stretch
#
# `scipy.linalg.null_space(X)` returns a basis of the vectors β with `X β = 0`. Compute it, scale
# it so its first entry is 1, and read the repeated column off it.

# %% [markdown]
# ## Check yourself
#
# <details><summary>1. Three vectors in three dimensions have rank 2. What does that say?</summary>
# They lie in one plane: one of them is a combination of the other two, and they span a plane, not
# all of space.</details>
#
# <details><summary>2. Why prefer solve to inverting and multiplying?</summary>
# Solving does less work and rounds less. The inverse is rarely needed itself; it is a step people
# take out of habit.</details>
#
# <details><summary>3. video_minutes is almost 7.5 × videos. Is the rank 3?</summary>
# No: "almost" is not "exactly", so the rank is still 4. But a column that is nearly a combination of
# others makes a fit unstable. Weeks 4 and 5 measure how unstable.</details>
#
# ## Carry forward
#
# Nothing to save. Week 4 starts again from `data/learner_features.csv`.
