# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1"]
# ///
# %% [markdown]
# # Lab 2.1 · Loop and vectorised matrix products, timed and costed in FLOPs
#
# Week 2 · Session 1 (Matrix products, attention scores and FLOPs) · 105 min.
#
# The search lead of the online learning platform wants to know what scoring costs before the
# help centre grows. You compute products by hand, write attention scores as one product, then
# time three ways of multiplying matrices, count their floating-point operations (FLOPs) and
# bytes, and cost the platform's daily scoring.
#
# **AI policy.** AI-tutor: hints, never answers (`labs/tutor/TUTOR.md`). Timings depend on your
# computer: yours will differ from your neighbour's, and that is expected.
#
# **Tasks**
# - [ ] 1 · Set up; a product by hand, then in code (15 min)
# - [ ] 2 · Attention scores as one product, with a row softmax (15 min)
# - [ ] 3 · Composition: two maps, one matrix, and why order matters (10 min)
# - [ ] 4 · Time a triple-loop product on growing sizes (15 min)
# - [ ] 5 · Count FLOPs and bytes, and compute the rate achieved (15 min)
# - [ ] 6 · Time the row-loop and vectorised products (20 min)
# - [ ] 7 · Cost the platform's daily scoring (15 min)
# - [ ] ✱ Stretch · Repeat the timing in 32-bit floats and compare time and memory

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
# ## 1 · A product by hand, then in code
#
# Entry (i, j) of `AB` is the dot product of row i of `A` with column j of `B`. **On paper:**
#
# `A = [[1, 0, 2], [-1, 3, 1]]` (2 × 3) and `B = [[3, 1], [2, 1], [1, 0]]` (3 × 2).
#
# What is the shape of `AB`? Of `BA`? Compute `AB` by hand, write it in `hand_AB`, then write
# `matmul_loops(A, B)` with three nested loops and no NumPy product.

# %%
hand_AB = None   # ✏️ your hand result, as a list of rows

def matmul_loops(A, B):
    """C = A B with three nested Python loops."""
    # BEGIN SOLUTION
    n, k = len(A), len(A[0])
    m = len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C
    # END SOLUTION

def reference_hand_AB():
    # BEGIN SOLUTION
    return [[5, 1], [4, 2]]
    # END SOLUTION

# %%
A = np.array([[1.0, 0, 2], [-1, 3, 1]])
B = np.array([[3.0, 1], [2, 1], [1, 0]])
try:
    C_loop = np.array(matmul_loops(A.tolist(), B.tolist()))
except NotImplementedError:
    C_loop = A @ B
print("A @ B =\n", A @ B, "\nshape of B @ A:", (B @ A).shape)
U.check(np.allclose(C_loop, A @ B), "the loop product equals A @ B")
if hand_AB is None:
    try:
        hand_AB = reference_hand_AB()
    except NotImplementedError:
        pass
if hand_AB is not None:
    U.check(np.allclose(hand_AB, A @ B), "hand and code agree")

# %% [markdown]
# ## 2 · Attention scores as one product
#
# In an attention layer, each of `q` queries is compared with each of `k` keys by a dot product.
# Stack the queries as rows of `Q` (q × d) and the keys as rows of `K` (k × d): every score at once
# is `S = Q Kᵀ / √d`, a q × k matrix. A softmax along each row turns the scores into weights that
# add up to 1. Write `attention_weights(Q, K)`.

# %%
def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)  # subtracting the row maximum changes nothing but avoids overflow
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)

def attention_weights(Q, K):
    # BEGIN SOLUTION
    return softmax_rows(Q @ K.T / np.sqrt(Q.shape[1]))
    # END SOLUTION

# %%
rng = np.random.default_rng(21)
Q = rng.standard_normal((4, 8))
K = rng.standard_normal((6, 8))
try:
    Wt = attention_weights(Q, K)
except NotImplementedError:
    Wt = np.array(U.reference("w2_attention"))
Wt = U.keep("w2_attention", Wt)
Wt = np.asarray(Wt)
print("weights, one row per query:\n", Wt)
U.check(Wt.shape == (4, 6) and np.allclose(Wt.sum(axis=1), 1, atol=1e-5), "4 x 6 weights, each row adds up to 1")
print("score (0, 0) as one dot product:", round(float(Q[0] @ K[0] / np.sqrt(8)), 4))

# %% [markdown]
# ## 3 · Composition: two maps, one matrix
#
# `R` turns the plane a quarter turn anticlockwise; `S` stretches the first coordinate by 2.
# Applying `R` first and `S` second is the single matrix `S R` (read right to left). Predict on
# paper where `(1, 0)` goes under `S R` and under `R S`, then check.

# %%
R = np.array([[0.0, -1], [1, 0]])
S = np.array([[2.0, 0], [0, 1]])
e1 = np.array([1.0, 0])
print("S R e1 =", S @ R @ e1, "| R S e1 =", R @ S @ e1)
U.check(not np.allclose(S @ R, R @ S), "the order of the maps matters: S R differs from R S")

# %% [markdown]
# ## 4 · Time a triple-loop product
#
# Time `matmul_loops` on random n × n matrices for growing n. Each time n doubles, by what factor
# does the time grow?

# %%
def best_time(f, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best

sizes_loop = U.pick([32, 64, 128], [16, 32, 64])
loop_times = {}
for n in sizes_loop:
    X, Y = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    Xl, Yl = X.tolist(), Y.tolist()
    try:
        loop_times[n] = best_time(lambda: matmul_loops(Xl, Yl), repeats=1)
    except NotImplementedError:
        loop_times[n] = float("nan")
    print(f"n = {n:4d}  loops: {loop_times[n]:.4f} s")

# %% [markdown]
# ## 5 · FLOPs, bytes and the rate achieved
#
# Each of the n·m entries of an (n × k)(k × m) product needs k multiplications and k additions:
# **2·n·k·m FLOPs**. Each float64 number takes 8 bytes. Write `flops(n, k, m)` and
# `bytes_needed(n, k, m)` (the two inputs and the output), then the rate of the loop version in
# FLOPs per second.

# %%
def flops(n, k, m):
    # BEGIN SOLUTION
    return 2 * n * k * m
    # END SOLUTION

def bytes_needed(n, k, m, itemsize=8):
    # BEGIN SOLUTION
    return itemsize * (n * k + k * m + n * m)
    # END SOLUTION

# %%
try:
    f128, b128 = flops(128, 128, 128), bytes_needed(128, 128, 128)
except NotImplementedError:
    f128, b128 = U.reference("w2_flops_bytes")
U.keep("w2_flops_bytes", [f128, b128])
U.check(f128 == 4_194_304 and b128 == 393_216, "a 128 x 128 product: 4,194,304 FLOPs and 393,216 bytes")
for n, t in loop_times.items():
    try:
        print(f"n = {n:4d}  loops: {2 * n**3 / t / 1e6:8.1f} million FLOPs per second")
    except ZeroDivisionError:
        pass

# %% [markdown]
# ## 6 · Row-loop and vectorised products
#
# Two faster versions: a loop over rows, each row a NumPy vector-matrix product (`X[i] @ Y`), and
# one vectorised call `X @ Y`. Time both on larger sizes and compute each one's FLOPs per second.

# %%
def matmul_rows(X, Y):
    # BEGIN SOLUTION
    out = np.empty((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        out[i] = X[i] @ Y
    return out
    # END SOLUTION

# %%
sizes = U.pick([128, 256, 512, 1024], [64, 128, 256])
rows = []
for n in sizes:
    X, Y = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    try:
        U.check(np.allclose(matmul_rows(X, Y), X @ Y), f"row loop equals X @ Y at n = {n}")
        t_rows = best_time(lambda: matmul_rows(X, Y))
    except NotImplementedError:
        t_rows = float("nan")
    t_vec = best_time(lambda: X @ Y)
    rows.append((n, t_rows, t_vec))
    print(f"n = {n:5d}  rows: {t_rows:.5f} s  vectorised: {t_vec:.5f} s  "
          f"vectorised rate: {2 * n**3 / t_vec / 1e9:6.2f} GFLOP/s")
n0 = sizes_loop[-1]
X, Y = rng.standard_normal((n0, n0)), rng.standard_normal((n0, n0))
speedup = loop_times[n0] / best_time(lambda: X @ Y)
print(f"at n = {n0}, the vectorised product is about {speedup:,.0f} times faster than the triple loop")

# %% [markdown]
# **Look at** the rates. The triple loop runs one Python operation per multiplication; the
# vectorised call hands the whole product to compiled code that uses the processor's vector
# units and caches. The arithmetic is the same: the FLOP count does not change, the rate does.

# %% [markdown]
# ## 7 · Cost the platform's daily scoring
#
# The search lead's plan: 50,000 questions a day, each scored against every article. Today: 60
# articles and 50-dimensional vectors. Planned: 5,000 articles and 384-dimensional vectors.
# For both, compute the FLOPs of one day's scores (one product `Q Aᵀ`), the bytes of the article
# matrix in float64, and the time at the vectorised rate you measured at the largest size.

# %%
rate = 2 * rows[-1][0] ** 3 / rows[-1][2]
plan = {}
for name, n_art, dim in [("today", 60, 50), ("planned", 5000, 384)]:
    try:
        f = flops(50_000, dim, n_art)
        b = 8 * n_art * dim
    except NotImplementedError:
        f, b = 2 * 50_000 * dim * n_art, 8 * n_art * dim
    plan[name] = (f, b)
    print(f"{name:8s} {f / 1e9:10.2f} GFLOPs a day | article matrix {b / 1e6:7.2f} MB | about {f / rate:8.3f} s at your rate")
U.keep("w2_plan", {k: list(v) for k, v in plan.items()})
U.check(plan["planned"][0] / plan["today"][0] == 640, "the planned scoring costs 640 times today's")

# %% [markdown]
# ## ✱ Stretch
#
# Repeat step 6 with `X.astype(np.float32)`. How do time and memory change? What do you lose?

# %% [markdown]
# ## Check yourself
#
# <details><summary>1. What is the shape of (5 × 3)(3 × 7)(7 × 2), and how many FLOPs does it take from left to right?</summary>
# 5 × 2. Left to right: 2·5·3·7 = 210 for the first product (5 × 7), then 2·5·7·2 = 140: 350 FLOPs.</details>
#
# <details><summary>2. Why does doubling n multiply the loop time by about eight?</summary>
# The product takes 2n³ FLOPs; doubling n multiplies n³ by 8, and the loop's rate stays about the same.</details>
#
# <details><summary>3. Does vectorising reduce the number of FLOPs?</summary>
# No. It changes how fast they run, not how many there are.</details>
#
# ## Carry forward
#
# Nothing to save. Week 3 starts from `data/learner_features.csv`.
