# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1"]
# ///
# %% [markdown]
# # Lab 4.1 · Projections and QR by Gram-Schmidt
#
# Week 4 · Session 1 (Orthogonal projection and QR) · 105 min.
#
# The data analyst wants to know how much of the learners' weekly study minutes can be explained
# as a combination of quizzes and forum posts: that is a projection. You project by hand and in
# code, check the properties of a projection matrix, build a QR factorisation by Gram-Schmidt,
# and watch it lose orthogonality when columns are nearly dependent: the warning for week 5.
#
# **AI policy.** AI-tutor: hints, never answers (`labs/tutor/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Set up; project onto a line by hand, then in code (15 min)
# - [ ] 2 · The projection matrix and its residual (15 min)
# - [ ] 3 · Project weekly minutes onto two features (15 min)
# - [ ] 4 · Classical Gram-Schmidt QR (20 min)
# - [ ] 5 · Check against the library (10 min)
# - [ ] 6 · Nearly dependent columns: classical against modified (20 min)
# - [ ] 7 · Record what this means for week 5 (10 min)
# - [ ] ✱ Stretch · Explain the columns whose signs differ from the library's Q

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
# ## 1 · Project onto a line
#
# The projection of `b` onto the line through `a` is `(b·a / a·a) a`. **On paper:** project
# `b = (3, 1)` onto `a = (1, 2)`, and compute the residual `b − p`. Is the residual orthogonal to
# `a`? Then write `project_line(b, a)`.

# %%
hand_p = None   # ✏️ your hand projection, (x, y)

def project_line(b, a):
    # BEGIN SOLUTION
    return (b @ a) / (a @ a) * a
    # END SOLUTION

def reference_hand_p():
    # BEGIN SOLUTION
    return [1, 2]
    # END SOLUTION

# %%
a, b = np.array([1.0, 2]), np.array([3.0, 1])
try:
    p = project_line(b, a)
except NotImplementedError:
    p = np.array(U.reference("w4_proj_line"))
p = np.asarray(U.keep("w4_proj_line", p))
print("projection:", p, "| residual:", b - p, "| residual · a =", (b - p) @ a)
U.check(abs((b - p) @ a) < 1e-6, "the residual is orthogonal to a")
if hand_p is None:
    try:
        hand_p = reference_hand_p()
    except NotImplementedError:
        pass
if hand_p is not None:
    U.check(np.allclose(hand_p, p), "hand and code agree")

# %% [markdown]
# ## 2 · The projection matrix
#
# Onto the span of the columns of `A` (independent columns), the projection matrix is
# `P = A (AᵀA)⁻¹ Aᵀ`. Write `projection_matrix(A)` without calling `inv`: solve `(AᵀA) Z = Aᵀ`
# for `Z`, then `P = A Z`. Check that `P² = P`, `Pᵀ = P` and that the residual of any vector is
# orthogonal to the columns.

# %%
def projection_matrix(A):
    # BEGIN SOLUTION
    return A @ np.linalg.solve(A.T @ A, A.T)
    # END SOLUTION

# %%
A3 = np.array([[1.0, 0], [1, 1], [1, 2]])
try:
    P = projection_matrix(A3)
except NotImplementedError:
    P = np.array(U.reference("w4_projmat"))
P = np.asarray(U.keep("w4_projmat", P))
v = np.array([6.0, 0, 0])
print("P =\n", P)
U.check(np.allclose(P @ P, P, atol=1e-5) and np.allclose(P, P.T, atol=1e-5), "P squared is P, and P is symmetric")
U.check(np.allclose(A3.T @ (v - P @ v), 0, atol=1e-4), "the residual of v is orthogonal to both columns")

# %% [markdown]
# ## 3 · Project weekly minutes onto two features
#
# Project the 400 learners' weekly minutes `y` onto the span of the columns quizzes and posts.
# (Do not build a 400 × 400 `P`: compute the coefficients by solving `(AᵀA) c = Aᵀy`, then
# `p = A c`.) What fraction of `‖y‖²` does the projection keep?

# %%
rows = U.load_csv("learner_features.csv")
y = np.array([float(r["weekly_minutes"]) for r in rows])
A = np.array([[float(r["quizzes"]), float(r["posts"])] for r in rows])

def project_onto_columns(y, A):
    # BEGIN SOLUTION
    c = np.linalg.solve(A.T @ A, A.T @ y)
    return A @ c
    # END SOLUTION

# %%
try:
    py = project_onto_columns(y, A)
except NotImplementedError:
    py = A @ np.linalg.lstsq(A, y, rcond=None)[0]
kept = U.keep("w4_kept", float(py @ py / (y @ y)))
print(f"share of ‖y‖² kept by the projection: {kept:.4f}")
print("largest |Aᵀ (y − p)|:", f"{np.abs(A.T @ (y - py)).max():.1e}")
U.check(np.abs(A.T @ (y - py)).max() < 1e-6, "the residual is orthogonal to quizzes and posts")

# %% [markdown]
# **Look at** the share. Projections through the origin keep a large share of `‖y‖²` easily,
# because the mean itself is large. Week 5 adds an intercept column and measures fit properly.

# %% [markdown]
# ## 4 · Classical Gram-Schmidt
#
# Build `Q` with orthonormal columns and an upper-triangular `R` with `A = Q R`. Column by column:
# subtract from column j its projections on the earlier `q`s, using the original column
# (`R[i, j] = qᵢ · aⱼ`), then divide by the norm of what is left.

# %%
def qr_classical(A):
    # BEGIN SOLUTION
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    for j in range(n):
        v = A[:, j].astype(float).copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R
    # END SOLUTION

# %% [markdown]
# ## 5 · Check against the library
#
# On the 400 × 3 matrix of videos, video minutes and quizzes, check `QᵀQ = I`, `QR = A`, and
# compare with `np.linalg.qr`.

# %%
X = np.array([[float(r[c]) for c in ["videos", "video_minutes", "quizzes"]] for r in rows])
try:
    Q, R = qr_classical(X)
except NotImplementedError:
    Q, R = np.linalg.qr(X)
print("largest |QᵀQ − I|:", f"{np.abs(Q.T @ Q - np.eye(3)).max():.1e}", "| largest |QR − A|:", f"{np.abs(Q @ R - X).max():.1e}")
Ql, Rl = np.linalg.qr(X)
print("diagonal of R, yours:", np.diag(R).round(3), "| library's:", np.diag(Rl).round(3))
U.check(np.allclose(Q.T @ Q, np.eye(3), atol=1e-10) and np.allclose(Q @ R, X), "Q is orthonormal and Q R = A")

# %% [markdown]
# ## 6 · Nearly dependent columns
#
# Three columns that differ only by a tiny ε: `A_ε = [[1, 1, 1], [ε, 0, 0], [0, ε, 0], [0, 0, ε]]`.
# Write `qr_modified(A)`, which subtracts each projection from the **updated** vector
# (`R[i, j] = qᵢ · v`). For ε = 10⁻⁴, 10⁻⁷ and 10⁻⁸, print the condition number of `A_ε` and the
# loss of orthogonality `max |QᵀQ − I|` for classical, modified and the library (Householder).

# %%
def qr_modified(A):
    # BEGIN SOLUTION
    m, n = A.shape
    Q, R = A.astype(float).copy(), np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] /= R[j, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j] @ Q[:, k]
            Q[:, k] -= R[j, k] * Q[:, j]
    return Q, R
    # END SOLUTION

# %%
def loss(Q):
    return float(np.abs(Q.T @ Q - np.eye(Q.shape[1])).max())

table = {}
for eps in (1e-4, 1e-7, 1e-8):
    Ae = np.vstack([np.ones(3), eps * np.eye(3)])
    row = {"cond": float(np.linalg.cond(Ae)), "householder": loss(np.linalg.qr(Ae)[0])}
    for name, f in [("classical", qr_classical), ("modified", qr_modified)]:
        try:
            row[name] = loss(f(Ae)[0])
        except NotImplementedError:
            row[name] = U.reference("w4_loss")[str(eps)][name]
    table[str(eps)] = row
    print(f"ε = {eps:.0e}  cond {row['cond']:.2e}  classical {row['classical']:.1e}  "
          f"modified {row['modified']:.1e}  householder {row['householder']:.1e}")
U.keep("w4_loss", {k: {kk: float(f"{vv:.3g}") for kk, vv in v.items()} for k, v in table.items()})
U.check(table["1e-08"]["classical"] > 1e-3 > table["1e-08"]["modified"], "classical Gram-Schmidt loses orthogonality first")

# %% [markdown]
# ## 7 · Record
#
# In two or three sentences: what does a large loss of orthogonality mean for a fit that uses `Q`,
# and which QR would you use in week 5? (No code.)

# %% [markdown]
# *Your note:*

# %% [markdown]
# ## ✱ Stretch
#
# Compare your `Q` with the library's on the learner matrix. Some columns differ only in sign. Why
# is a QR factorisation unique only up to those signs?

# %% [markdown]
# ## Check yourself
#
# <details><summary>1. Why is the residual of a projection orthogonal to the subspace?</summary>
# If it had a component along the subspace, moving the point along that component would bring it
# closer: the nearest point leaves only the orthogonal part.</details>
#
# <details><summary>2. What does R hold in A = Q R?</summary>
# The coordinates of each column of A in the orthonormal basis Q. It is upper triangular because
# column j uses only q₁ to qⱼ.</details>
#
# <details><summary>3. Classical and modified Gram-Schmidt are equal in exact arithmetic. Why do they differ here?</summary>
# Rounding. Classical projects every time from the original column; when columns are nearly equal,
# the small differences that remain are swamped by rounding. Modified works on the updated vector
# and keeps more of them.</details>
#
# ## Carry forward
#
# Nothing to save. Week 5 starts from `data/study_hours.csv`.
