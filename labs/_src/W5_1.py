# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1", "scipy>=1.14"]
# ///
# %% [markdown]
# # Lab 5.1 · Least squares by normal equations and by QR
#
# Week 5 · Session 1 (Least squares and conditioning) · 105 min.
#
# The content lead of the online learning platform wants to predict how many hours a course takes
# to complete from its number of videos, its video minutes and its number of quizzes
# (`data/study_hours.csv`, 120 synthetic courses). Video minutes are nearly 7.5 × the number of
# videos: two nearly collinear features. You fit the same model three ways, measure condition
# numbers, nudge the data, and explain why the normal equations lose accuracy that QR keeps.
#
# **AI policy.** AI-tutor: hints, never answers (`labs/tutor/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Set up; a line by hand with the normal equations (15 min)
# - [ ] 2 · Fit with the normal equations (15 min)
# - [ ] 3 · Fit with QR and with the library (15 min)
# - [ ] 4 · Condition numbers of X and of XᵀX (15 min)
# - [ ] 5 · Nudge the data and watch the coefficients (15 min)
# - [ ] 6 · Compare the residuals (10 min)
# - [ ] 7 · Diagnose five practice cases (15 min)
# - [ ] 8 · Explain the difference (5 min)
# - [ ] ✱ Stretch · Centre and scale the columns, then measure again

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
from scipy.linalg import solve_triangular
import mth as U

print("quick mode:", U.QUICK)

# %% [markdown]
# ## 1 · A line by hand
#
# Fit `y = β₀ + β₁ x` to the points (0, 1), (1, 2), (2, 4). **On paper:** write `X` (a column of
# ones, then x), compute `XᵀX` and `Xᵀy`, and solve `XᵀX β = Xᵀy`. Write `beta_normal(X, y)`.

# %%
hand_beta = None   # ✏️ your hand result, (β₀, β₁)

def beta_normal(X, y):
    """Least squares by the normal equations."""
    # BEGIN SOLUTION
    return np.linalg.solve(X.T @ X, X.T @ y)
    # END SOLUTION

def reference_hand_beta():
    # BEGIN SOLUTION
    return [5 / 6, 1.5]
    # END SOLUTION

# %%
X3 = np.array([[1.0, 0], [1, 1], [1, 2]])
y3 = np.array([1.0, 2, 4])
try:
    b3 = beta_normal(X3, y3)
except NotImplementedError:
    b3 = np.linalg.lstsq(X3, y3, rcond=None)[0]
print("β:", b3.round(4), "| residual:", (y3 - X3 @ b3).round(4), "| Xᵀ residual:", (X3.T @ (y3 - X3 @ b3)).round(10))
if hand_beta is None:
    try:
        hand_beta = reference_hand_beta()
    except NotImplementedError:
        pass
if hand_beta is not None:
    U.check(np.allclose(hand_beta, b3, atol=1e-3), "hand and code agree")

# %% [markdown]
# ## 2 · The course data, normal equations
#
# Build `X` with four columns: ones, video_count, video_minutes, quizzes; `y` is
# hours_to_complete. Fit with `beta_normal`.

# %%
rows = U.load_csv("study_hours.csv")
X = np.array([[1.0, float(r["video_count"]), float(r["video_minutes"]), float(r["quizzes"])] for r in rows])
y = np.array([float(r["hours_to_complete"]) for r in rows])
names = ["intercept", "video_count", "video_minutes", "quizzes"]
try:
    b_ne = beta_normal(X, y)
except NotImplementedError:
    b_ne = np.array(U.reference("w5_b_ne"))
b_ne = np.asarray(U.keep("w5_b_ne", b_ne))
print({k: float(v) for k, v in zip(names, b_ne.round(4))})

# %% [markdown]
# ## 3 · QR, and the library
#
# With `X = Q R`, the least-squares β solves the triangular system `R β = Qᵀ y`: no `XᵀX` is ever
# formed. Write `beta_qr(X, y)` with `np.linalg.qr` and `solve_triangular`. Then compare with
# `np.linalg.lstsq` (which uses the SVD).

# %%
def beta_qr(X, y):
    # BEGIN SOLUTION
    Q, R = np.linalg.qr(X)
    return solve_triangular(R, Q.T @ y)
    # END SOLUTION

# %%
try:
    b_qr = beta_qr(X, y)
except NotImplementedError:
    b_qr = np.array(U.reference("w5_b_qr"))
b_qr = np.asarray(U.keep("w5_b_qr", b_qr))
b_ls = np.linalg.lstsq(X, y, rcond=None)[0]
for name, b in [("normal eq.", b_ne), ("QR", b_qr), ("lstsq", b_ls)]:
    print(f"{name:11s}", b.round(6))
print("largest |β_normal − β_QR|:", f"{np.abs(b_ne - b_qr).max():.1e}")

# %% [markdown]
# ## 4 · Condition numbers
#
# The condition number `cond(X)` bounds how much a small relative change in the data can change
# the solution, relative to its size. A working rule: about `log₁₀ cond` decimal digits are lost.
# Compute `cond(X)` and `cond(XᵀX)`. What is the relation between them?

# %%
def conditions(X):
    # BEGIN SOLUTION
    return float(np.linalg.cond(X)), float(np.linalg.cond(X.T @ X))
    # END SOLUTION

# %%
try:
    cx, cg = conditions(X)
except NotImplementedError:
    cx, cg = U.reference("w5_conditions")
U.keep("w5_conditions", [float(f"{cx:.6g}"), float(f"{cg:.6g}")])
print(f"cond(X) = {cx:.4g} | cond(XᵀX) = {cg:.4g} | cond(X)² = {cx**2:.4g}")
print(f"digits at risk: about {np.log10(cx):.1f} through QR, {np.log10(cg):.1f} through XᵀX, of about 16")
U.check(abs(np.log10(cg) - 2 * np.log10(cx)) < 0.05, "forming XᵀX squares the condition number")

# %% [markdown]
# ## 5 · Nudge the data
#
# (a) Add noise of standard deviation 0.01 hours to `y` (seed 5) and refit: how far do the
# coefficients of video_count and video_minutes move, compared with quizzes?
# (b) Make the collinearity tighter: replace video_minutes by `7.5 × video_count` plus noise of
# standard deviation 0.6, 0.06 and 0.006 (seed 9). For each, print cond(X) and the largest gap
# between the normal-equation and QR coefficients.
# (c) Fit the original data in 32-bit floats both ways.

# %%
rng = np.random.default_rng(5)
yp = y + rng.normal(0, 0.01, len(y))
try:
    shift = beta_normal(X, yp) - b_ne
except NotImplementedError:
    shift = np.linalg.lstsq(X, yp, rcond=None)[0] - b_ne
print("(a) change in β after a 0.01-hour nudge:", {k: float(v) for k, v in zip(names, shift.round(4))})

gaps = {}
for sd in (0.6, 0.06, 0.006):
    Xs = X.copy()
    Xs[:, 2] = 7.5 * X[:, 1] + np.random.default_rng(9).normal(0, sd, len(y))
    try:
        gap = float(np.abs(beta_normal(Xs, y) - beta_qr(Xs, y)).max())
    except NotImplementedError:
        gap = U.reference("w5_gaps")[str(sd)]["gap"]
    gaps[str(sd)] = {"cond": float(f"{np.linalg.cond(Xs):.3g}"), "gap": float(f"{gap:.3g}")}
    print(f"(b) noise {sd:<6} cond(X) {np.linalg.cond(Xs):9.3g}   largest |β_normal − β_QR| {gap:.1e}")
U.keep("w5_gaps", gaps)

X32, y32 = X.astype(np.float32), y.astype(np.float32)
try:
    b32 = {"normal eq.": beta_normal(X32, y32), "QR": beta_qr(X32, y32)}
except NotImplementedError:
    b32 = {"normal eq.": np.linalg.solve(X32.T @ X32, X32.T @ y32), "QR": b_qr.astype(np.float32)}
for k, b in b32.items():
    print(f"(c) float32 {k:11s}", np.asarray(b, dtype=float).round(4), "| residual norm", round(float(np.linalg.norm(y - X @ b)), 6))

# %% [markdown]
# ## 6 · Residuals
#
# Print `‖y − X β‖` for the three float64 fits. Then look at (a) to (c) again: which quantity is
# sensitive, the residual or the coefficients?

# %%
for name, b in [("normal eq.", b_ne), ("QR", b_qr), ("lstsq", b_ls)]:
    print(f"{name:11s} residual norm {np.linalg.norm(y - X @ b):.10f}")

# %% [markdown]
# ## 7 · Five practice cases
#
# For each matrix below, compute its rank and condition number and label it
# `"rank-deficient"` (rank below the number of columns), `"ill-conditioned"` (full rank, cond ≥ 10⁶:
# the normal equations would put 12 or more of 16 digits at risk; this threshold is the course's
# own working rule) or `"well-conditioned"`. Write `diagnose(M)`.

# %%
rng = np.random.default_rng(55)
base = rng.normal(0, 1, (60, 2))
cases = {
    "P1": np.column_stack([np.ones(60), base]),
    "P2": np.column_stack([np.ones(60), base[:, 0], base[:, 0] + rng.normal(0, 1e-6, 60)]),
    "P3": np.column_stack([np.ones(60), base[:, 0], 3 * base[:, 0] - 2]),
    "P4": np.column_stack([np.ones(60), 1e7 * base[:, 0], base[:, 1]]),
    "P5": np.column_stack([base, base[:, 0] - base[:, 1] + rng.normal(0, 0.5, 60)]),
}

def diagnose(M):
    # BEGIN SOLUTION
    if np.linalg.matrix_rank(M) < M.shape[1]:
        return "rank-deficient"
    return "ill-conditioned" if np.linalg.cond(M) >= 1e6 else "well-conditioned"
    # END SOLUTION

# %%
labels = {}
for k, M in cases.items():
    try:
        labels[k] = diagnose(M)
    except NotImplementedError:
        labels[k] = U.reference("w5_labels")[k]
    print(f"{k}: rank {np.linalg.matrix_rank(M)} of {M.shape[1]}, cond {np.linalg.cond(M):.2e} -> {labels[k]}")
labels = U.keep("w5_labels", labels)
U.check(labels["P3"] == "rank-deficient" and labels["P1"] == "well-conditioned", "the clear cases are labelled")

# %% [markdown]
# **Look at** P4. Its columns are not nearly dependent: one is measured on a scale ten million
# times larger. What would fix it without changing the information? (The stretch tries it.)

# %% [markdown]
# ## 8 · Explain
#
# Three sentences for the content lead: why the normal equations and QR agreed on the original
# data in float64, when they would not, and what the coefficients of video_count and video_minutes
# can and cannot tell her. (No code.)

# %% [markdown]
# *Your note:*

# %% [markdown]
# ## ✱ Stretch
#
# Centre every column except the intercept and divide by its standard deviation. Measure cond(X)
# for the course data and for P4 again.

# %% [markdown]
# ## Check yourself
#
# <details><summary>1. Why is least squares a projection?</summary>
# X β ranges over the span of the columns. The β that minimises ‖y − X β‖ makes X β the point of that
# span nearest to y: the orthogonal projection, with residual orthogonal to every column.</details>
#
# <details><summary>2. Why do residuals stay put while coefficients move?</summary>
# With nearly collinear columns, many different β give almost the same X β. The fit is stable;
# how it is shared between the two columns is not.</details>
#
# <details><summary>3. When do the normal equations fail visibly in float64?</summary>
# When cond(X)² approaches 10¹⁶, the reciprocal of float64's precision: about cond(X) of 10⁸.</details>
#
# ## Carry forward
#
# Nothing to save. Week 6 starts from `data/help_links.csv`.
