# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1", "scikit-learn>=1.5", "scikit-image>=0.24"]
# ///
# %% [markdown]
# # Lab 7.1 · SVD image compression and PCA on public handwritten digits
#
# Week 7 · Session 1 (The SVD and principal components) · 105 min.
#
# The online learning platform stores many scanned images and wants to know how much a low-rank
# approximation can save. You compress a public test image (a brick wall, CC0, bundled with
# scikit-image) by truncated SVD and report error and storage by rank. Then you find the principal
# components of the public handwritten digits bundled with scikit-learn (8 × 8 images of the
# digits 0 to 9, CC BY 4.0) and interpret the leading ones. Both datasets are installed with the
# libraries: nothing is downloaded and nothing is committed to the course.
#
# **AI policy.** AI-tutor: hints, never answers (`labs/tutor/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Set up; an SVD of a two-by-two by hand (15 min)
# - [ ] 2 · The image and its singular values (10 min)
# - [ ] 3 · Rank-k approximations: error and storage (20 min)
# - [ ] 4 · The error is the tail of the singular values (10 min)
# - [ ] 5 · Principal components of the digits (20 min)
# - [ ] 6 · Explained variance and the leading components (20 min)
# - [ ] 7 · Record what the leading components show (10 min)
# - [ ] ✱ Stretch · Rebuild one digit from 5, 10 and 20 components and say which details come back first

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
# ## 1 · An SVD by hand
#
# `A = U Σ Vᵀ`: `V`'s columns are orthonormal input directions, `U`'s orthonormal output
# directions, and the singular values on Σ's diagonal say how much each direction is stretched.
# The singular values are the square roots of the eigenvalues of `AᵀA`. **On paper:** for
# `A = [[3, 0], [4, 5]]`, compute `AᵀA`, its eigenvalues, and the two singular values.

# %%
hand_sigma = None   # ✏️ (σ₁, σ₂), largest first

def reference_hand_sigma():
    # BEGIN SOLUTION
    return [45 ** 0.5, 5 ** 0.5]
    # END SOLUTION

# %%
A2 = np.array([[3.0, 0], [4, 5]])
Uu, s2, Vt = np.linalg.svd(A2)
print("AᵀA =", (A2.T @ A2).tolist(), "| singular values:", s2.round(4))
print("U Σ Vᵀ rebuilds A:", np.allclose(Uu @ np.diag(s2) @ Vt, A2))
if hand_sigma is None:
    try:
        hand_sigma = reference_hand_sigma()
    except NotImplementedError:
        pass
if hand_sigma is not None:
    U.check(np.allclose(hand_sigma, s2, atol=1e-3), "hand and code agree")

# %% [markdown]
# ## 2 · The image and its singular values
#
# Load the image as a 512 × 512 matrix of grey levels and compute its SVD. How fast do the
# singular values fall? Print the first five and the share of `‖A‖²_F` (the sum of all σ²) held
# by the first 10 and the first 50.

# %%
from skimage import data as skdata

img = skdata.brick().astype(float)
Ui, si, Vti = np.linalg.svd(img, full_matrices=False)
share = np.cumsum(si ** 2) / np.sum(si ** 2)
print("shape", img.shape, "| first five σ:", si[:5].round(1))
print("share of ‖A‖² in the first 10:", round(float(share[9]), 4), "| first 50:", round(float(share[49]), 4))

# %% [markdown]
# ## 3 · Rank-k approximations
#
# `A_k = U[:, :k] Σ[:k] Vᵀ[:k]` keeps the k largest layers. Write `rank_k(U, s, Vt, k)`. For
# k = 1, 5, 10, 20, 50 and 100, report the relative error `‖A − A_k‖_F / ‖A‖_F` and the storage
# `k (m + n + 1)` numbers against `m n` for the full image.

# %%
def rank_k(Uu, s, Vt, k):
    # BEGIN SOLUTION
    return (Uu[:, :k] * s[:k]) @ Vt[:k]
    # END SOLUTION

# %%
m, n = img.shape
ks = [1, 5, 10, 20, 50, 100]
errors = {}
for k in ks:
    try:
        Ak = rank_k(Ui, si, Vti, k)
        err = float(np.linalg.norm(img - Ak) / np.linalg.norm(img))
    except NotImplementedError:
        err = U.reference("w7_errors")[str(k)]
    errors[str(k)] = err
    print(f"k = {k:3d}  relative error {err:.4f}  storage {k * (m + n + 1) / (m * n):6.1%} of the full image")
U.keep("w7_errors", errors)

# %% [markdown]
# ## 4 · The error is the tail of the singular values
#
# The Eckart-Young theorem: no rank-k matrix is closer to A than `A_k`, and
# `‖A − A_k‖²_F = σ²ₖ₊₁ + … + σ²ₙ`. Check it for each k from the singular values alone.

# %%
for k in ks:
    tail = float(np.sqrt(np.sum(si[k:] ** 2)) / np.linalg.norm(img))
    print(f"k = {k:3d}  from the tail {tail:.4f}  measured {errors[str(k)]:.4f}")
U.check(all(abs(np.sqrt(np.sum(si[k:] ** 2)) / np.linalg.norm(img) - errors[str(k)]) < 1e-4 for k in ks),
        "the error equals the tail of the singular values")

# %% [markdown]
# **Look at** the storage column. Where does a rank-k copy stop saving space? (Solve
# `k (m + n + 1) < m n`.)

# %% [markdown]
# ## 5 · Principal components of the digits
#
# Load the digits: 1,797 images, each flattened to 64 pixel values. Centre the columns, take the
# SVD of the centred matrix `Xc = U Σ Vᵀ`: the rows of `Vᵀ` are the principal components and
# `σ² / (n − 1)` their variances. Write `pca(X)` returning the components and the share of
# variance each explains. Check against the eigenvectors of the covariance matrix (week 6).

# %%
from sklearn.datasets import load_digits

digits = load_digits()
X, labels = digits.data, digits.target
n_rows = U.pick(len(X), 600)
X, labels = X[:n_rows], labels[:n_rows]

def pca(X):
    # BEGIN SOLUTION
    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt, s ** 2 / np.sum(s ** 2)
    # END SOLUTION

# %%
try:
    comps, ratio = pca(X)
except NotImplementedError:
    Xc = X - X.mean(axis=0)
    w, V = np.linalg.eigh(np.cov(Xc, rowvar=False))
    comps, ratio = V[:, ::-1].T, w[::-1] / w.sum()
C = np.cov(X, rowvar=False)
w = np.linalg.eigvalsh(C)[::-1]
print(X.shape, "| variance share of the first five:", ratio[:5].round(4))
print("largest |σ²/(n−1) share − covariance eigenvalue share|:", f"{np.abs(ratio - w / w.sum()).max():.1e}")
U.check(np.allclose(ratio, w / w.sum(), atol=1e-8), "PCA through the SVD equals the covariance eigenvalues")

# %% [markdown]
# ## 6 · Explained variance and the leading components
#
# How many components hold 80% and 90% of the variance? Project the digits onto the first two
# components and print, for each digit 0 to 9, the mean of its two coordinates. Then print the
# first component as an 8 × 8 grid: which pixels does it weigh positively and which negatively?

# %%
cum = np.cumsum(ratio)
k80, k90 = int(np.searchsorted(cum, 0.8) + 1), int(np.searchsorted(cum, 0.9) + 1)
U.keep("w7_pca", {"first_two": [float(ratio[0]), float(ratio[1])], "k80": k80, "k90": k90})
print("components for 80%:", k80, "| for 90%:", k90, "| of 64")
Z = (X - X.mean(axis=0)) @ comps[:2].T
for d in range(10):
    print(f"digit {d}: mean position ({Z[labels == d, 0].mean():6.2f}, {Z[labels == d, 1].mean():6.2f})")
print("first component as 8 x 8 (rounded to 0.1):")
print(comps[0].reshape(8, 8).round(1))

# %% [markdown]
# **Look at** the digit means: which digits sit at the two ends of the first component? Then look
# at the grid. The sign of a component is arbitrary: flipping it flips every coordinate.

# %% [markdown]
# ## 7 · Record
#
# Three sentences: what rank you would store the image at and why, what share of the digits'
# variance two components hold, and what the first component separates. (No code.)

# %% [markdown]
# *Your note:*

# %% [markdown]
# ## ✱ Stretch
#
# Rebuild one digit from 5, 10 and 20 components (`mean + z[:k] @ comps[:k]`) and print the three
# 8 × 8 grids. Which details come back first?

# %% [markdown]
# ## Check yourself
#
# <details><summary>1. Why is the best rank-k approximation the first k layers of the SVD?</summary>
# The layers are orthogonal and ordered by σ; dropping a layer costs exactly its σ², so keeping the
# largest k costs the least (Eckart-Young).</details>
#
# <details><summary>2. Why centre before PCA?</summary>
# Without centring, the first component points at the mean of the data, not along its spread.</details>
#
# <details><summary>3. What is the link between the SVD of Xc and the eigenvectors of the covariance?</summary>
# Xcᵀ Xc = V Σ² Vᵀ: the right singular vectors are the covariance's eigenvectors, and σ²/(n − 1) are
# its eigenvalues.</details>
#
# ## Carry forward
#
# Nothing to save. Week 8 starts from `data/ratings.csv`.
