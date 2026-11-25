# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1"]
# ///
# %% [markdown]
# # Lab 6.1 · Power iteration and link ranking on a synthetic web graph
#
# Week 6 · Session 1 (Eigenvectors and link ranking) · 105 min.
#
# The help centre of the online learning platform has 40 pages (P01 to P40) that link to each
# other (`data/help_links.csv`, synthetic). The content lead wants the pages ranked by how likely a
# reader who follows links at random is to be on each one: the leading eigenvector of a link
# matrix. You find eigenpairs by hand, build the matrix, run power iteration, meet a dangling
# page, an island of archived pages and a link farm, and fix them with damping.
#
# **AI policy.** AI-tutor: hints, never answers (`labs/tutor/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Set up; eigenpairs of a two-by-two by hand (15 min)
# - [ ] 2 · Build the link matrix of the 40 pages (15 min)
# - [ ] 3 · Power iteration without damping (15 min)
# - [ ] 4 · Fix the dangling page and add damping (20 min)
# - [ ] 5 · Convergence, and a check with the library (15 min)
# - [ ] 6 · The link farm and a damping sweep (15 min)
# - [ ] 7 · Record the ranking for the content lead (10 min)
# - [ ] ✱ Stretch · Show that the number of steps follows the damping factor

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
# ## 1 · Eigenpairs by hand
#
# An eigenvector `v` of `A` keeps its direction: `A v = λ v`. **On paper:** for
# `S = [[2, 1], [1, 2]]`, solve `det(S − λI) = 0` for the two eigenvalues and find a unit
# eigenvector for the larger one. Then check with `np.linalg.eigh`, the routine for symmetric
# matrices.

# %%
hand_eig = None   # ✏️ (larger eigenvalue, smaller eigenvalue)

def reference_hand_eig():
    # BEGIN SOLUTION
    return [3, 1]
    # END SOLUTION

# %%
Sm = np.array([[2.0, 1], [1, 2]])
vals, vecs = np.linalg.eigh(Sm)
print("eigenvalues (ascending):", vals, "\neigenvectors as columns:\n", vecs)
print("S v for the larger:", Sm @ vecs[:, 1], "= λ v:", vals[1] * vecs[:, 1])
print("the eigenvectors are orthogonal:", round(float(vecs[:, 0] @ vecs[:, 1]), 12))
if hand_eig is None:
    try:
        hand_eig = reference_hand_eig()
    except NotImplementedError:
        pass
if hand_eig is not None:
    U.check(np.allclose(sorted(hand_eig), vals), "hand and code agree")

# %% [markdown]
# ## 2 · The link matrix
#
# Column `i` of the link matrix `M` describes a reader on page `i` who clicks one of its links at
# random: `M[j, i] = 1 / (number of links out of i)` for each link i → j. Every column with links
# adds up to 1. Write `link_matrix(edges, pages)`. Which column adds up to 0, and why?

# %%
edges = [(r["from_page"], r["to_page"]) for r in U.load_csv("help_links.csv")]
pages = sorted({p for e in edges for p in e})
idx = {p: k for k, p in enumerate(pages)}
n = len(pages)

def link_matrix(edges, pages):
    # BEGIN SOLUTION
    M = np.zeros((len(pages), len(pages)))
    out = {p: [b for a, b in edges if a == p] for p in pages}
    for p, targets in out.items():
        for q in targets:
            M[idx[q], idx[p]] = 1 / len(targets)
    return M
    # END SOLUTION

# %%
try:
    M = link_matrix(edges, pages)
except NotImplementedError:
    M = np.array(U.reference("w6_M"))
    s = M.sum(axis=0)
    M[:, s > 0] /= s[s > 0]   # the reference is stored to six decimals: restore exact column sums
U.keep("w6_M", M)
col_sums = M.sum(axis=0)
print(n, "pages,", len(edges), "links")
print("columns that do not add up to 1:", [pages[k] for k in np.where(abs(col_sums - 1) > 1e-9)[0]])
U.check(abs(col_sums.sum() - (n - 1)) < 1e-9, "every column adds up to 1 except the dangling page's")

# %% [markdown]
# ## 3 · Power iteration without damping
#
# Start from the uniform vector `x = (1/n, …, 1/n)` and repeat `x ← M x`. If `M` has a leading
# eigenvector with eigenvalue 1 and the others are smaller, `x` converges to it. Run 200 steps and
# print the total of `x` and the change in the last step.

# %%
def power_iteration(G, steps, x0=None):
    """Return the list of iterates x0, G x0, G² x0, ... (steps + 1 vectors)."""
    # BEGIN SOLUTION
    x = np.full(G.shape[0], 1 / G.shape[0]) if x0 is None else x0
    xs = [x]
    for _ in range(steps):
        x = G @ x
        xs.append(x)
    return xs
    # END SOLUTION

# %%
try:
    xs = power_iteration(M, 200)
except NotImplementedError:
    xs = [np.full(n, 1 / n)]
    for _ in range(200):
        xs.append(M @ xs[-1])
x = xs[-1]
island = [idx[p] for p in ["P36", "P37", "P38", "P39", "P40"]]
print("total after 200 steps:", round(float(x.sum()), 4))
print("change in the last step:", f"{np.abs(xs[-1] - xs[-2]).max():.2e}")
print("share on the island P36 to P40:", round(float(x[island].sum() / x.sum()), 4))

# %% [markdown]
# **Look at** the three numbers. Probability leaks out through the dangling page P35 at every
# step, and the island is a closed loop of five pages: the walk inside it cycles and never settles.

# %% [markdown]
# ## 4 · Fix the dangling page and add damping
#
# Two repairs. (1) A reader on a page with no links jumps to any page at random: replace the zero
# column by `1/n` everywhere. (2) With probability `1 − d` a reader jumps to a random page instead
# of clicking: `G = d M̂ + (1 − d)/n · 11ᵀ`, with `d = 0.85`. Write `google_matrix(M, d)` and run
# power iteration until the change is below 10⁻¹⁰.

# %%
def google_matrix(M, d=0.85):
    # BEGIN SOLUTION
    Mh = M.copy()
    Mh[:, Mh.sum(axis=0) == 0] = 1 / M.shape[0]
    return d * Mh + (1 - d) / M.shape[0] * np.ones_like(M)
    # END SOLUTION

def rank_pages(G, tol=1e-10, max_steps=1000):
    x = np.full(G.shape[0], 1 / G.shape[0])
    for step in range(1, max_steps + 1):
        new = G @ x
        if np.abs(new - x).sum() < tol:
            return new, step
        x = new
    return x, max_steps

# %%
try:
    G = google_matrix(M, 0.85)
except NotImplementedError:
    G = np.array(U.reference("w6_G"))
    G /= G.sum(axis=0)   # stored to six decimals: restore exact column sums
U.keep("w6_G", G)
r, steps = rank_pages(G)
order = np.argsort(-r)
top = [(pages[k], round(float(r[k]), 4)) for k in order[:6]]
U.keep("w6_top", top)
print("converged in", steps, "steps; total", round(float(r.sum()), 6))
print("top six:", top)
U.check(abs(r.sum() - 1) < 1e-9 and steps < 1000, "a probability vector, reached by power iteration")

# %% [markdown]
# ## 5 · Convergence, and a check with the library
#
# How many steps does the tolerance cost? Run `rank_pages` with tolerances 10⁻⁴, 10⁻⁷ and 10⁻¹⁰.
# Then take the eigenvector of `G` for its largest eigenvalue with `np.linalg.eig`, scale it to add
# up to 1, and compare.

# %%
for tol in (1e-4, 1e-7, 1e-10):
    print(f"tolerance {tol:.0e}: {rank_pages(G, tol)[1]} steps")
w, V = np.linalg.eig(G)
k = int(np.argmax(w.real))
v = np.real(V[:, k])
v = v / v.sum()
second = sorted(np.abs(w))[-2]
print("largest eigenvalue:", round(float(w[k].real), 10), "| second largest |λ|:", round(float(second), 4))
print("largest |power iteration − eigenvector|:", f"{np.abs(v - r).max():.1e}")
U.check(np.abs(v - r).max() < 1e-8, "power iteration found the leading eigenvector")

# %% [markdown]
# **Look at** the second-largest |λ|. Each step shrinks the error by about that factor; with
# damping it is at most `d`.

# %% [markdown]
# ## 6 · The link farm
#
# P32, P33 and P34 link to each other and to P31; P31 links back. Where do P31 and the farm rank?
# Sweep the damping factor over 0.5, 0.85, 0.95 and 0.99 and print the farm's total share
# (P31 to P34) and P31's position in the ranking.

# %%
farm = [idx[p] for p in ["P31", "P32", "P33", "P34"]]
sweep = {}
for d in (0.5, 0.85, 0.95, 0.99):
    try:
        rd, sd = rank_pages(google_matrix(M, d), max_steps=5000)
    except NotImplementedError:
        break
    pos = int(np.where(np.argsort(-rd) == idx["P31"])[0][0]) + 1
    sweep[str(d)] = {"farm_share": round(float(rd[farm].sum()), 4), "p31_position": pos, "steps": sd}
    print(f"d = {d}: farm share {rd[farm].sum():.4f}, P31 ranked {pos} of {n}, {sd} steps")
U.keep("w6_sweep", sweep)

# %% [markdown]
# ## 7 · Record
#
# Three sentences for the content lead: which pages rank highest and why, what the farm does to
# the ranking, and which damping you would use. (No code.)

# %% [markdown]
# *Your note:*

# %% [markdown]
# ## ✱ Stretch
#
# Plot or print the number of steps to 10⁻¹⁰ against d for d from 0.5 to 0.99, and compare with
# `log(10⁻¹⁰) / log(d)`.

# %% [markdown]
# ## Check yourself
#
# <details><summary>1. Why does power iteration find the leading eigenvector?</summary>
# Write the start as a combination of eigenvectors. Each step multiplies each part by its
# eigenvalue; the part with the largest |λ| grows fastest relative to the rest.</details>
#
# <details><summary>2. Why does damping guarantee convergence?</summary>
# Every entry of G is positive, so the leading eigenvalue 1 is unique and every other eigenvalue
# has absolute value at most d.</details>
#
# <details><summary>3. Is the ranking a measure of quality?</summary>
# No: it measures where links lead. A farm of pages linking to each other can lift a page without
# making it better.</details>
#
# ## Carry forward
#
# Nothing to save. Week 7 uses public data: a test image and handwritten digits.
