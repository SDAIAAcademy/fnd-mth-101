# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1"]
# ///
"""Every number on the slides that does not come from a lab run is computed here.

    cd labs
    uv run python _tools/slide_numbers.py [week]

The slide examples use their own vectors and matrices, never those of the week's lab or
problem set. Numbers quoted from a lab run (the debrief slides) come from that lab's solution.
"""
from __future__ import annotations

import sys

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def week1():
    x, y = np.array([1.0, 2, 2]), np.array([2.0, 1, 2])
    print("x.y", x @ y, "|x|", np.linalg.norm(x), "|y|", np.linalg.norm(y))
    c = x @ y / np.linalg.norm(x) / np.linalg.norm(y)
    print("cos", round(c, 4), "angle deg", round(np.degrees(np.arccos(c)), 1))
    z = 3 * x
    print("3x", z, "|3x|", np.linalg.norm(z), "x.3x", x @ z, "cos(x,3x)", x @ z / np.linalg.norm(x) / np.linalg.norm(z))
    for a, b in [((1, 0), (1, 1)), ((1, 0), (0, 1)), ((1, 0), (-1, 1))]:
        a, b = np.array(a, float), np.array(b, float)
        print("pair", a, b, "dot", a @ b, "angle", round(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))), 1))
    # the cone, on a toy 2-D example: three vectors that share a large common part
    common = np.array([5.0, 5.0])
    parts = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    M = common + parts
    N = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = N @ N.T
    print("toy vectors", M.tolist(), "pair cosines", C[np.triu_indices(3, 1)].round(3))
    Mc = M - M.mean(axis=0)
    Nc = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    print("centred", Mc.round(3).tolist(), "pair cosines", (Nc @ Nc.T)[np.triu_indices(3, 1)].round(3))
    # distance contrast with a seed the lab does not use
    for d in (2, 10, 100, 1000):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((500, d))
        dist = np.linalg.norm(X[1:] - X[0], axis=1)
        print("dim", d, "nearest", round(dist.min(), 2), "farthest", round(dist.max(), 2), "contrast", round((dist.max() - dist.min()) / dist.min(), 2))


def week2():
    P = np.array([[2.0, 1], [1, 3]])
    G = np.array([[1.0, 0, 1], [2, 1, 0]])
    print("P G =", (P @ G).tolist(), "shape", (P @ G).shape)
    q = np.array([1.0, 0.0])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    s = K @ q / np.sqrt(2)
    w = np.exp(s - s.max()); w /= w.sum()
    print("scores", s.round(4), "weights", w.round(4), "sum", w.sum())
    Rot = np.array([[0.0, -1], [1, 0]])
    Ref = np.array([[1.0, 0], [0, -1]])
    e = np.array([1.0, 1.0])
    print("Ref Rot e", Ref @ Rot @ e, "Rot Ref e", Rot @ Ref @ e)
    print("flops (3x4)(4x2)", 2 * 3 * 4 * 2, "bytes 1000x1000 float64 MB", 8 * 1000 * 1000 / 1e6)
    for n in (100, 1000, 10000):
        print("n", n, "GFLOPs", 2 * n**3 / 1e9, "MB per matrix", 8 * n * n / 1e6)


def week3():
    A = np.array([[2.0, 1, 1], [4, 3, 3], [8, 7, 9]])
    b = np.array([4.0, 10, 24])
    print("3x3 solution", np.linalg.solve(A, b))
    M = A.copy(); bb = b.copy()
    M[1] -= 2 * M[0]; bb[1] -= 2 * bb[0]
    M[2] -= 4 * M[0]; bb[2] -= 4 * bb[0]
    print("after step 1", M.tolist(), bb.tolist())
    f = M[2, 1] / M[1, 1]
    M[2] -= f * M[1]; bb[2] -= f * bb[1]
    print("after step 2", M.tolist(), bb.tolist())
    P = np.array([[4.0, 7], [2, 6]])
    print("inverse of [[4,7],[2,6]]", np.linalg.inv(P).round(4).tolist(), "det", round(np.linalg.det(P), 6))
    T = np.array([[2.0, 120, 8], [3, 180, 10], [1, 60, 5], [4, 240, 12]])
    print("course table hours, minutes, lessons rank", np.linalg.matrix_rank(T), "sv", np.linalg.svd(T, compute_uv=False).round(4))
    print("rank of (1,2),(2,4)", np.linalg.matrix_rank(np.array([[1.0, 2], [2, 4]])))


def week4():
    a, b = np.array([3.0, 1]), np.array([2.0, 4])
    p = (b @ a) / (a @ a) * a
    print("projection", p, "residual", b - p, "dot", (b - p) @ a)
    A = np.array([[1.0, 0], [0, 1], [1, 1]])
    P = A @ np.linalg.solve(A.T @ A, A.T)
    v = np.array([1.0, 2, 0])
    print("P", P.round(4).tolist(), "Pv", (P @ v).round(4), "resid", (v - P @ v).round(4))
    M = np.array([[3.0, 1], [4, 0]])
    q1 = M[:, 0] / np.linalg.norm(M[:, 0])
    r12 = q1 @ M[:, 1]
    w = M[:, 1] - r12 * q1
    print("q1", q1, "r12", r12, "w", w.round(4), "r22", np.linalg.norm(w), "q2", (w / np.linalg.norm(w)).round(4))


def week5():
    X = np.array([[1.0, -1], [1, 0], [1, 1]])
    y = np.array([1.0, 1, 4])
    b = np.linalg.solve(X.T @ X, X.T @ y)
    r = y - X @ b
    print("XtX", (X.T @ X).tolist(), "Xty", (X.T @ y).tolist(), "beta", b, "fitted", X @ b, "resid", r, "Xt r", X.T @ r)
    A = np.array([[1.0, 1], [1, 1.0001]])
    print("cond A", f"{np.linalg.cond(A):.4g}", "cond AtA", f"{np.linalg.cond(A.T @ A):.4g}")
    for rhs in ([2.0, 2.0001], [2.0, 2.0002]):
        print("rhs", rhs, "solution", np.linalg.solve(A, np.array(rhs)).round(4))


def week6():
    A = np.array([[4.0, 1], [1, 4]])
    w, V = np.linalg.eigh(A)
    print("eig [[4,1],[1,4]]", w, V.round(4).tolist())
    print("A (1,0)", A @ np.array([1.0, 0]), "A (1,1)", A @ np.array([1.0, 1]))
    D = np.array([[60.0, 65], [70, 72], [80, 79], [90, 88], [50, 56]])
    Dc = D - D.mean(axis=0)
    C = Dc.T @ Dc / (len(D) - 1)
    w, V = np.linalg.eigh(C)
    print("scores mean", D.mean(axis=0), "cov", C.round(2).tolist(), "eig", w.round(3), "lead", V[:, -1].round(4), "share", round(w[-1] / w.sum(), 4))
    # a 3-page graph: P to Q and R, Q to R, R to P
    M = np.array([[0, 0, 1.0], [0.5, 0, 0], [0.5, 1, 0]])
    x = np.full(3, 1 / 3)
    for k in range(1, 4):
        x = M @ x
        print("step", k, x.round(4))
    for _ in range(200):
        x = M @ x
    print("limit", x.round(4))
    print("damped steps to 1e-10 for d=0.85 roughly", round(np.log(1e-10) / np.log(0.85), 1))


def week7():
    A = np.array([[2.0, 2], [-1, 1]])
    print("AtA", (A.T @ A).tolist(), "sigma", np.linalg.svd(A, compute_uv=False).round(4))
    M = np.array([[5.0, 4, 1], [4, 5, 1], [1, 1, 3], [5, 5, 1]])
    s = np.linalg.svd(M, compute_uv=False)
    tot = np.sqrt((s ** 2).sum())
    print("M sigma", s.round(3), "rank1 rel err", round(np.sqrt((s[1:] ** 2).sum()) / tot, 4), "rank2 rel err", round(np.sqrt((s[2:] ** 2).sum()) / tot, 4))
    print("storage 1000x800 rank 20", 20 * (1000 + 800 + 1), "of", 1000 * 800, round(20 * 1801 / 800000, 4))
    print("break-even k for 512x512", 512 * 512 / (512 + 512 + 1))


WEEKS = {1: week1, 2: week2, 3: week3, 4: week4, 5: week5, 6: week6, 7: week7}

if __name__ == "__main__":
    for w in ([int(a) for a in sys.argv[1:]] or sorted(WEEKS)):
        print(f"== week {w}")
        WEEKS[w]()
