"""The auto-checker of the weekly problem sets (PS1 to PS8).

The last cell of each problem-set notebook calls `pscheck.check(n, globals())`. It prints one
line per item and a score. It checks two things, and stores no answer:

- agreement: each hand answer agrees, within a tolerance, with the result of your own code on
  the same small case (the course's evidence for outcome LO4);
- correctness of the code: your functions are run on random matrices and checked against
  properties (A x = b, Qᵀ Q = I, a symmetric matrix of cosines with ones on its diagonal) or
  against the library.

A hand answer left as None scores nothing; so does a function that is not written yet.
"""
from __future__ import annotations

import numpy as np

TOL = 1e-3


def _close(a, b, tol=TOL) -> bool:
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=tol))
    except (TypeError, ValueError):
        return False


def _run(fn, *args):
    try:
        return fn(*args), None
    except NotImplementedError:
        return None, "not written yet"
    except Exception as e:  # noqa: BLE001 - a learner's error is a result, not a crash
        return None, f"{type(e).__name__}: {e}"


def _agree(ns, key, fn_name, *args, transform=None):
    """A hand answer against the learner's own function on the same case."""
    hand = ns.get("hand", {}).get(key)
    if hand is None:
        return False, "hand answer missing"
    fn = ns.get(fn_name)
    if fn is None:
        return False, f"function {fn_name} missing"
    got, err = _run(fn, *args)
    if err:
        return False, err
    if transform:
        got = transform(got)
    return _close(hand, got), "hand and code differ" if not _close(hand, got) else ""


def _code(ns, fn_name, test):
    fn = ns.get(fn_name)
    if fn is None:
        return False, f"function {fn_name} missing"
    try:
        ok = bool(test(fn))
        return ok, "" if ok else "wrong on a random test case"
    except NotImplementedError:
        return False, "not written yet"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


RNG = np.random.default_rng(2027)


# ---------------------------------------------------------------- PS1 vectors and similarity
def _ps1(ns):
    p, q, r = ns["p"], ns["q"], ns["r"]
    yield "1a hand p·q agrees with dot(p, q)", _agree(ns, "dot_pq", "dot", p, q)
    yield "1b hand ‖p‖ agrees with norm(p)", _agree(ns, "norm_p", "norm", p)
    yield "1c hand cos(p, q) agrees with cosine(p, q)", _agree(ns, "cos_pq", "cosine", p, q)
    yield "2 hand unit vector of r agrees with unit(r)", _agree(ns, "unit_r", "unit", r)
    X = RNG.standard_normal((7, 5))
    yield "3 dot and norm match the library", _code(ns, "dot", lambda f: _close(f(X[0], X[1]), X[0] @ X[1]))
    yield "4 cosine_matrix: symmetric, ones on the diagonal, right values", _code(
        ns, "cosine_matrix",
        lambda f: _close(f(X), (X / np.linalg.norm(X, axis=1, keepdims=True)) @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T))
    def nearest_ok(f):
        N = X / np.linalg.norm(X, axis=1, keepdims=True)
        C = N @ N.T
        np.fill_diagonal(C, -np.inf)
        return all(int(f(X, i)) == int(np.argmax(C[i])) for i in range(len(X)))
    yield "5 nearest(X, i) returns the most similar other row", _code(ns, "nearest", nearest_ok)


# ---------------------------------------------------------------- PS2 products, composition, FLOPs
def _ps2(ns):
    A, B, F, S, v = ns["A"], ns["B"], ns["F"], ns["S"], ns["v"]
    yield "1 hand A B agrees with matmul(A, B)", _agree(ns, "AB", "matmul", A, B)
    yield "2 hand FLOPs of a (4 x 3)(3 x 5) product agree with flops(4, 3, 5)", _agree(ns, "flops_435", "flops", 4, 3, 5)
    yield "3 hand image of v under F then S agrees with compose(S, F) @ v", _agree(
        ns, "SFv", "compose", S, F, transform=lambda M: np.asarray(M) @ v)
    X, Y = RNG.standard_normal((4, 6)), RNG.standard_normal((6, 3))
    yield "4 matmul matches the library on a random case", _code(ns, "matmul", lambda f: _close(f(X, Y), X @ Y))
    yield "5 flops follows 2 n k m", _code(ns, "flops", lambda f: all(f(n, k, m) == 2 * n * k * m for n, k, m in [(3, 7, 2), (10, 1, 4)]))
    def chain_ok(f):
        return f([(5, 3), (3, 7), (7, 2)]) == 2 * 5 * 3 * 7 + 2 * 5 * 7 * 2 and f([(2, 9), (9, 4)]) == 2 * 2 * 9 * 4
    yield "6 chain_cost counts a left-to-right chain", _code(ns, "chain_cost", chain_ok)


# ---------------------------------------------------------------- PS3 elimination, inverse, rank
def _ps3(ns):
    M, b, R = ns["M"], ns["b"], ns["R"]
    yield "1 hand inverse of M agrees with inverse2(M)", _agree(ns, "M_inv", "inverse2", M)
    yield "2 hand solution of M x = b agrees with solve(M, b)", _agree(ns, "x", "solve", M, b)
    yield "3 hand rank of R agrees with rank(R)", _agree(ns, "rank_R", "rank", R)
    def inv_ok(f):
        for _ in range(3):
            A = RNG.standard_normal((2, 2)) + 2 * np.eye(2)
            if not _close(np.asarray(f(A)) @ A, np.eye(2)):
                return False
        return True
    yield "4 inverse2 gives M⁻¹ M = I on random matrices", _code(ns, "inverse2", inv_ok)
    def solve_ok(f):
        A = RNG.standard_normal((5, 5)) + 3 * np.eye(5)
        rhs = RNG.standard_normal(5)
        return _close(A @ np.asarray(f(A, rhs)), rhs)
    yield "5 solve gives A x = b on a random 5 x 5 system", _code(ns, "solve", solve_ok)
    def rank_ok(f):
        L = RNG.standard_normal((6, 2)) @ RNG.standard_normal((2, 4))
        return int(f(L)) == 2 and int(f(np.eye(3))) == 3
    yield "6 rank finds the rank of a random rank-2 matrix", _code(ns, "rank", rank_ok)


# ---------------------------------------------------------------- PS4 projections and Gram-Schmidt
def _ps4(ns):
    a, b, G = ns["a"], ns["b"], ns["G"]
    yield "1 hand projection of b on a agrees with project_line(b, a)", _agree(ns, "proj_ba", "project_line", b, a)
    yield "2 hand second column of Q agrees with gram_schmidt(G)", _agree(
        ns, "q2", "gram_schmidt", G, transform=lambda QR: np.asarray(QR[0])[:, 1])
    def pl_ok(f):
        u, w = RNG.standard_normal(4), RNG.standard_normal(4)
        pr = np.asarray(f(w, u))
        return _close((w - pr) @ u, 0) and _close(pr, (w @ u) / (u @ u) * u)
    yield "3 project_line leaves a residual orthogonal to the line", _code(ns, "project_line", pl_ok)
    def pm_ok(f):
        A = RNG.standard_normal((6, 2))
        P = np.asarray(f(A))
        return _close(P @ P, P) and _close(P, P.T) and _close(P @ A, A)
    yield "4 projection_matrix: P squared is P, P symmetric, P A = A", _code(ns, "projection_matrix", pm_ok)
    def gs_ok(f):
        A = RNG.standard_normal((7, 4))
        Q, R = (np.asarray(x) for x in f(A))
        return _close(Q.T @ Q, np.eye(4)) and _close(Q @ R, A) and _close(np.tril(R, -1), 0)
    yield "5 gram_schmidt: orthonormal Q, triangular R, Q R = A", _code(ns, "gram_schmidt", gs_ok)


# ---------------------------------------------------------------- PS5 least squares and conditioning
def _ps5(ns):
    X, y, D = ns["X"], ns["y"], ns["D"]
    yield "1 hand β by the normal equations agrees with fit_normal(X, y)", _agree(ns, "beta", "fit_normal", X, y)
    yield "2 hand condition number of D agrees with condition(D)", _agree(ns, "cond_D", "condition", D)
    yield "3 hand condition number of DᵀD agrees with condition(DᵀD)", _agree(ns, "cond_DtD", "condition", D.T @ D)
    def qr_ok(f):
        A = RNG.standard_normal((20, 4))
        b = RNG.standard_normal(20)
        return _close(f(A, b), np.linalg.lstsq(A, b, rcond=None)[0])
    yield "4 fit_qr matches the library's least squares", _code(ns, "fit_qr", qr_ok)
    def cond_ok(f):
        A = RNG.standard_normal((8, 3)) @ np.diag([1, 10, 1000])
        return _close(f(A), np.linalg.cond(A), tol=1e-6)
    yield "5 condition gives the ratio of the largest to smallest singular value", _code(ns, "condition", cond_ok)
    def resid_ok(f):
        A = RNG.standard_normal((15, 3))
        b = RNG.standard_normal(15)
        r = np.asarray(f(A, b))
        return _close(A.T @ r, 0) and _close(r, b - A @ np.linalg.lstsq(A, b, rcond=None)[0])
    yield "6 residual(A, b) is orthogonal to the columns", _code(ns, "residual", resid_ok)


# ---------------------------------------------------------------- PS6 eigenpairs, covariance, power iteration
def _ps6(ns):
    A, Z = ns["A"], ns["Z"]
    yield "1 hand eigenvalues of A agree with eigenvalues2(A)", _agree(
        ns, "eig_A", "eigenvalues2", A, transform=lambda e: sorted(np.asarray(e, dtype=float)))
    yield "2 hand covariance of Z agrees with covariance(Z)", _agree(ns, "cov_Z", "covariance", Z)
    def ev2_ok(f):
        B = RNG.standard_normal((2, 2))
        B = B + B.T
        return _close(sorted(np.asarray(f(B), dtype=float)), np.linalg.eigvalsh(B))
    yield "3 eigenvalues2 is right on a random symmetric 2 x 2", _code(ns, "eigenvalues2", ev2_ok)
    def cov_ok(f):
        Y = RNG.standard_normal((30, 4))
        return _close(f(Y), np.cov(Y, rowvar=False))
    yield "4 covariance matches the library on random data", _code(ns, "covariance", cov_ok)
    def pi_ok(f):
        B = RNG.standard_normal((6, 6))
        B = B @ B.T + np.diag([20, 0, 0, 0, 0, 0])
        lam, v = f(B, 500)
        v = np.asarray(v, dtype=float)
        return _close(lam, np.linalg.eigvalsh(B)[-1], tol=1e-4) and _close(B @ v, lam * v, tol=1e-3)
    yield "5 dominant(B, steps) finds the largest eigenvalue and its vector", _code(ns, "dominant", pi_ok)


CHECKS = {1: _ps1, 2: _ps2, 3: _ps3, 4: _ps4, 5: _ps5, 6: _ps6}


def check(n: int, ns: dict) -> int:
    rows = list(CHECKS[n](ns))
    score = 0
    for label, (ok, why) in rows:
        score += ok
        print(("PASS " if ok else "---- ") + label + ("" if ok else f"  ({why})"))
    print(f"PS{n} score: {score} of {len(rows)}")
    return score
