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


CHECKS = {1: _ps1}


def check(n: int, ns: dict) -> int:
    rows = list(CHECKS[n](ns))
    score = 0
    for label, (ok, why) in rows:
        score += ok
        print(("PASS " if ok else "---- ") + label + ("" if ok else f"  ({why})"))
    print(f"PS{n} score: {score} of {len(rows)}")
    return score
