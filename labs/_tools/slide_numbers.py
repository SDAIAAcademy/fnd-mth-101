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


WEEKS = {1: week1}

if __name__ == "__main__":
    for w in ([int(a) for a in sys.argv[1:]] or sorted(WEEKS)):
        print(f"== week {w}")
        WEEKS[w]()
