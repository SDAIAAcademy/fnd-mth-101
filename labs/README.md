# Linear Algebra for Machine Learning: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository and installs `requirements-colab.txt`.
- Locally: clone the repository and, from `labs/`, open the notebooks with `uv run --with jupyterlab jupyter lab` (the dependencies are in `pyproject.toml`). The data in `data/` is already there; `uv run python prepare_data.py` writes it again, identically.

| Unit | Session | Lab | Notebook |
|---|---|---|---|
| Week 1 | Vectors, dot products and similarity | Cosine search over public sentence embeddings | [`W1_1.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-101/blob/main/labs/W1_1.ipynb) |
| Week 2 | Matrix products, attention scores and FLOPs | Loop and vectorised matrix products, costed in FLOPs | [`W2_1.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-101/blob/main/labs/W2_1.ipynb) |
| Week 3 | Span, rank and linear systems | Rank and linear systems on a feature table | [`W3_1.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-101/blob/main/labs/W3_1.ipynb) |
| Week 4 | Orthogonal projection and QR | Projections and QR by Gram-Schmidt | [`W4_1.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-101/blob/main/labs/W4_1.ipynb) |
| Week 5 | Least squares and conditioning | Least squares by normal equations and by QR | [`W5_1.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-101/blob/main/labs/W5_1.ipynb) |
| Week 6 | Eigenvectors and link ranking | Power iteration and link ranking on a synthetic web graph | [`W6_1.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-101/blob/main/labs/W6_1.ipynb) |
| Week 7 | The SVD and principal components | SVD image compression and PCA on public handwritten digits | [`W7_1.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-101/blob/main/labs/W7_1.ipynb) |
| Week 8 | Low rank in practice | Low-rank ratings and adapter updates, checked by hand | [`W8_1.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-101/blob/main/labs/W8_1.ipynb) |

Every lab starts with a small case done by hand, then does the same at scale in NumPy and checks one against the other. An exercise left undone raises `NotImplementedError`; the notebook then falls back on a reference value in `data/reference/` so that every later step still runs. The worked versions are in [`solutions/`](solutions/).

## Problem sets

[`problem-sets/PS1.ipynb`](problem-sets/) to `PS8.ipynb`, one a week, 60 minutes each, AI off for the first attempt. The last cell of each calls [`pscheck.py`](problem-sets/pscheck.py), which prints one line per item and a score. It stores no answers: it checks that each hand answer agrees with your own code on the same small case, and runs your functions on random matrices against properties or the library. The worked answers are kept by the trainer.

## Data

- `prepare_data.py` wrote the platform's synthetic data in `data/` with seed 101: help articles and judged search questions, the words for the bias measurement, learner features, study hours, help-centre links and ratings.
- Lab 1.1 downloads the public GloVe word vectors (6B tokens, 50 dimensions; Public Domain Dedication and License) once, about 69 MB, from the gensim-data release, checks the file's MD5 and keeps it in `cache/`. Set `MTH_CACHE` to keep it elsewhere.
- Lab 7.1 uses the handwritten digits bundled with scikit-learn (CC BY 4.0) and a CC0 test image bundled with scikit-image; nothing is downloaded.

## Settings

| Variable | Effect |
|---|---|
| `MTH_QUICK=1` | Smaller sizes and fewer repeats, for a fast run-through |
| `MTH_CACHE=<folder>` | Where the GloVe vectors are kept (default `labs/cache/`) |

Timings in labs 2.1 and 3.1 depend on the machine: compare your own numbers, not the solution's.

## Tutor

[`tutor/TUTOR.md`](tutor/TUTOR.md) is the instruction for the hint-only tutor allowed in labs: hints, never answers.

## For maintainers

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks. `labs/_tools/slide_numbers.py` recomputes the numbers quoted on the slides.
