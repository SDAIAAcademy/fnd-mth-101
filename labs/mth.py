"""Helpers shared by the labs of Linear Algebra for Machine Learning (FND-MTH-101).

    import mth as U

- U.QUICK is True when the environment variable MTH_QUICK is 1: smaller sizes, fewer repeats.
- U.pick(full, quick) returns the small value in quick mode.
- U.reference(name) reads labs/data/reference/<name>.json, the safety net of an exercise.
- U.check(condition, message) prints a tick or raises AssertionError.
- U.glove(words) returns public GloVe vectors (6B tokens, 50 dimensions, Public Domain Dedication
  and License), downloaded once into the cache folder (MTH_CACHE, default labs/cache/).
- U.embed_sentence(text, vectors) is the mean of the word vectors of a sentence.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REF = DATA / "reference"
QUICK = os.environ.get("MTH_QUICK", "0") == "1"
CACHE = Path(os.environ.get("MTH_CACHE", LABS / "cache"))

np.set_printoptions(precision=4, suppress=True, linewidth=110)

GLOVE_URL = ("https://github.com/piskvorky/gensim-data/releases/download/"
             "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")
GLOVE_MD5 = "c289bc5d7f2f02c6dc9f2f9b67641813"


def pick(full, quick):
    """The full value, or the small one in quick mode."""
    return quick if QUICK else full


def reference(name: str):
    """The reference result of an exercise, for the safety net."""
    return json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError("not yet: " + message)
    print("[ok]", message)


def _glove_file() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "glove-wiki-gigaword-50.gz"
    if not path.exists():
        print("downloading GloVe 50d (69 MB, once) ...")
        tmp = path.with_suffix(".part")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(GLOVE_URL, tmp)
                break
            except OSError as e:  # a busy network: try again
                print("download failed, retrying:", type(e).__name__)
        else:
            raise RuntimeError("GloVe could not be downloaded; check the network and run the cell again")
        digest = hashlib.md5(tmp.read_bytes()).hexdigest()
        if digest != GLOVE_MD5:
            tmp.unlink()
            raise RuntimeError("the downloaded file does not match its published checksum")
        tmp.replace(path)
    return path


def glove(words=None, first: int | None = None) -> dict[str, np.ndarray]:
    """Vectors for the given words (and the `first` most frequent words, when asked)."""
    wanted = set(words or [])
    out: dict[str, np.ndarray] = {}
    with gzip.open(_glove_file(), "rt", encoding="utf-8") as f:
        next(f)  # header line: count and dimension
        for i, line in enumerate(f):
            word, rest = line.split(" ", 1)
            if word in wanted or (first is not None and i < first):
                out[word] = np.array(rest.split(), dtype=np.float64)
            if first is not None and i >= first and wanted <= out.keys():
                break
    return out


def tokens(text: str) -> list[str]:
    return [w.strip(".,?!") for w in text.lower().split() if w.strip(".,?!")]


def embed_sentence(text: str, vectors: dict[str, np.ndarray]) -> np.ndarray:
    """The mean of the GloVe vectors of the words of a sentence (words not in the table are skipped)."""
    vs = [vectors[w] for w in tokens(text) if w in vectors]
    return np.mean(vs, axis=0)


def load_csv(name: str):
    """A CSV file of labs/data as a list of dictionaries (strings)."""
    import csv

    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short(x, nd: int = 4):
    """Round a number or an array for printing and for reference files."""
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def _round(obj, nd: int = 6):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, (list, tuple)):
        return [_round(x, nd) for x in obj]
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _round(obj.tolist(), nd)
    if isinstance(obj, np.generic):
        return _round(obj.item(), nd)
    return obj


def keep(name: str, value):
    """Return value; when MTH_WRITE_REF=1 (the trainer's tool), also save it as a reference file."""
    if os.environ.get("MTH_WRITE_REF") == "1":
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"{name}.json").write_text(json.dumps(_round(value), indent=1) + "\n", encoding="utf-8", newline="\n")
    return value
