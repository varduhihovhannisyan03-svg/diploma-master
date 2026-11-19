"""Duplicate detection: exact (SHA-256 of bytes) and near (perceptual hash, Hamming distance)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def phash_int(path: str | Path, hash_size: int = 8) -> int:
    """64-bit perceptual hash as an integer (hash_size must be 8 so it fits in uint64)."""
    import imagehash
    from PIL import Image

    if hash_size * hash_size > 64:
        raise ValueError("hash_size > 8 does not fit in uint64")
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
    bits = "".join("1" if b else "0" for b in h.hash.flatten())
    return int(bits, 2)


def hamming_to(hashes: np.ndarray, value: int) -> np.ndarray:
    """Hamming distance between every uint64 in `hashes` and one uint64 `value`."""
    x = np.bitwise_xor(hashes.astype(np.uint64), np.uint64(value))
    return _POPCOUNT8[x.view(np.uint8).reshape(-1, 8)].sum(axis=1, dtype=np.int64)


def find_near_pairs(hashes: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    """All index pairs (i<j) with Hamming distance <= threshold. O(n^2) vectorized per row."""
    pairs: list[tuple[int, int, int]] = []
    for i in range(len(hashes) - 1):
        d = hamming_to(hashes[i + 1:], int(hashes[i]))
        for k in np.nonzero(d <= threshold)[0]:
            pairs.append((i, i + 1 + int(k), int(d[k])))
    return pairs


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_duplicates(sha256s: list[str], phashes: np.ndarray, threshold: int) -> np.ndarray:
    """Return a group id per record. Records linked by exact or near matches share an id (transitively).

    Singletons get their own id too, so the output is one id per row.
    """
    n = len(sha256s)
    uf = _UnionFind(n)
    first_by_sha: dict[str, int] = {}
    for i, digest in enumerate(sha256s):
        if digest in first_by_sha:
            uf.union(first_by_sha[digest], i)
        else:
            first_by_sha[digest] = i
    for i, j, _ in find_near_pairs(phashes, threshold):
        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def choose_keepers(df: pd.DataFrame) -> pd.DataFrame:
    """One keeper per group. Rule: if the group has any OFFICIAL TEST member, keep the test member with the
    smallest image_id and drop everything else (so no train copy of a test image survives).
    Otherwise keep the smallest image_id of the train-only group. Expects columns:
    group, image_id, official_split. Returns df with a boolean column `is_keeper`.
    """
    out = df.copy()
    out["is_keeper"] = False
    for _, sub in out.groupby("group", sort=False):
        if len(sub) == 1:
            out.loc[sub.index, "is_keeper"] = True
            continue
        test_members = sub[sub["official_split"] == "test"]
        pool = test_members if len(test_members) else sub
        keeper = pool.sort_values("image_id").index[0]
        out.loc[keeper, "is_keeper"] = True
    return out
