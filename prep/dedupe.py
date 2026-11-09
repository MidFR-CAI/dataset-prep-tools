"""Perceptual-hash near-duplicate detection."""
from __future__ import annotations

import numpy as np
from PIL import Image

HASH_SIZE = 16  # 16x16 = 256 bit pHash
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def signature(im: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Returns (256-bit pHash as uint8 bits, 8x8 RGB thumbnail as float32)."""
    import imagehash

    bits = imagehash.phash(im, hash_size=HASH_SIZE).hash.flatten().astype(np.uint8)
    thumb = np.asarray(im.resize((8, 8), Image.BOX), dtype=np.float32).flatten()
    return bits, thumb


def find_groups(
    bits: list[np.ndarray],
    thumbs: list[np.ndarray],
    max_bits: int,
    max_color_diff: float = 24.0,
) -> list[list[int]]:
    """Cluster near-identical images. Two images match when their pHash differs by
    <= max_bits AND their coarse colours agree (so recolours aren't merged)."""
    n = len(bits)
    if n < 2:
        return []
    packed = np.packbits(np.stack(bits), axis=1)
    thumbs_a = np.stack(thumbs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n - 1):
        d = _POP[packed[i + 1:] ^ packed[i]].sum(axis=1, dtype=np.int32)
        for k in np.nonzero(d <= max_bits)[0]:
            j = i + 1 + int(k)
            if np.abs(thumbs_a[i] - thumbs_a[j]).mean() <= max_color_diff:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]
