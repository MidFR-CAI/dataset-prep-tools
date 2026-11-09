"""Steps 1-3: load, flatten transparency onto white, resize to target megapixels."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def load_flat_rgb(path: Path) -> Image.Image:
    """Open any supported image and return an RGB image with transparency
    composited onto pure white."""
    im = Image.open(path)
    im.seek(0)  # first frame of animated webp
    im = ImageOps.exif_transpose(im)

    if im.mode.startswith("I") or im.mode == "F":  # 16/32-bit grayscale
        arr = np.asarray(im).astype(np.float32)
        if arr.max() > 255:
            arr = arr / 256.0
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")

    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    return im


def target_size(w: int, h: int, megapixels: float, multiple: int) -> tuple[int, int]:
    scale = math.sqrt(megapixels * 1_000_000 / (w * h))
    m = max(1, multiple)
    nw = max(m, int(round(w * scale / m)) * m)
    nh = max(m, int(round(h * scale / m)) * m)
    return nw, nh


def resize_to_megapixels(im: Image.Image, megapixels: float, multiple: int) -> Image.Image:
    """Scales up or down so the result is ~`megapixels` MP."""
    nw, nh = target_size(*im.size, megapixels, multiple)
    if (nw, nh) == im.size:
        return im
    return im.resize((nw, nh), Image.LANCZOS)
