"""The full dataset-prep pipeline (GUI-agnostic)."""
from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .config import IMAGE_EXTS, Config
from .dedupe import find_groups, signature
from .imaging import load_flat_rgb, resize_to_megapixels


class Cancelled(Exception):
    pass


@dataclass
class Callbacks:
    log: Callable[[str], None] = print
    stage: Callable[[str, int, int], None] = lambda name, i, n: None
    preview: Callable[[Image.Image], None] = lambda im: None
    # receives list of groups (each a list of Record), returns Records to DELETE
    review: Callable[[list], list] | None = None
    cancelled: Callable[[], bool] = lambda: False
    hf_token: str | None = None


@dataclass
class Record:
    src: Path
    path: Path                      # current working file (PNG)
    orig_px: int
    file_size: int
    bits: np.ndarray = field(repr=False, default=None)
    thumb: np.ndarray = field(repr=False, default=None)


def _safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(" .")
    return s or "image"


def collect_images(folder: Path, recursive: bool, exclude: Path | None = None) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    files = []
    for p in it:
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        if exclude and exclude in p.parents:
            continue
        files.append(p)
    return sorted(files, key=lambda p: [int(t) if t.isdigit() else t.lower()
                                        for t in re.split(r"(\d+)", str(p))])


def run(cfg: Config, cb: Callbacks) -> None:
    def check():
        if cb.cancelled():
            raise Cancelled()

    in_dir = Path(cfg.input_dir)
    if not in_dir.is_dir():
        raise ValueError("Input folder does not exist.")
    out_dir = Path(cfg.output_dir) if cfg.output_dir else in_dir.parent / f"{in_dir.name}_prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    base_dir, mask_dir = work / "base", work / "masks"
    for d in (base_dir, mask_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    name = _safe_name(cfg.base_name.strip() or in_dir.name)
    files = collect_images(in_dir, cfg.recursive, exclude=out_dir)
    if not files:
        raise ValueError("No webp/png/jpg/jpeg images found in that folder.")
    cb.log(f"Found {len(files)} images. Output: {out_dir}")

    # ---- Steps 1-3: convert, flatten alpha to white, resize ---------------------------
    def work_one(args):
        idx, src = args
        if cb.cancelled():
            return None
        try:
            im = load_flat_rgb(src)
            orig_px = im.width * im.height
            im = resize_to_megapixels(im, cfg.megapixels, cfg.multiple_of)
            dst = base_dir / f"{idx:06d}.png"
            im.save(dst, "PNG", compress_level=3)
            bits, thumb = signature(im) if cfg.dedupe else (None, None)
            return Record(src, dst, orig_px, src.stat().st_size, bits, thumb)
        except Exception as e:
            cb.log(f"  ! skipped {src.name}: {type(e).__name__}: {e}")
            return None

    records: list[Record] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, rec in enumerate(ex.map(work_one, enumerate(files)), 1):
            check()
            if rec:
                records.append(rec)
            cb.stage("Converting / resizing", i, len(files))

    # ---- Duplicate removal ------------------------------------------------------------
    if cfg.dedupe and len(records) > 1:
        cb.stage("Finding duplicates", 0, 1)
        groups_idx = find_groups(
            [r.bits for r in records], [r.thumb for r in records], cfg.dedupe_bits
        )
        groups = [
            sorted((records[i] for i in g), key=lambda r: (-r.orig_px, -r.file_size))
            for g in groups_idx
        ]
        cb.log(f"Duplicates: {len(groups)} group(s) found.")
        if groups:
            if cfg.dedupe_review and cb.review:
                to_delete = cb.review(groups)
                check()
            else:
                to_delete = [r for g in groups for r in g[1:]]  # keep best of each group
            dead = {id(r) for r in to_delete}
            for r in to_delete:
                r.path.unlink(missing_ok=True)
            records = [r for r in records if id(r) not in dead]
            cb.log(f"Removed {len(dead)} duplicate image(s); {len(records)} remain.")
        cb.stage("Finding duplicates", 1, 1)

    # ---- Rename to final names (gapless numbering) ------------------------------------
    pad = max(4, len(str(len(records))))
    for n, r in enumerate(records, 1):
        new = base_dir / f"{name}_{n:0{pad}d}.png"
        r.path.rename(new)
        r.path = new

    # ---- Step 4: SAM masking ----------------------------------------------------------
    has_mask: set[Path] = set()
    if cfg.mask_enabled and records:
        prompts = cfg.prompt_list()
        if not prompts:
            raise ValueError("No mask prompts given.")
        from .masking import Masker

        cb.stage("Loading SAM", 0, 1)
        masker = Masker(cfg.sam_version, cfg.confidence, cb.hf_token, cb.log)
        try:
            for i, r in enumerate(records, 1):
                check()
                im = Image.open(r.path).convert("RGB")
                mask = masker.detect(im, prompts, cfg.grow_px, cfg.max_area_pct)
                if mask.any():
                    Image.fromarray(mask, "L").save(mask_dir / r.path.name)
                    has_mask.add(r.path)
                    ov = np.asarray(im).copy()
                    ov[mask > 0] = (0.4 * ov[mask > 0] + 0.6 * np.array([255, 0, 60])).astype(np.uint8)
                    cb.preview(Image.fromarray(ov))
                cb.stage("Detecting watermarks/text", i, len(records))
        finally:
            masker.close()
        cb.log(f"Masks found on {len(has_mask)} of {len(records)} images.")

    # ---- Step 5: IOPaint --------------------------------------------------------------
    if cfg.inpaint_enabled and has_mask:
        from .inpaint import Cleaner

        cb.stage("Loading IOPaint", 0, 1)
        cleaner = Cleaner(cfg.inpaint_model, cb.log)
        try:
            todo = [r for r in records if r.path in has_mask]
            for i, r in enumerate(todo, 1):
                check()
                im = Image.open(r.path).convert("RGB")
                mask = np.asarray(Image.open(mask_dir / r.path.name).convert("L"))
                res = cleaner.clean(im, mask)
                res.save(out_dir / r.path.name, "PNG", compress_level=6)
                cb.preview(res)
                cb.stage("Cleaning with IOPaint", i, len(todo))
        finally:
            cleaner.close()

    # ---- Finish: copy untouched images, tidy up ---------------------------------------
    for r in records:
        dst = out_dir / r.path.name
        if not dst.exists():
            shutil.copy2(r.path, dst)
    if cfg.keep_intermediates:
        keep = out_dir / "_masks"
        if keep.exists():
            shutil.rmtree(keep)
        if mask_dir.exists():
            shutil.copytree(mask_dir, keep)
    shutil.rmtree(work, ignore_errors=True)
    cb.stage("Done", 1, 1)
    cb.log(f"Finished. {len(records)} images written to {out_dir}")
