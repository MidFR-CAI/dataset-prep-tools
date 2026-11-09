"""Settings shared between the GUI and the pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"

IMAGE_EXTS = {".webp", ".png", ".jpg", ".jpeg"}


@dataclass
class Config:
    input_dir: str = ""
    output_dir: str = ""
    base_name: str = ""          # empty -> use input folder name
    recursive: bool = False

    # steps 1-3
    megapixels: float = 1.5
    multiple_of: int = 8

    # dedupe
    dedupe: bool = True
    dedupe_bits: int = 8         # max differing bits out of 256 (perceptual hash)
    dedupe_review: bool = False

    # masking (SAM)
    mask_enabled: bool = True
    sam_version: str = "sam3.1"
    prompts: str = "watermark, text"
    confidence: float = 0.4
    grow_px: int = 3
    max_area_pct: float = 25.0   # drop single detections bigger than this % of the image

    # inpainting (IOPaint)
    inpaint_enabled: bool = True
    inpaint_model: str = "lama"

    keep_intermediates: bool = False

    def prompt_list(self) -> list[str]:
        return [p.strip() for p in self.prompts.split(",") if p.strip()]


def load_settings() -> Config:
    cfg = Config()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        names = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in names:
                setattr(cfg, k, v)
    except Exception:
        pass
    return cfg


def save_settings(cfg: Config) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    except Exception:
        pass
