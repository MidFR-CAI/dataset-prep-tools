"""Step 5: remove masked regions with IOPaint (in-process, no server)."""
from __future__ import annotations

import gc

import cv2
import numpy as np
import torch
from PIL import Image


class Cleaner:
    def __init__(self, model_name: str, log=print):
        from iopaint.model_manager import ModelManager
        from iopaint.schema import HDStrategy, InpaintRequest

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Loading IOPaint model '{model_name}' on {device} (first run downloads it) ...")
        from iopaint.download import cli_download_model, scan_models

        if model_name not in [m.name for m in scan_models()]:
            cli_download_model(model_name)  # fetches weights into TORCH_HOME
        self.mm = ModelManager(name=model_name, device=device)
        self.cfg = InpaintRequest(
            hd_strategy=HDStrategy.CROP,
            hd_strategy_crop_margin=196,
            hd_strategy_crop_trigger_size=800,
            hd_strategy_resize_limit=2048,
        )

    def clean(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        rgb = np.array(image.convert("RGB"))  # writable copy (IOPaint edits in place)
        bin_mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        out_bgr = self.mm(rgb, bin_mask, self.cfg)
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out)

    def close(self) -> None:
        del self.mm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
