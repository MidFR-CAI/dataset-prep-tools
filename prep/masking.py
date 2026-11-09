"""Step 4: watermark / text detection with SAM 3.x text prompts."""
from __future__ import annotations

import gc

import cv2
import numpy as np
import torch
from PIL import Image

GATED_HELP = (
    "Could not download the SAM checkpoint. The SAM 3 / 3.1 weights are gated on "
    "Hugging Face: open https://huggingface.co/facebook/sam3.1 (and /facebook/sam3), "
    "request access, then paste a read token into the 'Hugging Face token' box."
)


class Masker:
    def __init__(self, version: str, confidence: float, hf_token: str | None, log=print):
        if hf_token:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Loading {version} on {self.device} ...")
        try:
            ckpt = download_ckpt_from_hf(version=version)
        except Exception as e:  # gated repo / no token / offline
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

            if isinstance(e, (GatedRepoError, HfHubHTTPError)):
                raise RuntimeError(f"{GATED_HELP}\n\n({type(e).__name__}: {e})") from e
            raise

        model = build_sam3_image_model(
            checkpoint_path=ckpt, load_from_HF=False, device=self.device
        )
        self.proc = Sam3Processor(
            model, device=self.device, confidence_threshold=confidence
        )
        self._model = model

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        prompts: list[str],
        grow_px: int,
        max_area_pct: float,
    ) -> np.ndarray:
        """Returns a uint8 (H, W) mask, 255 where watermark/text was found."""
        w, h = image.size
        total = float(w * h)
        union = np.zeros((h, w), dtype=bool)

        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with ctx:
            state = self.proc.set_image(image)
            for prompt in prompts:
                self.proc.reset_all_prompts(state)
                out = self.proc.set_text_prompt(state=state, prompt=prompt)
                masks = out.get("masks")
                if masks is None or len(masks) == 0:
                    continue
                m = masks.detach().cpu().numpy()
                if m.ndim == 4:
                    m = m[:, 0]
                for inst in m:
                    inst = inst.astype(bool)
                    if inst.shape != union.shape:
                        inst = cv2.resize(
                            inst.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    # a "text" prompt that grabs half the picture is a false positive
                    if inst.sum() / total * 100.0 > max_area_pct:
                        continue
                    union |= inst

        mask = union.astype(np.uint8) * 255
        if grow_px > 0 and mask.any():
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)
            )
            mask = cv2.dilate(mask, k)
        return mask

    def close(self) -> None:
        del self.proc, self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
