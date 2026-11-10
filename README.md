# Dataset Prep Tool

Self-contained GUI (own `.venv`, models stored in `./models`) that prepares images for image-gen training.

**Run:** double-click `run.bat` (it calls `setup.bat` first if `.venv` is missing).

Pipeline: convert to PNG + rename -> resize to target MP (up or down) -> flatten transparency to white ->
remove near-duplicate images (optional review) -> SAM 3.1 masks watermarks/text -> IOPaint (LaMa) cleans them.

## Manual cleaning (IOPaint + LaMa)
Double-click `manual_clean.bat` to open IOPaint's editor in your browser (http://localhost:8080): brush over
anything you want removed and LaMa fills it in. Drag a folder of images (e.g. the tool's output) onto the
`.bat` to browse that folder in the editor; each cleaned image is saved over the original file in that same folder
(so the dataset stays in place, same filename, no suffix). Keep *Auto Saving* on in the editor's settings and
don't use the Download button, which is the only thing that adds `_cleanup`. Close the console window to stop it.

## One-time SAM setup
SAM 3 / 3.1 weights are gated. Request access at https://huggingface.co/facebook/sam3.1 (and `facebook/sam3`),
create a read token, and paste it in the GUI's *Hugging Face token* box on first run.

## Notes
- Python 3.12 + CUDA 12.8 PyTorch + `triton-windows` (needed by sam3 on Windows).
- Output: `<name>_0001.png ...` in the output folder; `_masks/` kept if "Keep mask images" is ticked.
- Tight masks: SAM instance masks + small `Mask grow` (default 3 px) so anti-aliased edges are covered.
- Detections larger than `Ignore if larger` (% of image) are discarded as false positives.
