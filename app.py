import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keep every downloaded model inside this folder so the tool stays self-contained.
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(ROOT))

# pythonw.exe has no console: stdout/stderr are None, which crashes tqdm progress bars
# (e.g. Hugging Face downloads). Send them to a log file instead.
if sys.stdout is None or sys.stderr is None:
    _log = open(ROOT / "app.log", "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

from prep.gui import main  # noqa: E402

if __name__ == "__main__":
    main()
