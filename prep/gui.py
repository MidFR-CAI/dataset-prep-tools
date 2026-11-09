"""PySide6 front-end."""
from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from .config import Config, load_settings, save_settings
from .pipeline import Callbacks, Cancelled, run

STYLE = """
* { font-family: 'Segoe UI'; font-size: 13px; color: #e6e8ee; }
QMainWindow, QDialog, QScrollArea, QScrollArea > QWidget > QWidget { background: #14161b; }
QGroupBox { background: #1c1f26; border: 1px solid #2a2e38; border-radius: 10px;
            margin-top: 14px; padding: 14px 12px 10px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #8ab4ff; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background: #12141a; border: 1px solid #2f3440; border-radius: 6px; padding: 5px 8px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #5b8def; }
QPushButton { background: #2a2f3a; border: none; border-radius: 6px; padding: 7px 14px; }
QPushButton:hover { background: #343a48; }
QPushButton:disabled { color: #6b7080; background: #20242d; }
QPushButton#primary { background: #4f7cf0; font-weight: 600; padding: 10px 18px; }
QPushButton#primary:hover { background: #6490ff; }
QPushButton#primary:disabled { background: #2c3550; color: #7f8aa8; }
QProgressBar { background: #12141a; border: 1px solid #2f3440; border-radius: 6px;
               text-align: center; height: 20px; }
QProgressBar::chunk { background: #4f7cf0; border-radius: 5px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px;
                       border: 1px solid #3a4050; background: #12141a; }
QCheckBox::indicator:checked { background: #4f7cf0; border-color: #4f7cf0; }
QSlider::groove:horizontal { height: 4px; background: #2f3440; border-radius: 2px; }
QSlider::handle:horizontal { background: #8ab4ff; width: 14px; margin: -6px 0; border-radius: 7px; }
QLabel#hint { color: #8a90a2; font-size: 12px; }
QLabel#title { font-size: 20px; font-weight: 700; }
"""


def pil_to_pixmap(im: Image.Image, box: int = 520) -> QPixmap:
    im = im.convert("RGB")
    im.thumbnail((box, box))
    q = QImage(im.tobytes(), im.width, im.height, 3 * im.width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(q)


# --------------------------------------------------------------------------- worker
class Worker(QObject):
    log = Signal(str)
    stage = Signal(str, int, int)
    preview = Signal(object)
    review_needed = Signal(object)
    finished = Signal(str)  # error text or ""

    def __init__(self, cfg: Config, token: str):
        super().__init__()
        self.cfg, self.token = cfg, token
        self._cancel = False
        self._event = threading.Event()
        self._result: list = []

    def cancel(self):
        self._cancel = True
        self._event.set()

    def deliver_review(self, to_delete: list):
        self._result = to_delete
        self._event.set()

    def _review(self, groups):
        self._event.clear()
        self.review_needed.emit(groups)
        self._event.wait()
        return self._result

    def go(self):
        cb = Callbacks(
            log=self.log.emit,
            stage=self.stage.emit,
            preview=lambda im: self.preview.emit(im.copy()),
            review=self._review,
            cancelled=lambda: self._cancel,
            hf_token=self.token or None,
        )
        try:
            run(self.cfg, cb)
            self.finished.emit("")
        except Cancelled:
            self.finished.emit("Cancelled.")
        except Exception as e:
            self.log.emit(traceback.format_exc())
            self.finished.emit(f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- review dialog
class ReviewDialog(QDialog):
    def __init__(self, groups, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review duplicates")
        self.resize(1000, 700)
        self.groups = groups
        self.checks: list[tuple[QCheckBox, object]] = []

        lay = QVBoxLayout(self)
        hint = QLabel(
            "Each row is a set of near-identical images. Ticked images are KEPT; "
            "unticked are deleted. The highest-resolution copy is pre-ticked. "
            "Double-click a thumbnail to open it full size."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        lay.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        v = QVBoxLayout(inner)
        for gi, g in enumerate(groups, 1):
            box = QGroupBox(f"Group {gi}  ({len(g)} images)")
            row = QHBoxLayout(box)
            for ri, rec in enumerate(g):
                col = QVBoxLayout()
                lab = QLabel()
                lab.setAlignment(Qt.AlignCenter)
                rd = QImageReader(str(rec.path))
                rd.setScaledSize(QSize(190, int(190 * rd.size().height() / max(1, rd.size().width()))))
                lab.setPixmap(QPixmap.fromImage(rd.read()))
                lab.mousePressEvent = lambda e, p=rec.path: None
                lab.mouseDoubleClickEvent = lambda e, p=rec.path: os.startfile(p)
                col.addWidget(lab)
                chk = QCheckBox(f"Keep   ({rec.orig_px / 1e6:.2f} MP orig)")
                chk.setChecked(ri == 0)
                col.addWidget(chk)
                nm = QLabel(rec.src.name)
                nm.setObjectName("hint")
                nm.setMaximumWidth(200)
                col.addWidget(nm)
                row.addLayout(col)
                self.checks.append((chk, rec))
            row.addStretch()
            v.addWidget(box)
        v.addStretch()
        scroll.setWidget(inner)
        lay.addWidget(scroll)

        bb = QDialogButtonBox()
        bb.addButton("Apply", QDialogButtonBox.AcceptRole)
        bb.addButton("Keep everything", QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def to_delete(self):
        if self.result() != QDialog.Accepted:
            return []
        return [rec for chk, rec in self.checks if not chk.isChecked()]


# --------------------------------------------------------------------------- main window
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Prep Tool")
        self.resize(1180, 820)
        self.cfg = load_settings()
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self._build()
        self._load_into_ui()

    # ---- ui construction
    def _row(self, label: str, widget: QWidget, hint: str = "") -> QHBoxLayout:
        h = QHBoxLayout()
        l = QLabel(label)
        l.setMinimumWidth(120)
        h.addWidget(l)
        h.addWidget(widget, 1)
        if hint:
            widget.setToolTip(hint)
        return h

    def _browse_row(self, edit: QLineEdit, cb):
        h = QHBoxLayout()
        h.addWidget(edit, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(cb)
        h.addWidget(b)
        return h

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(16)

        # ----- left: settings
        left = QVBoxLayout()
        title = QLabel("Dataset Prep Tool")
        title.setObjectName("title")
        left.addWidget(title)
        sub = QLabel("Prepare images for training: normalise → dedupe → mask watermarks → clean")
        sub.setObjectName("hint")
        left.addWidget(sub)

        g = QGroupBox("1 · Source & naming")
        v = QVBoxLayout(g)
        self.in_edit, self.out_edit = QLineEdit(), QLineEdit()
        self.in_edit.setPlaceholderText("Folder containing webp / png / jpg / jpeg")
        self.out_edit.setPlaceholderText("Default: <input folder>_prepared next to it")
        self.in_edit.textChanged.connect(self._input_changed)
        v.addWidget(QLabel("Input folder"))
        v.addLayout(self._browse_row(self.in_edit, self._pick_in))
        v.addWidget(QLabel("Output folder"))
        v.addLayout(self._browse_row(self.out_edit, self._pick_out))
        self.name_edit = QLineEdit()
        v.addLayout(self._row("File name", self.name_edit))
        h = QLabel("Files become <name>_0001.png … Leave blank to use the input folder's name.")
        h.setObjectName("hint")
        h.setWordWrap(True)
        v.addWidget(h)
        self.recursive = QCheckBox("Include subfolders")
        v.addWidget(self.recursive)
        left.addWidget(g)

        g = QGroupBox("2 · Resize")
        v = QVBoxLayout(g)
        self.mp = QDoubleSpinBox()
        self.mp.setRange(0.1, 16)
        self.mp.setSingleStep(0.1)
        self.mp.setSuffix(" MP")
        self.mult = QSpinBox()
        self.mult.setRange(1, 128)
        self.mult.setSuffix(" px")
        v.addLayout(self._row("Target size", self.mp))
        v.addLayout(self._row("Dimensions multiple of", self.mult))
        h = QLabel("Images are scaled up or down to hit the target. Transparency becomes white.")
        h.setObjectName("hint")
        h.setWordWrap(True)
        v.addWidget(h)
        left.addWidget(g)

        g = QGroupBox("3 · Duplicates")
        v = QVBoxLayout(g)
        self.dedupe = QCheckBox("Remove identical / near-identical images")
        self.review = QCheckBox("Let me review before anything is removed")
        v.addWidget(self.dedupe)
        v.addWidget(self.review)
        sl = QHBoxLayout()
        sl.addWidget(QLabel("Strict"))
        self.bits = QSlider(Qt.Horizontal)
        self.bits.setRange(0, 40)
        sl.addWidget(self.bits, 1)
        sl.addWidget(QLabel("Loose"))
        self.bits_lbl = QLabel()
        self.bits_lbl.setMinimumWidth(24)
        sl.addWidget(self.bits_lbl)
        self.bits.valueChanged.connect(lambda x: self.bits_lbl.setText(str(x)))
        v.addLayout(sl)
        left.addWidget(g)

        left.addStretch()
        root.addLayout(left, 4)

        # ----- middle: AI steps
        mid = QVBoxLayout()
        g = QGroupBox("4 · Watermark & text masking (SAM)")
        v = QVBoxLayout(g)
        self.mask_on = QCheckBox("Detect and mask")
        v.addWidget(self.mask_on)
        self.sam = QComboBox()
        self.sam.addItems(["sam3.1", "sam3"])
        v.addLayout(self._row("Model", self.sam))
        self.prompts = QLineEdit()
        v.addLayout(self._row("Prompts", self.prompts,
                              "Comma-separated text prompts, e.g. watermark, text, logo, signature"))
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        v.addLayout(self._row("Confidence", self.conf, "Lower = catches more, more false positives"))
        self.grow = QSpinBox()
        self.grow.setRange(0, 30)
        self.grow.setSuffix(" px")
        v.addLayout(self._row("Mask grow", self.grow,
                              "Dilation so anti-aliased edges are covered. 0 = raw SAM mask."))
        self.maxarea = QDoubleSpinBox()
        self.maxarea.setRange(1, 100)
        self.maxarea.setSuffix(" %")
        v.addLayout(self._row("Ignore if larger", self.maxarea,
                              "Detections covering more than this % of the image are discarded"))
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("Only needed once — SAM weights are gated")
        v.addLayout(self._row("Hugging Face token", self.token))
        h = QLabel("Request access at huggingface.co/facebook/sam3.1, then paste a read token. "
                   "It is saved by huggingface_hub inside this tool's models folder.")
        h.setObjectName("hint")
        h.setWordWrap(True)
        v.addWidget(h)
        mid.addWidget(g)

        g = QGroupBox("5 · Cleaning (IOPaint)")
        v = QVBoxLayout(g)
        self.inp_on = QCheckBox("Inpaint masked areas")
        v.addWidget(self.inp_on)
        self.inp_model = QComboBox()
        self.inp_model.addItems(["lama", "mat", "migan", "fcf", "manga", "ldm"])
        v.addLayout(self._row("Model", self.inp_model))
        self.keep = QCheckBox("Keep mask images (saved to _masks)")
        v.addWidget(self.keep)
        self.mask_on.toggled.connect(self._sync_enabled)
        mid.addWidget(g)
        mid.addStretch()
        root.addLayout(mid, 4)

        # ----- right: run + preview + log
        right = QVBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        hb = QHBoxLayout()
        hb.addWidget(self.start_btn, 1)
        hb.addWidget(self.cancel_btn)
        right.addLayout(hb)
        self.stage_lbl = QLabel("Idle")
        right.addWidget(self.stage_lbl)
        self.bar = QProgressBar()
        right.addWidget(self.bar)
        self.preview = QLabel("Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 260)
        self.preview.setFrameShape(QFrame.StyledPanel)
        self.preview.setStyleSheet("background:#12141a; border:1px solid #2a2e38; border-radius:8px; color:#6b7080;")
        right.addWidget(self.preview, 3)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        right.addWidget(self.logbox, 2)
        root.addLayout(right, 5)

    # ---- ui <-> config
    def _sync_enabled(self):
        on = self.mask_on.isChecked()
        self.inp_on.setEnabled(on)
        if not on:
            self.inp_on.setChecked(False)

    def _load_into_ui(self):
        c = self.cfg
        self.in_edit.setText(c.input_dir)
        self.out_edit.setText(c.output_dir)
        self.name_edit.setText(c.base_name)
        self.recursive.setChecked(c.recursive)
        self.mp.setValue(c.megapixels)
        self.mult.setValue(c.multiple_of)
        self.dedupe.setChecked(c.dedupe)
        self.review.setChecked(c.dedupe_review)
        self.bits.setValue(c.dedupe_bits)
        self.bits_lbl.setText(str(c.dedupe_bits))
        self.mask_on.setChecked(c.mask_enabled)
        self.sam.setCurrentText(c.sam_version)
        self.prompts.setText(c.prompts)
        self.conf.setValue(c.confidence)
        self.grow.setValue(c.grow_px)
        self.maxarea.setValue(c.max_area_pct)
        self.inp_on.setChecked(c.inpaint_enabled)
        self.inp_model.setCurrentText(c.inpaint_model)
        self.keep.setChecked(c.keep_intermediates)
        self._input_changed(c.input_dir)
        self._sync_enabled()

    def _read_ui(self) -> Config:
        c = self.cfg
        c.input_dir = self.in_edit.text().strip()
        c.output_dir = self.out_edit.text().strip()
        c.base_name = self.name_edit.text().strip()
        c.recursive = self.recursive.isChecked()
        c.megapixels = self.mp.value()
        c.multiple_of = self.mult.value()
        c.dedupe = self.dedupe.isChecked()
        c.dedupe_review = self.review.isChecked()
        c.dedupe_bits = self.bits.value()
        c.mask_enabled = self.mask_on.isChecked()
        c.sam_version = self.sam.currentText()
        c.prompts = self.prompts.text()
        c.confidence = self.conf.value()
        c.grow_px = self.grow.value()
        c.max_area_pct = self.maxarea.value()
        c.inpaint_enabled = self.inp_on.isChecked()
        c.inpaint_model = self.inp_model.currentText()
        c.keep_intermediates = self.keep.isChecked()
        return c

    def _input_changed(self, text: str):
        folder = Path(text.strip()).name if text.strip() else ""
        self.name_edit.setPlaceholderText(folder or "Defaults to the input folder's name")

    def _pick_in(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder", self.in_edit.text())
        if d:
            self.in_edit.setText(d)

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    # ---- running
    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def _start(self):
        cfg = self._read_ui()
        if not Path(cfg.input_dir).is_dir():
            self.logbox.appendPlainText("Pick a valid input folder first.")
            return
        save_settings(cfg)
        self.logbox.clear()
        self.bar.setValue(0)
        self._set_running(True)
        self.worker = Worker(cfg, self.token.text().strip())
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.go)
        self.worker.log.connect(self.logbox.appendPlainText)
        self.worker.stage.connect(self._on_stage)
        self.worker.preview.connect(self._on_preview)
        self.worker.review_needed.connect(self._on_review)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()

    def _cancel(self):
        if self.worker:
            self.logbox.appendPlainText("Cancelling after the current image…")
            self.worker.cancel()

    def _on_stage(self, name: str, i: int, n: int):
        self.stage_lbl.setText(f"{name}  ({i}/{n})")
        self.bar.setMaximum(max(1, n))
        self.bar.setValue(i)

    def _on_preview(self, im):
        self.preview.setPixmap(pil_to_pixmap(im, 520).scaled(
            self.preview.size() - QSize(8, 8), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_review(self, groups):
        dlg = ReviewDialog(groups, self)
        dlg.exec()
        self.worker.deliver_review(dlg.to_delete())

    def _on_finished(self, err: str):
        self.thread.quit()
        self.thread.wait()
        self._set_running(False)
        self.stage_lbl.setText(err or "Done")
        if err:
            self.logbox.appendPlainText(err)
        elif self.cfg.output_dir or self.cfg.input_dir:
            pass

    def closeEvent(self, e):
        if self.worker and self.thread and self.thread.isRunning():
            self.worker.cancel()
            self.thread.quit()
            self.thread.wait(5000)
        super().closeEvent(e)


def main():
    app = QApplication([])
    app.setStyleSheet(STYLE)
    w = Main()
    w.show()
    app.exec()
