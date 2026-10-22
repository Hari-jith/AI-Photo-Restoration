"""
colorize_backup.py

A second, independent colorizer (Zhang, Isola & Efros, SIGGRAPH 2017) plus the
fallback logic that uses it when DDColor is unavailable.

    from src.colorize_backup import ColorizerChain

    chain = ColorizerChain()                      # DDColor first, backup second
    result = chain.colorize(gray_image)
    result.image          # colourized Pillow image
    result.method         # "ddcolor" or "siggraph17"
    result.notes          # plain-English remarks, e.g. why the backup was used

Why have a backup?
  * It is much smaller (about 130 MB) and faster than DDColor.
  * If DDColor fails to load or runs out of memory, the website still works.
  * It ALWAYS runs on the CPU: when DDColor fails because the graphics card
    ran out of memory, the backup must not compete for that memory.
Its colours are usually MORE MUTED and less accurate than DDColor's, and it
looks at the photo at only 256 x 256 pixels. It is a safety net, not an equal.

History: this module first used the authors' 2016 Caffe model, whose download
address now answers "404 Not Found". It now uses the authors' current PyTorch
SIGGRAPH 2017 model, hosted by them on S3.

Set-up (once):   python scripts\\setup_backup_colorizer.py
Source: https://github.com/richzhang/colorization   (BSD 2-Clause)

Self-test (no weights needed):   python -m src.colorize_backup
Backup colorizer on a file:      python -m src.colorize_backup outputs\\preview_gray.png
Full chain on a file:            python -m src.colorize_backup outputs\\preview_gray.png --chain
"""

import argparse
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src import config
from src.colorize import (MAX_SATURATION, MIN_SATURATION, ColorizationError,
                          Colorizer, load_weights_into)
from src.logging_setup import get_logger

logger = get_logger(__name__)

# --- Settings ----------------------------------------------------------------
BACKUP_DIR = config.MODELS_DIR / "colorizer_backup"
WEIGHTS_PATH = BACKUP_DIR / "siggraph17-df00044c.pth"
INPUT_SIZE = 256            # the network looks at a 256 x 256 copy of the photo

METHOD_DDCOLOR = "ddcolor"
METHOD_BACKUP = "siggraph17"


class BackupWeightsMissingError(ColorizationError):
    """The backup colorizer's weights file is missing."""


# ---------------------------------------------------------------------------
# Loading the network
# ---------------------------------------------------------------------------
def _load_model():
    """Build the SIGGRAPH 2017 network and load the downloaded weights (CPU)."""
    if not WEIGHTS_PATH.is_file():
        raise BackupWeightsMissingError(
            f"The backup colorizer weights were not found at {WEIGHTS_PATH}. "
            "Download them with:  python scripts\\setup_backup_colorizer.py")
    try:
        from src.vendor.siggraph17.siggraph17 import SIGGRAPHGenerator
    except ImportError as error:
        raise ColorizationError(
            "The backup colorizer code is missing. Run this once from the project "
            f"folder:  python scripts\\setup_backup_colorizer.py  (details: {error})"
        ) from error
    model = SIGGRAPHGenerator().eval()
    # Verifies that the file really fits this network (see colorize.py).
    load_weights_into(model, WEIGHTS_PATH)
    return model


# ---------------------------------------------------------------------------
# The backup colorizer
# ---------------------------------------------------------------------------
class BackupColorizer:
    """Colorizes with the SIGGRAPH 2017 network. Loads lazily on first use."""

    def __init__(self, model=None):
        """model: an already-built network (used by the self-test)."""
        self._model = model
        self._lock = threading.Lock()   # one image at a time

    def weights_available(self) -> bool:
        return WEIGHTS_PATH.is_file()

    def colorize(self, image: Image.Image, saturation: float = 1.0) -> Image.Image:
        """Return a colourized copy (same size, RGB). Same interface as
        Colorizer.colorize, so the two are interchangeable."""
        if not MIN_SATURATION <= saturation <= MAX_SATURATION:
            raise ValueError(
                f"saturation must be between {MIN_SATURATION} and {MAX_SATURATION}.")
        with self._lock:
            if self._model is None:
                started = time.time()
                self._model = _load_model()
                logger.info("Backup colorizer loaded in %.1f s", time.time() - started)
            started = time.time()
            try:
                result = self._run(image, saturation)
            except MemoryError as error:
                raise ColorizationError(
                    "Not enough memory to colorize this image.") from error
            except Exception as error:
                raise ColorizationError(
                    f"Backup colorization failed: {error}") from error
            logger.info("Backup colorized %dx%d image in %.1f s",
                        image.width, image.height, time.time() - started)
            return result

    def _run(self, image: Image.Image, saturation: float) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]

        # Lightness of the full-size photo goes back into the result unchanged.
        original_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[:, :, 0]

        # Network input: lightness (0..100) of a 256 x 256 copy. The network
        # normalises it itself. Bicubic resizing can overshoot, so clip first.
        small = np.clip(cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                                   interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
        small_l = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)[:, :, 0]
        tensor = torch.from_numpy(small_l).float().reshape(1, 1, INPUT_SIZE, INPUT_SIZE)

        with torch.no_grad():
            predicted = self._model(tensor).float()            # (1, 2, 256, 256), ab units

        ab = F.interpolate(predicted, size=(height, width), mode="bilinear",
                           align_corners=False)[0].numpy().transpose(1, 2, 0) * saturation
        lab = np.concatenate((original_l[:, :, np.newaxis], ab), axis=2).astype(np.float32)
        result_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # clip BEFORE converting to bytes, or out-of-range values wrap around
        result_bytes = (np.clip(result_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return Image.fromarray(result_bytes, "RGB")


# ---------------------------------------------------------------------------
# Fallback logic
# ---------------------------------------------------------------------------
@dataclass
class ChainResult:
    image: Image.Image
    method: str                       # METHOD_DDCOLOR or METHOD_BACKUP
    notes: list = field(default_factory=list)


class ColorizerChain:
    """Tries DDColor first; if that fails, uses the backup colorizer.

    Only ColorizationError triggers the fallback. Other errors (for example a
    ValueError from a bad argument) are programming mistakes and are raised
    as they are, instead of being hidden by the backup.
    """

    def __init__(self, primary=None, backup=None):
        self.primary = primary if primary is not None else Colorizer()
        self.backup = backup if backup is not None else BackupColorizer()

    def colorize(self, image: Image.Image, saturation: float = 1.0) -> ChainResult:
        try:
            result = self.primary.colorize(image, saturation=saturation)
            return ChainResult(result, METHOD_DDCOLOR)
        except ColorizationError as primary_error:
            logger.warning("DDColor failed (%s); trying the backup colorizer.",
                           primary_error)
            primary_reason = str(primary_error)

        try:
            result = self.backup.colorize(image, saturation=saturation)
        except ColorizationError as backup_error:
            raise ColorizationError(
                "Colorization failed with both colorizers. "
                f"DDColor: {primary_reason} | Backup: {backup_error}") from backup_error
        note = ("The main colorizer (DDColor) was unavailable, so the simpler "
                "backup colorizer was used. Colours may look more muted. "
                f"Reason: {primary_reason}")
        return ChainResult(result, METHOD_BACKUP, [note])


# ---------------------------------------------------------------------------
# Self-test (needs the vendored code, but no weights)
# ---------------------------------------------------------------------------
class _FakeModel(torch.nn.Module):
    """Stand-in network: predicts one constant colour, records its input."""

    def __init__(self, a: float, b: float):
        super().__init__()
        self.a, self.b = a, b
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        out = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])
        out[:, 0] = self.a
        out[:, 1] = self.b
        return out


class _Stub:
    """Stand-in colorizer for testing the fallback chain."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def colorize(self, image, saturation=1.0):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _lightness(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return cv2.cvtColor(array, cv2.COLOR_RGB2Lab)[:, :, 0]


def _run_self_test() -> int:
    failures = 0

    def check(description: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        if condition:
            print(f" [PASS] {description}")
        else:
            failures += 1
            print(f" [FAIL] {description} {detail}")

    def raises(expected, function) -> bool:
        try:
            function()
        except expected:
            return True
        except Exception:
            return False
        return False

    ramp = np.tile(np.linspace(30, 220, 300), (120, 1)).astype(np.uint8)   # 120 x 300
    gray_photo = Image.fromarray(ramp, "L").convert("RGB")

    # ---- 1. Our pre/post-processing, with a stand-in network ----
    fake = _FakeModel(0.0, 0.0)
    plain = BackupColorizer(model=fake).colorize(gray_photo)
    pixels = np.asarray(plain).astype(int)
    check("output keeps size (non-square 300x120) and RGB mode",
          plain.size == gray_photo.size and plain.mode == "RGB")
    check("a=b=0 gives a gray picture (R = G = B)",
          np.abs(pixels[..., 0] - pixels[..., 1]).max() <= 1
          and np.abs(pixels[..., 1] - pixels[..., 2]).max() <= 1)
    check("network input is 1x1x256x256 with lightness in 0..100",
          tuple(fake.last_input.shape) == (1, 1, 256, 256)
          and float(fake.last_input.min()) >= 0.0 and float(fake.last_input.max()) <= 100.0)

    tinted = BackupColorizer(model=_FakeModel(15.0, 20.0)).colorize(gray_photo)
    t = np.asarray(tinted).astype(int)
    check("predicted colour appears in the result",
          np.abs(t[..., 0] - t[..., 2]).mean() > 10)
    check("original lightness is preserved (mean change under 3)",
          float(np.abs(_lightness(tinted) - _lightness(gray_photo)).mean()) < 3.0)
    muted = BackupColorizer(model=_FakeModel(15.0, 20.0)).colorize(gray_photo, saturation=0.0)
    m = np.asarray(muted).astype(int)
    check("saturation 0 removes all colour", np.abs(m[..., 0] - m[..., 2]).max() <= 1)
    check("invalid saturation is rejected",
          raises(ValueError, lambda: BackupColorizer(model=fake).colorize(
              gray_photo, saturation=-1)))

    # ---- 2. The real network definition and the weights handling ----
    try:
        from src.vendor.siggraph17.siggraph17 import SIGGRAPHGenerator
        with torch.no_grad():
            out = SIGGRAPHGenerator().eval()(torch.rand(1, 1, 64, 64) * 100)
        check("real SIGGRAPH17 network builds and predicts a 2-channel colour map",
              tuple(out.shape) == (1, 2, 64, 64) and bool(torch.isfinite(out).all()))
    except ImportError as error:
        failures += 1
        print(f" [FAIL] backup code missing: run python scripts\\setup_backup_colorizer.py ({error})")

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        with mock.patch.object(sys.modules[__name__], "WEIGHTS_PATH", folder / "none.pth"):
            check("missing weights give BackupWeightsMissingError",
                  raises(BackupWeightsMissingError,
                         lambda: BackupColorizer().colorize(gray_photo)))
        broken = folder / "broken.pth"
        broken.write_bytes(b"this is not a model" * 100)
        with mock.patch.object(sys.modules[__name__], "WEIGHTS_PATH", broken):
            check("a damaged weights file gives a clear ColorizationError",
                  raises(ColorizationError, lambda: BackupColorizer().colorize(gray_photo)))

    # ---- 3. Fallback chain, with stand-in colorizers ----
    good_primary, good_backup = Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")

    stub_primary, stub_backup = _Stub(good_primary), _Stub(good_backup)
    outcome = ColorizerChain(stub_primary, stub_backup).colorize(gray_photo)
    check("primary works: used, backup untouched, no notes",
          outcome.method == METHOD_DDCOLOR and outcome.image is good_primary
          and stub_backup.calls == 0 and outcome.notes == [])

    stub_primary, stub_backup = _Stub(ColorizationError("model broke")), _Stub(good_backup)
    outcome = ColorizerChain(stub_primary, stub_backup).colorize(gray_photo)
    check("primary fails: backup used and the reason is reported",
          outcome.method == METHOD_BACKUP and outcome.image is good_backup
          and "model broke" in outcome.notes[0])

    try:
        ColorizerChain(_Stub(ColorizationError("reasonA")),
                       _Stub(ColorizationError("reasonB"))).colorize(gray_photo)
        check("both fail: one clear error naming both reasons", False)
    except ColorizationError as error:
        check("both fail: one clear error naming both reasons",
              "reasonA" in str(error) and "reasonB" in str(error))

    check("programming errors (ValueError) are NOT hidden by the backup",
          raises(ValueError, lambda: ColorizerChain(
              _Stub(ValueError("bad argument")), _Stub(good_backup)
          ).colorize(gray_photo)))

    chain = ColorizerChain(Colorizer(weights_path="no_such_folder/none.pt"),
                           _Stub(good_backup))
    outcome = chain.colorize(gray_photo)
    check("real Colorizer with missing weights falls back to the backup",
          outcome.method == METHOD_BACKUP and "not found" in outcome.notes[0])

    print("-" * 62)
    print(f" Backup weights present: {'yes' if WEIGHTS_PATH.is_file() else 'NO'} ({WEIGHTS_PATH})")
    print(" (The self-test does not load the real weights. To test them, run\n"
          "  python -m src.colorize_backup outputs\\preview_gray.png)")
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


def _colorize_file(path: str, saturation: float, use_chain: bool) -> int:
    from src.image_io import ImageValidationError, load_image_from_path
    from src.preprocessing import limit_working_size

    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1
    working = limit_working_size(loaded.image).image

    started = time.time()
    try:
        if use_chain:
            outcome = ColorizerChain().colorize(working, saturation=saturation)
            result, label = outcome.image, outcome.method
            for note in outcome.notes:
                print(f" NOTE: {note}")
        else:
            result, label = BackupColorizer().colorize(working, saturation), METHOD_BACKUP
    except ColorizationError as error:
        print(f" ERROR: {error}")
        return 1

    config.ensure_directories()
    out_path = config.OUTPUTS_DIR / f"colorized_{label}_result.png"
    result.save(out_path)
    print(f" Method used: {label}")
    print(f" Done in {time.time() - started:.1f} s")
    print(f" Saved: {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup colorizer and fallback chain")
    parser.add_argument("image", nargs="?", help="image to colorize (omit to run the self-test)")
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--chain", action="store_true",
                        help="use DDColor first and fall back to the backup")
    arguments = parser.parse_args()
    if arguments.image:
        sys.exit(_colorize_file(arguments.image, arguments.saturation, arguments.chain))
    sys.exit(_run_self_test())