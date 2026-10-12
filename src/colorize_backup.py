"""
colorize_backup.py

A second, independent colorizer (Zhang, Isola & Efros, ECCV 2016, run with
OpenCV's built-in neural-network engine) plus the fallback logic that uses it
when DDColor is unavailable.

    from src.colorize_backup import ColorizerChain

    chain = ColorizerChain()                      # DDColor first, backup second
    result = chain.colorize(gray_image)
    result.image          # colourized Pillow image
    result.method         # "ddcolor" or "opencv_zhang"
    result.notes          # plain-English remarks, e.g. why the backup was used

Why have a backup?
  * It needs no PyTorch and no extra Python packages (only OpenCV).
  * It is small (about 120 MB) and fast on a CPU.
  * If DDColor fails to load or runs out of memory, the website still works.
Its colours are usually MORE MUTED and less accurate than DDColor's, and it
looks at the photo at only 224 x 224 pixels. It is a safety net, not an equal.

Files (downloaded by scripts/download_weights.py into models/colorizer_backup/):
  colorization_deploy_v2.prototxt   network description   (GitHub, BSD-2)
  pts_in_hull.npy                   the 313 colour bins    (GitHub, BSD-2)
  colorization_release_v2.caffemodel the trained weights   (official Berkeley server)
Source: https://github.com/richzhang/colorization

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
from PIL import Image

from src import config
from src.colorize import (MAX_SATURATION, MIN_SATURATION, ColorizationError,
                          Colorizer, ModelWeightsMissingError)
from src.logging_setup import get_logger

logger = get_logger(__name__)

# --- Settings ----------------------------------------------------------------
BACKUP_DIR = config.MODELS_DIR / "colorizer_backup"
PROTOTXT_PATH = BACKUP_DIR / "colorization_deploy_v2.prototxt"
CLUSTER_CENTRES_PATH = BACKUP_DIR / "pts_in_hull.npy"
CAFFEMODEL_PATH = BACKUP_DIR / "colorization_release_v2.caffemodel"

INPUT_SIZE = 224            # the network always looks at 224 x 224 pixels
LIGHTNESS_OFFSET = 50.0     # the network was trained on lightness minus 50
NUM_COLOUR_BINS = 313       # the network picks among 313 colour "bins"
REBALANCE_VALUE = 2.606     # colour-boost constant from the authors' demo

METHOD_DDCOLOR = "ddcolor"
METHOD_BACKUP = "opencv_zhang"


class BackupWeightsMissingError(ColorizationError):
    """One or more backup-colorizer files are missing."""


# ---------------------------------------------------------------------------
# Loading the OpenCV network
# ---------------------------------------------------------------------------
def _load_network():
    """Read the Caffe network with OpenCV and install the two pieces the
    authors ship separately (the colour bins and a rebalancing constant)."""
    required = [PROTOTXT_PATH, CLUSTER_CENTRES_PATH, CAFFEMODEL_PATH]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise BackupWeightsMissingError(
            f"Backup colorizer files missing: {', '.join(missing)}. Download "
            "them with:  python scripts\\download_weights.py   (the script "
            "prints manual steps if an automatic download fails).")
    try:
        network = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
        centres = np.load(CLUSTER_CENTRES_PATH).astype(np.float32)
        if centres.shape != (NUM_COLOUR_BINS, 2):
            raise ColorizationError(
                f"pts_in_hull.npy has shape {centres.shape}, expected "
                f"({NUM_COLOUR_BINS}, 2). Delete it and download it again.")
        centres = centres.transpose().reshape(2, NUM_COLOUR_BINS, 1, 1)
        network.getLayer(network.getLayerId("class8_ab")).blobs = [centres]
        network.getLayer(network.getLayerId("conv8_313_rh")).blobs = [
            np.full((1, NUM_COLOUR_BINS), REBALANCE_VALUE, dtype=np.float32)]
    except cv2.error as error:
        raise ColorizationError(
            "OpenCV could not load the backup colorizer files (they may be "
            "damaged or incomplete: delete the folder models\\colorizer_backup "
            f"and download again). OpenCV {cv2.__version__} said: "
            f"{str(error)[:200]}") from error
    return network


# ---------------------------------------------------------------------------
# The backup colorizer
# ---------------------------------------------------------------------------
class BackupColorizer:
    """Colorizes with the Zhang et al. network. Loads lazily on first use."""

    def __init__(self, network=None):
        """network: an already-loaded network (used by the self-test)."""
        self._network = network
        self._lock = threading.Lock()   # the network keeps internal state

    def weights_available(self) -> bool:
        return all(p.is_file() for p in
                   (PROTOTXT_PATH, CLUSTER_CENTRES_PATH, CAFFEMODEL_PATH))

    def colorize(self, image: Image.Image, saturation: float = 1.0) -> Image.Image:
        """Return a colourized copy (same size, RGB). Same interface as
        Colorizer.colorize, so the two are interchangeable."""
        if not MIN_SATURATION <= saturation <= MAX_SATURATION:
            raise ValueError(
                f"saturation must be between {MIN_SATURATION} and {MAX_SATURATION}.")
        with self._lock:
            if self._network is None:
                self._network = _load_network()
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

        # Network input: lightness of a 224 x 224 copy, shifted by -50.
        small = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        small_l = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)[:, :, 0] - LIGHTNESS_OFFSET
        blob = small_l.reshape(1, 1, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

        self._network.setInput(blob)
        predicted = self._network.forward()[0]            # (2, 56, 56)
        ab = predicted.transpose(1, 2, 0).astype(np.float32)
        ab = cv2.resize(ab, (width, height)) * saturation  # stretch to photo size

        lab = np.concatenate((original_l[:, :, np.newaxis], ab), axis=2)
        result_rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
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
# Self-test (needs no weights)
# ---------------------------------------------------------------------------
class _FakeNetwork:
    """Stand-in for the OpenCV network: predicts one constant colour."""

    def __init__(self, a: float, b: float):
        self.a, self.b = a, b
        self.last_blob = None

    def setInput(self, blob):
        self.last_blob = blob

    def forward(self):
        out = np.zeros((1, 2, 56, 56), dtype=np.float32)   # real output is 56 x 56
        out[0, 0] = self.a
        out[0, 1] = self.b
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
    fake = _FakeNetwork(0.0, 0.0)
    plain = BackupColorizer(network=fake).colorize(gray_photo)
    pixels = np.asarray(plain).astype(int)
    check("output keeps size (non-square 300x120) and RGB mode",
          plain.size == gray_photo.size and plain.mode == "RGB")
    check("a=b=0 gives a gray picture (R = G = B)",
          np.abs(pixels[..., 0] - pixels[..., 1]).max() <= 1
          and np.abs(pixels[..., 1] - pixels[..., 2]).max() <= 1)
    check("network input is 1x1x224x224 with lightness shifted into -50..50",
          fake.last_blob.shape == (1, 1, 224, 224)
          and fake.last_blob.min() >= -50.0 and fake.last_blob.max() <= 50.0)

    tinted = BackupColorizer(network=_FakeNetwork(15.0, 20.0)).colorize(gray_photo)
    t = np.asarray(tinted).astype(int)
    check("predicted colour appears in the result",
          np.abs(t[..., 0] - t[..., 2]).mean() > 10)
    check("original lightness is preserved (mean change under 3)",
          float(np.abs(_lightness(tinted) - _lightness(gray_photo)).mean()) < 3.0)
    muted = BackupColorizer(network=_FakeNetwork(15.0, 20.0)).colorize(
        gray_photo, saturation=0.0)
    m = np.asarray(muted).astype(int)
    check("saturation 0 removes all colour", np.abs(m[..., 0] - m[..., 2]).max() <= 1)
    check("invalid saturation is rejected",
          raises(ValueError, lambda: BackupColorizer(network=fake).colorize(
              gray_photo, saturation=-1)))

    # ---- 2. Files missing / damaged ----
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        with mock.patch.object(sys.modules[__name__], "PROTOTXT_PATH", folder / "a.prototxt"), \
             mock.patch.object(sys.modules[__name__], "CLUSTER_CENTRES_PATH", folder / "b.npy"), \
             mock.patch.object(sys.modules[__name__], "CAFFEMODEL_PATH", folder / "c.caffemodel"):
            check("missing files give BackupWeightsMissingError",
                  raises(BackupWeightsMissingError,
                         lambda: BackupColorizer().colorize(gray_photo)))

            (folder / "a.prototxt").write_text('name: "broken"\n')
            np.save(folder / "b.npy", np.zeros((313, 2), dtype=np.int64))
            (folder / "c.caffemodel").write_bytes(b"this is not a model" * 100)
            check("damaged files give a clear ColorizationError (no crash)",
                  raises(ColorizationError,
                         lambda: BackupColorizer().colorize(gray_photo)))

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

    check("both fail: one clear error naming both reasons",
          raises(ColorizationError, lambda: ColorizerChain(
              _Stub(ColorizationError("A")), _Stub(ColorizationError("B"))
          ).colorize(gray_photo)))
    try:
        ColorizerChain(_Stub(ColorizationError("reasonA")),
                       _Stub(ColorizationError("reasonB"))).colorize(gray_photo)
    except ColorizationError as error:
        check("error message contains both reasons",
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
    print(f" Backup files present: "
          f"{'yes' if BackupColorizer().weights_available() else 'NO'} ({BACKUP_DIR})")
    print(" (The self-test does not load the real files. To test them, run\n"
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