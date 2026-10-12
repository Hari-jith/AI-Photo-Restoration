"""
colorize.py

Colorizes black-and-white photos with DDColor (ICCV 2023, Apache 2.0).

    from src.colorize import Colorizer

    colorizer = Colorizer()                   # nothing is loaded yet
    color_image = colorizer.colorize(gray_image)       # loads on first use
    color_image = colorizer.colorize(gray_image, saturation=1.2)

How DDColor works (short version)
  Photos can be split into LIGHTNESS ("L": the black-and-white picture) and
  COLOUR ("a" and "b": two channels that carry the colour). DDColor keeps the
  original L untouched and only predicts a and b. So your photo's detail and
  brightness are preserved exactly; only colour is added.
  The network always looks at a 512 x 512 copy, then its colour prediction
  is stretched back to the full photo size and combined with the full-
  resolution L. Result: processing time does not grow with photo size.

Honest limits
  The colours are a plausible GUESS, not the real historical colours.
  Unusual objects, uniforms or clothing can come out wrong.

Self-test (no weights needed):   python -m src.colorize
Colorize a file:                 python -m src.colorize samples\\test_1.jpg
"""

import argparse
import pickle
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src import config
from src.device import free_memory, get_device_info, is_out_of_memory_error
from src.logging_setup import get_logger

logger = get_logger(__name__)

# --- Settings ----------------------------------------------------------------
DDCOLOR_WEIGHTS_PATH = config.MODELS_DIR / "ddcolor" / "pytorch_model.pt"
MODEL_INPUT_SIZE = 512            # the size DDColor's large model was built for
MIN_SATURATION = 0.0              # 0 = no colour at all
MAX_SATURATION = 2.0              # 1 = as predicted, 2 = twice as vivid
# Entries the model creates itself and that need not be in a checkpoint.
NON_LEARNED_KEYS = {"mean", "std"}


# ---------------------------------------------------------------------------
# Errors (messages are written for beginners)
# ---------------------------------------------------------------------------
class ColorizationError(Exception):
    """Colorization could not be completed."""


class ModelWeightsMissingError(ColorizationError):
    """The DDColor weights file is not where we expect it."""


# ---------------------------------------------------------------------------
# Building the model and loading weights
# ---------------------------------------------------------------------------
def build_model(model_size: str = "large", input_size: int = MODEL_INPUT_SIZE):
    """Create the DDColor network (random weights, CPU, evaluation mode).

    The weights we downloaded are the 'large' model. 'tiny' exists only so
    our self-test can run quickly without any download.
    """
    if model_size not in ("tiny", "large"):
        raise ValueError(f"model_size must be 'tiny' or 'large', got '{model_size}'.")
    try:
        from src.vendor.ddcolor.model import DDColor
    except ImportError as error:
        raise ColorizationError(
            "The DDColor model code is missing. Run this once from the "
            "project folder:  python scripts\\setup_ddcolor_code.py"
            f"  (details: {error})") from error

    encoder = "convnext-l" if model_size == "large" else "convnext-t"
    model = DDColor(
        encoder_name=encoder,
        decoder_name="MultiScaleColorDecoder",
        input_size=[input_size, input_size],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )
    return model.eval()


def _read_checkpoint(path: Path) -> dict:
    """Read a checkpoint file and return the dictionary of weights.

    We first try PyTorch's "weights only" mode, which refuses to run any
    code stored inside the file (safer). Only if that mode cannot read the
    file do we fall back to full loading, with a warning.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        logger.warning(
            "Safe loading of %s failed (%s). Falling back to full loading. "
            "This is acceptable only for the official file downloaded by "
            "our script.", path.name, str(error)[:120])
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as inner_error:
            raise ColorizationError(
                f"The weights file {path.name} could not be read. The download "
                "may be damaged: delete it and download it again.") from inner_error
    except Exception as error:
        raise ColorizationError(
            f"The weights file {path.name} could not be read. The download "
            "may be damaged (incomplete): delete it and download it again."
        ) from error

    state = checkpoint
    if isinstance(checkpoint, dict):
        for wrapper_key in ("params", "state_dict"):
            if wrapper_key in checkpoint and isinstance(checkpoint[wrapper_key], dict):
                state = checkpoint[wrapper_key]
                break
    if not isinstance(state, dict) or not state:
        raise ColorizationError(
            f"{path.name} does not look like a model weights file.")
    # Some tools save keys with a "module." prefix (from DataParallel).
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def load_weights_into(model, path) -> None:
    """Load a checkpoint into the model and VERIFY that it fitted.

    DDColor's own code loads with strict=False, which silently ignores
    mismatches and would give a model with random (garbage) colours. We
    check explicitly and raise a clear error instead.
    """
    path = Path(path)
    state = _read_checkpoint(path)
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise ColorizationError(
            "The weights file does not match the model (different sizes). "
            "Make sure it is the file downloaded by "
            "scripts\\download_weights.py.") from error

    missing = [key for key in result.missing_keys if key not in NON_LEARNED_KEYS]
    if missing:
        raise ColorizationError(
            f"The weights file is missing {len(missing)} expected entries "
            f"(for example '{missing[0]}'), so it does not belong to this "
            "model. Delete models\\ddcolor\\pytorch_model.pt and download it again.")
    if result.unexpected_keys:
        logger.warning("Weights file has %d unused entries (e.g. '%s'); ignored.",
                       len(result.unexpected_keys), result.unexpected_keys[0])


# ---------------------------------------------------------------------------
# The colorizer
# ---------------------------------------------------------------------------
class Colorizer:
    """Loads DDColor lazily (on first use) and colorizes images."""

    def __init__(self, weights_path=None, model_size: str = "large",
                 input_size: int = MODEL_INPUT_SIZE, model=None):
        """
        Args:
            weights_path: path to pytorch_model.pt (default: models/ddcolor/).
            model_size: "large" (real weights) or "tiny" (testing only).
            input_size: side of the square the network looks at.
            model: an already-built model. Used by the self-test to inject a
                stand-in; normal code leaves this as None.
        """
        self.weights_path = Path(weights_path) if weights_path else DDCOLOR_WEIGHTS_PATH
        self.model_size = model_size
        self.input_size = input_size
        self._model = model
        self._device = torch.device("cpu") if model is not None else None
        # The model keeps temporary state inside, so only one image at a time.
        self._lock = threading.Lock()

    def weights_available(self) -> bool:
        return self.weights_path.is_file()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.weights_available():
            raise ModelWeightsMissingError(
                f"The DDColor weights were not found at {self.weights_path}. "
                "Download them with:  python scripts\\download_weights.py "
                "--only ddcolor   (the script prints manual steps if the "
                "automatic download fails).")
        started = time.time()
        info = get_device_info()
        model = build_model(self.model_size, self.input_size)
        load_weights_into(model, self.weights_path)
        self._model = model.to(info.device)
        self._device = info.device
        logger.info("DDColor loaded on %s in %.1f s", info.summary(), time.time() - started)

    def colorize(self, image: Image.Image, saturation: float = 1.0) -> Image.Image:
        """Return a colorized copy of the image (same size, RGB).

        Args:
            image: any Pillow image (only its lightness is used).
            saturation: colour strength. 1.0 = as predicted, below 1 = more
                muted, above 1 = more vivid (range 0.0 to 2.0).
        """
        if not MIN_SATURATION <= saturation <= MAX_SATURATION:
            raise ValueError(
                f"saturation must be between {MIN_SATURATION} and {MAX_SATURATION}.")
        with self._lock:
            self._ensure_loaded()
            started = time.time()
            try:
                result = self._run(image, saturation)
            except Exception as error:
                if self._device.type == "cuda" and is_out_of_memory_error(error):
                    logger.warning("GPU ran out of memory; retrying on the CPU.")
                    free_memory()
                    self._model.to("cpu")
                    self._device = torch.device("cpu")
                    try:
                        result = self._run(image, saturation)
                    except Exception as retry_error:
                        raise ColorizationError(
                            f"Colorization failed even on the CPU: {retry_error}"
                        ) from retry_error
                elif is_out_of_memory_error(error):
                    raise ColorizationError(
                        "Not enough memory to colorize this image. Close other "
                        "programs, or try a smaller image.") from error
                else:
                    raise ColorizationError(f"Colorization failed: {error}") from error
            logger.info("Colorized %dx%d image in %.1f s on %s", image.width,
                        image.height, time.time() - started, self._device.type)
            return result

    def _run(self, image: Image.Image, saturation: float) -> Image.Image:
        size = self.input_size
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]

        # Lightness of the full-size photo: this goes back into the result.
        original_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[:, :, :1]

        # Network input: the lightness of a size x size copy, shown to the
        # network as a gray RGB picture (a and b set to zero).
        small = cv2.resize(rgb, (size, size))
        small_l = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)[:, :, :1]
        gray_lab = np.concatenate(
            (small_l, np.zeros_like(small_l), np.zeros_like(small_l)), axis=-1)
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
        tensor = (torch.from_numpy(gray_rgb.transpose(2, 0, 1).copy())
                  .float().unsqueeze(0).to(self._device))

        with torch.inference_mode():
            predicted_ab = self._model(tensor).float().cpu()   # (1, 2, size, size)

        # Stretch the colour prediction to the photo's size. (The official
        # script uses nearest-neighbour; bilinear gives smoother colour edges.)
        ab = F.interpolate(predicted_ab, size=(height, width),
                           mode="bilinear", align_corners=False)[0]
        ab = ab.numpy().transpose(1, 2, 0) * saturation

        lab = np.concatenate((original_l, ab), axis=-1).astype(np.float32)
        result_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # Clip BEFORE converting to bytes: some Lab colours fall just outside
        # what a screen can show, and unclipped values would wrap around
        # (a very bright pixel turning dark).
        result_bytes = (np.clip(result_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return Image.fromarray(result_bytes, "RGB")


# ---------------------------------------------------------------------------
# Self-test (needs the model code, but NOT the weights)
# ---------------------------------------------------------------------------
class _FakeAbModel(torch.nn.Module):
    """Stand-in model that predicts a constant colour. Lets us test our own
    pre- and post-processing without the real weights."""

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


class _OutOfMemoryModel(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("out of memory (simulated)")


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

    # ---- 1. Our colour pipeline, using stand-in models ----
    ramp = np.tile(np.linspace(30, 220, 300), (120, 1)).astype(np.uint8)   # 120 x 300
    gray_photo = Image.fromarray(ramp, "L").convert("RGB")

    recorder = _FakeAbModel(0.0, 0.0)
    plain = Colorizer(model=recorder).colorize(gray_photo)
    pixels = np.asarray(plain).astype(int)
    check("output keeps the photo's size (non-square 300x120) and RGB mode",
          plain.size == gray_photo.size and plain.mode == "RGB")
    check("a=b=0 gives a gray picture (R = G = B)",
          np.abs(pixels[..., 0] - pixels[..., 1]).max() <= 1
          and np.abs(pixels[..., 1] - pixels[..., 2]).max() <= 1)
    check("network input is 1x3x512x512 and gray",
          tuple(recorder.last_input.shape) == (1, 3, 512, 512)
          and float((recorder.last_input[:, 0] - recorder.last_input[:, 2]).abs().max()) < 1e-3)

    tinted = Colorizer(model=_FakeAbModel(15.0, 20.0)).colorize(gray_photo)
    t = np.asarray(tinted).astype(int)
    check("predicted colour really appears in the result",
          np.abs(t[..., 0] - t[..., 2]).mean() > 10)
    check("original lightness is preserved (mean change under 3)",
          float(np.abs(_lightness(tinted) - _lightness(gray_photo)).mean()) < 3.0)

    muted = Colorizer(model=_FakeAbModel(15.0, 20.0)).colorize(gray_photo, saturation=0.0)
    m = np.asarray(muted).astype(int)
    check("saturation 0 removes all colour",
          np.abs(m[..., 0] - m[..., 2]).max() <= 1)
    check("invalid saturation is rejected",
          raises(ValueError, lambda: Colorizer(model=recorder).colorize(gray_photo, saturation=5)))
    check("out-of-memory becomes a friendly ColorizationError",
          raises(ColorizationError,
                 lambda: Colorizer(model=_OutOfMemoryModel()).colorize(gray_photo)))

    # ---- 2. Weights handling, using the TINY model and temporary files ----
    check("missing weights file gives ModelWeightsMissingError",
          raises(ModelWeightsMissingError,
                 lambda: Colorizer(weights_path="no_such_folder/none.pt").colorize(gray_photo)))

    try:
        torch.manual_seed(0)
        source = build_model("tiny", 256)
        probe = torch.rand(1, 3, 256, 256)
        with torch.inference_mode():
            expected = source(probe)
        good_state = source.state_dict()

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            variants = {
                "wrapped in 'params'": {"params": good_state},
                "plain state dict": dict(good_state),
                "'module.' prefix": {f"module.{k}": v for k, v in good_state.items()},
            }
            for name, content in variants.items():
                file_path = folder / "w.pt"
                torch.save(content, file_path)
                torch.manual_seed(1)                       # different random start
                target = build_model("tiny", 256)
                load_weights_into(target, file_path)
                with torch.inference_mode():
                    actual = target(probe)
                check(f"weights load correctly ({name})",
                      float((actual - expected).abs().max()) < 1e-4)

            wrong_names = folder / "wrong.pt"
            torch.save({"params": {"foo": torch.zeros(1)}}, wrong_names)
            check("checkpoint with wrong entries is rejected",
                  raises(ColorizationError,
                         lambda: load_weights_into(build_model("tiny", 256), wrong_names)))

            wrong_shape = folder / "shape.pt"
            bad_state = dict(good_state)
            first_key = next(k for k, v in bad_state.items() if v.ndim >= 2)
            bad_state[first_key] = torch.zeros(3, 3)
            torch.save(bad_state, wrong_shape)
            check("checkpoint with wrong tensor sizes is rejected",
                  raises(ColorizationError,
                         lambda: load_weights_into(build_model("tiny", 256), wrong_shape)))

            broken = folder / "broken.pt"
            broken.write_bytes(b"this is not a model" * 100)
            check("damaged weights file is rejected with a clear error",
                  raises(ColorizationError,
                         lambda: load_weights_into(build_model("tiny", 256), broken)))
    except ColorizationError as error:
        failures += 1
        print(f" [FAIL] could not build the model: {error}")

    print("-" * 62)
    present = DDCOLOR_WEIGHTS_PATH.is_file()
    print(f" Real weights file present: {'yes' if present else 'NO'} "
          f"({DDCOLOR_WEIGHTS_PATH})")
    print(" (The self-test does not load the real weights. To test them, run\n"
          "  python -m src.colorize samples\\test_1.jpg)")
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


def _colorize_file(path: str, saturation: float) -> int:
    from src.analysis import analyze_image
    from src.image_io import ImageValidationError, load_image_from_path
    from src.preprocessing import limit_working_size

    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1

    working = limit_working_size(loaded.image).image
    report = analyze_image(working)
    print(f" Colour type of input: {report.color_kind}")
    if not report.needs_colorization:
        print(" Note: this photo already has colour. DDColor only uses its\n"
              "       lightness, so it will be RECOLOURED, not preserved.")

    print(" Loading DDColor and colorizing (the first run can take a minute)...")
    started = time.time()
    try:
        result = Colorizer().colorize(working, saturation=saturation)
    except ColorizationError as error:
        print(f" ERROR: {error}")
        return 1

    config.ensure_directories()
    before_path = config.OUTPUTS_DIR / "colorized_before.png"
    result_path = config.OUTPUTS_DIR / "colorized_result.png"
    working.save(before_path)
    result.save(result_path)
    print(f" Done in {time.time() - started:.1f} s on {get_device_info().kind.upper()}")
    print(f" Saved: {before_path}")
    print(f" Saved: {result_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDColor colorization")
    parser.add_argument("image", nargs="?", help="image to colorize (omit to run the self-test)")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="colour strength 0.0-2.0 (default 1.0)")
    arguments = parser.parse_args()
    if arguments.image:
        sys.exit(_colorize_file(arguments.image, arguments.saturation))
    sys.exit(_run_self_test())