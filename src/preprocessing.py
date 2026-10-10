"""
preprocessing.py

Prepares images for the AI models:

  * conversions between Pillow, RGB NumPy and BGR NumPy formats
  * limiting the working size (never enlarging)
  * making a clean grayscale copy for colorization

    from src.preprocessing import limit_working_size, pil_to_bgr_array

    result = limit_working_size(loaded.image)
    working_image = result.image          # Pillow RGB, longest side <= limit
    bgr = pil_to_bgr_array(working_image) # for OpenCV / GFPGAN / Real-ESRGAN

Self-test:    python -m src.preprocessing
Try a file:   python -m src.preprocessing samples\\test1.jpg
"""

import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src import config
from src.logging_setup import get_logger

logger = get_logger(__name__)

# The longest side of the image the models will work on. Larger photos are
# shrunk to this size before restoration. 1024 keeps CPU processing
# practical. Raised later when a GPU is available (Modules 7 and 22).
MAX_WORKING_SIDE = 1024


# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------
def _check_rgb_array(array: np.ndarray, name: str) -> None:
    """Raise a clear error unless the array is uint8 with shape (H, W, 3)."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array, got {type(array).__name__}.")
    if array.dtype != np.uint8:
        raise ValueError(
            f"{name} must have dtype uint8 (values 0-255), got {array.dtype}."
        )
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"{name} must have shape (height, width, 3), got {array.shape}."
        )


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Pillow image -> writable uint8 array of shape (H, W, 3) in RGB order."""
    return np.array(image.convert("RGB"), dtype=np.uint8)


def rgb_array_to_pil(array: np.ndarray) -> Image.Image:
    """RGB uint8 array (H, W, 3) -> Pillow image."""
    _check_rgb_array(array, "RGB array")
    return Image.fromarray(array, mode="RGB")


def rgb_to_bgr(array: np.ndarray) -> np.ndarray:
    """Swap channel order RGB -> BGR (the array stays contiguous)."""
    _check_rgb_array(array, "RGB array")
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(array: np.ndarray) -> np.ndarray:
    """Swap channel order BGR -> RGB (the array stays contiguous)."""
    _check_rgb_array(array, "BGR array")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def pil_to_bgr_array(image: Image.Image) -> np.ndarray:
    """Pillow image -> BGR uint8 array, as OpenCV-style libraries expect."""
    return rgb_to_bgr(pil_to_rgb_array(image))


def bgr_array_to_pil(array: np.ndarray) -> Image.Image:
    """BGR uint8 array -> Pillow image (RGB)."""
    return rgb_array_to_pil(bgr_to_rgb(array))


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------
@dataclass
class ResizeResult:
    """Outcome of limit_working_size."""

    image: Image.Image
    scale: float            # 1.0 = unchanged, 0.5 = shrunk to half size
    original_size: tuple    # (width, height) before resizing

    @property
    def was_resized(self) -> bool:
        return self.scale < 1.0


def limit_working_size(
    image: Image.Image, max_side: int = MAX_WORKING_SIDE
) -> ResizeResult:
    """Shrink the image so its longest side is at most max_side.

    Images that are already small enough are returned unchanged. This
    function never enlarges an image (that is the super-resolution
    model's job). The aspect ratio is always preserved.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return ResizeResult(image=image, scale=1.0, original_size=(width, height))

    scale = max_side / longest
    # max(1, ...) protects extremely thin images from rounding to 0 pixels.
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # LANCZOS is a high-quality filter that avoids jagged edges when shrinking.
    resized = image.resize(new_size, Image.LANCZOS)
    logger.info("Working size limited: %dx%d -> %dx%d",
                width, height, new_size[0], new_size[1])
    return ResizeResult(image=resized, scale=scale, original_size=(width, height))


def resize_to_match(image: Image.Image, size: tuple) -> Image.Image:
    """Resize an image to an exact (width, height). Used later to bring a
    result back to a known size (for example, to merge colour back into
    the original). Returns the same image if the size already matches."""
    if image.size == tuple(size):
        return image
    return image.resize(tuple(size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Grayscale preparation
# ---------------------------------------------------------------------------
def to_grayscale_rgb(image: Image.Image) -> Image.Image:
    """Remove all colour but keep 3 channels (R = G = B).

    Used before colorization so that a sepia or colour-cast photo does not
    bias the model. Brightness is computed with the standard luminance
    weights (green counts most, because the eye is most sensitive to it).
    """
    return image.convert("L").convert("RGB")


# ---------------------------------------------------------------------------
# Self-test and file preview
# ---------------------------------------------------------------------------
def _run_self_test() -> int:
    failures = 0

    def check(description: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        if condition:
            print(f" [PASS] {description}")
        else:
            failures += 1
            print(f" [FAIL] {description} {detail}")

    rng = np.random.default_rng(seed=0)
    random_array = rng.integers(0, 256, (50, 80, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_array, "RGB")

    # --- conversions ---
    round_trip = rgb_array_to_pil(pil_to_rgb_array(random_image))
    check("Pillow -> array -> Pillow gives identical pixels",
          np.array_equal(np.array(round_trip), random_array))

    array = pil_to_rgb_array(random_image)
    check("array is writable and contiguous",
          array.flags.writeable and array.flags.c_contiguous)

    red_image = Image.new("RGB", (4, 4), (255, 0, 0))
    bgr_pixel = pil_to_bgr_array(red_image)[0, 0].tolist()
    check("pure red becomes [0, 0, 255] in BGR", bgr_pixel == [0, 0, 255],
          f"(got {bgr_pixel})")

    bgr_round_trip = bgr_array_to_pil(pil_to_bgr_array(random_image))
    check("Pillow -> BGR -> Pillow gives identical pixels",
          np.array_equal(np.array(bgr_round_trip), random_array))

    check("BGR array is contiguous",
          pil_to_bgr_array(random_image).flags.c_contiguous)

    # --- input checking ---
    try:
        rgb_array_to_pil(random_array.astype(np.float32))
        check("float array is rejected", False)
    except ValueError:
        check("float array is rejected", True)

    try:
        rgb_to_bgr(random_array[:, :, 0])      # 2D grayscale array
        check("2-D array is rejected", False)
    except ValueError:
        check("2-D array is rejected", True)

    # --- resizing ---
    big = Image.new("RGB", (3000, 2000))
    result = limit_working_size(big, max_side=1024)
    ratio_before = 3000 / 2000
    ratio_after = result.image.width / result.image.height
    check("large image shrinks to longest side 1024",
          max(result.image.size) == 1024 and result.was_resized,
          f"(got {result.image.size})")
    check("aspect ratio preserved",
          abs(ratio_after - ratio_before) / ratio_before < 0.01)

    small = Image.new("RGB", (500, 400))
    small_result = limit_working_size(small, max_side=1024)
    check("small image is NOT enlarged",
          small_result.image.size == (500, 400) and small_result.scale == 1.0
          and not small_result.was_resized)

    thin_result = limit_working_size(Image.new("RGB", (5000, 20)), max_side=1024)
    check("very thin image keeps at least 1 pixel",
          min(thin_result.image.size) >= 1, f"(got {thin_result.image.size})")

    check("resize_to_match gives exact size",
          resize_to_match(Image.new("RGB", (100, 50)), (200, 120)).size
          == (200, 120))

    # --- grayscale ---
    red_gray = to_grayscale_rgb(red_image)
    r, g, b = red_gray.getpixel((0, 0))
    check("grayscale copy has R = G = B and 3 channels",
          red_gray.mode == "RGB" and r == g == b)
    check("pure red becomes about 76 in grayscale (luminance)",
          74 <= r <= 78, f"(got {r})")

    print("-" * 62)
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


def _preview_file(path: str) -> int:
    """Show the effect of preprocessing and save two preview images."""
    from src.image_io import ImageValidationError, load_image_from_path

    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1

    config.ensure_directories()
    resized = limit_working_size(loaded.image)
    gray = to_grayscale_rgb(resized.image)

    working_path = config.OUTPUTS_DIR / "preview_working.png"
    gray_path = config.OUTPUTS_DIR / "preview_gray.png"
    resized.image.save(working_path)
    gray.save(gray_path)

    print(f" Loaded size       : {loaded.width} x {loaded.height} px")
    print(f" Working size      : {resized.image.width} x {resized.image.height} px "
          f"(scale {resized.scale:.2f}, limit {MAX_WORKING_SIDE})")
    print(f" Saved             : {working_path}")
    print(f" Saved             : {gray_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(_preview_file(sys.argv[1]))
    sys.exit(_run_self_test())