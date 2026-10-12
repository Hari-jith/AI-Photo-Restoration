"""
analysis.py

Measures basic properties of a photo so the pipeline can decide what to do.

    from src.analysis import analyze_image
    report = analyze_image(loaded.image)      # a Pillow RGB image
    print(report.summary())
    report.needs_colorization                 # True / False

All thresholds are starting guesses (see the constants below). They are
hints for the pipeline, not exact truths. We tune them in Module 16.

Self-test:       python -m src.analysis
Analyze a file:  python -m src.analysis samples/test1.jpg
"""

import math
import sys
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from PIL import Image

from src.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Settings (starting guesses, tuned later)
# ---------------------------------------------------------------------------
# Large images are shrunk (never enlarged) to this longest side before
# measuring. This keeps analysis fast. Note: shrinking slightly reduces the
# measured noise and changes sharpness, so numbers are only comparable
# between images analyzed the same way.
ANALYSIS_MAX_SIDE = 800

# A pixel counts as colourless if its chroma is below this.
GRAYSCALE_MEAN_CHROMA = 3.0      # average chroma of the whole image
GRAYSCALE_P99_CHROMA = 10.0      # 99th-percentile chroma (ignores a few specks)

# For tint detection: only pixels with at least this much chroma are used
# to judge the hue, since the hue of a nearly gray pixel is meaningless.
TINT_MIN_PIXEL_CHROMA = 1.5
TINT_MIN_PIXEL_FRACTION = 0.05   # need at least 5% of pixels to judge
TINT_MIN_CONCENTRATION = 0.90    # 1.0 = every pixel has the same hue
TINT_MAX_MEAN_CHROMA = 30.0      # a strongly colourful image is not "tinted"

FADED_COLOR_MEAN_CHROMA = 12.0   # a colour image this dull is probably faded
LOW_CONTRAST_RANGE = 150         # p99 - p1 brightness below this = low contrast
NOISY_SIGMA = 7.0                # estimated noise (on a 0-255 scale)
BLURRY_LAPLACIAN_VARIANCE = 100.0

# Settings for the Haar face detector (a crude hint, see module docstring).
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (40, 40)         # pixels, at analysis size


@dataclass
class ImageAnalysis:
    """Everything we measured about one image."""

    width: int
    height: int
    color_kind: str              # "grayscale", "tinted_monochrome" or "color"
    mean_chroma: float
    hue_concentration: float     # 0..1, only meaningful if enough colour pixels
    brightness: float            # mean brightness, 0..255
    contrast_range: float        # p99 - p1 of brightness
    sharpness: float             # Laplacian variance (higher = sharper)
    noise_sigma: float           # estimated noise strength
    face_count_hint: int         # crude Haar detector, may miss or false-alarm

    # Hints derived from the numbers above
    needs_colorization: bool
    is_low_contrast: bool
    is_noisy: bool
    is_blurry: bool
    likely_has_faces: bool

    def to_dict(self) -> dict:
        """Plain dictionary, handy for the web API later."""
        return asdict(self)

    def summary(self) -> str:
        """A readable multi-line report."""
        yes_no = lambda flag: "yes" if flag else "no"
        return "\n".join([
            f" Size              : {self.width} x {self.height} px",
            f" Colour type       : {self.color_kind} "
            f"(mean chroma {self.mean_chroma:.1f}, "
            f"hue concentration {self.hue_concentration:.2f})",
            f" Brightness        : {self.brightness:.0f} / 255",
            f" Contrast range    : {self.contrast_range:.0f} / 255",
            f" Sharpness         : {self.sharpness:.0f}",
            f" Estimated noise   : {self.noise_sigma:.1f}",
            f" Faces (hint)      : {self.face_count_hint}",
            " ---- hints for the pipeline ----",
            f" Needs colourizing : {yes_no(self.needs_colorization)}",
            f" Low contrast      : {yes_no(self.is_low_contrast)}",
            f" Noisy             : {yes_no(self.is_noisy)}",
            f" Blurry            : {yes_no(self.is_blurry)}",
            f" Probably faces    : {yes_no(self.likely_has_faces)}",
        ])


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def _downscale(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so the longest side is at most max_side. Never enlarges."""
    height, width = rgb.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return rgb
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def _measure_color(rgb: np.ndarray):
    """Return (color_kind, mean_chroma, hue_concentration)."""
    # OpenCV stores 8-bit Lab with a and b shifted by +128 (neutral = 128).
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    a_channel = lab[:, :, 1].astype(np.float32) - 128.0
    b_channel = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a_channel ** 2 + b_channel ** 2)

    mean_chroma = float(chroma.mean())
    p99_chroma = float(np.percentile(chroma, 99))

    # Hue concentration: average the unit vectors that point in each
    # pixel's colour direction. If they all agree, the length is near 1;
    # if the directions are spread out, the vectors cancel and it is near 0.
    colorful = chroma > TINT_MIN_PIXEL_CHROMA
    hue_concentration = 0.0
    if colorful.mean() >= TINT_MIN_PIXEL_FRACTION:
        angles = np.arctan2(b_channel[colorful], a_channel[colorful])
        hue_concentration = float(math.hypot(
            np.cos(angles).mean(), np.sin(angles).mean()))

    if (mean_chroma <= GRAYSCALE_MEAN_CHROMA
            and p99_chroma <= GRAYSCALE_P99_CHROMA):
        kind = "grayscale"
    elif (hue_concentration >= TINT_MIN_CONCENTRATION
          and mean_chroma <= TINT_MAX_MEAN_CHROMA):
        kind = "tinted_monochrome"
    else:
        kind = "color"
    return kind, mean_chroma, hue_concentration


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """Estimate noise strength with the Immerkaer (1996) method.

    The kernel below responds to noise but cancels out smooth areas and
    straight edges. Real texture (grass, hair) also responds, so on very
    textured photos this overestimates noise.
    """
    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    response = response[1:-1, 1:-1]            # drop unreliable border pixels
    return float(math.sqrt(math.pi / 2.0) * np.abs(response).mean() / 6.0)


_face_detector = None


def _count_faces(gray: np.ndarray) -> int:
    """Count faces with OpenCV's built-in Haar detector (a crude hint)."""
    global _face_detector
    try:
        if _face_detector is None:
            cascade_path = (cv2.data.haarcascades
                            + "haarcascade_frontalface_default.xml")
            _face_detector = cv2.CascadeClassifier(cascade_path)
        if _face_detector.empty():
            logger.warning("Face detector data file not found; skipping.")
            return 0
        faces = _face_detector.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE,
        )
        return len(faces)
    except Exception as error:   # a hint must never crash the pipeline
        logger.warning("Face hint failed: %s", error)
        return 0


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def analyze_image(image: Image.Image) -> ImageAnalysis:
    """Measure an image. Expects an RGB Pillow image (from image_io)."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    rgb = _downscale(np.asarray(image), ANALYSIS_MAX_SIDE)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    color_kind, mean_chroma, hue_concentration = _measure_color(rgb)
    p1, p99 = np.percentile(gray, [1, 99])
    contrast_range = float(p99 - p1)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise_sigma = _estimate_noise_sigma(gray)
    face_count = _count_faces(gray)

    report = ImageAnalysis(
        width=image.width,
        height=image.height,
        color_kind=color_kind,
        mean_chroma=mean_chroma,
        hue_concentration=hue_concentration,
        brightness=float(gray.mean()),
        contrast_range=contrast_range,
        sharpness=sharpness,
        noise_sigma=noise_sigma,
        face_count_hint=face_count,
        needs_colorization=color_kind in ("grayscale", "tinted_monochrome"),
        is_low_contrast=contrast_range < LOW_CONTRAST_RANGE,
        is_noisy=noise_sigma > NOISY_SIGMA,
        is_blurry=sharpness < BLURRY_LAPLACIAN_VARIANCE,
        likely_has_faces=face_count > 0,
    )
    logger.info("Analysis: kind=%s noise=%.1f sharpness=%.0f contrast=%.0f "
                "faces=%d", color_kind, noise_sigma, sharpness,
                contrast_range, face_count)
    return report


# ---------------------------------------------------------------------------
# Self-test and file analyzer
# ---------------------------------------------------------------------------
def _run_self_test() -> int:
    """Check the analyzer on synthetic images with known properties."""
    rng = np.random.default_rng(seed=0)    # fixed seed = repeatable test
    size = 300
    ramp = np.tile(np.linspace(0, 255, size), (size, 1))        # left-to-right
    vertical = np.tile(np.linspace(0, 255, size)[:, None], (1, size))

    def to_image(array) -> Image.Image:
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")

    gray = to_image(np.stack([ramp] * 3, axis=-1))
    sepia = to_image(np.stack([ramp, ramp * 0.85, ramp * 0.65], axis=-1))
    colorful = to_image(np.stack([ramp, vertical, 255 - ramp], axis=-1))

    shared_noise = rng.normal(0, 15, (size, size))     # same noise on R, G, B
    noisy_gray = to_image(np.stack([ramp + shared_noise] * 3, axis=-1))

    # Tiny colour noise like JPEG compression adds to gray scans.
    jpeg_like_gray = to_image(
        np.stack([ramp] * 3, axis=-1) + rng.normal(0, 1, (size, size, 3)))

    flat_gray = to_image(np.stack([100 + ramp * 0.16] * 3, axis=-1))  # 100..141

    # (description, image, check function)
    cases = [
        ("pure grayscale", gray,
         lambda r: r.color_kind == "grayscale" and r.needs_colorization),
        ("gray with tiny colour noise", jpeg_like_gray,
         lambda r: r.color_kind == "grayscale"),
        ("sepia tint", sepia,
         lambda r: r.color_kind == "tinted_monochrome"
         and r.needs_colorization),
        ("colourful image", colorful,
         lambda r: r.color_kind == "color" and not r.needs_colorization),
        ("noisy image flagged noisy", noisy_gray, lambda r: r.is_noisy),
        ("clean image not noisy", gray, lambda r: not r.is_noisy),
        ("low-contrast image", flat_gray, lambda r: r.is_low_contrast),
        ("full-range image not low contrast", gray,
         lambda r: not r.is_low_contrast),
        ("noise estimate near true value (15)", noisy_gray,
         lambda r: 11 <= r.noise_sigma <= 17),
    ]

    failures = 0
    for description, image, check in cases:
        report = analyze_image(image)
        if check(report):
            print(f" [PASS] {description}")
        else:
            failures += 1
            print(f" [FAIL] {description}")
            print(f"        kind={report.color_kind} "
                  f"chroma={report.mean_chroma:.1f} "
                  f"hue_conc={report.hue_concentration:.2f} "
                  f"noise={report.noise_sigma:.1f} "
                  f"range={report.contrast_range:.0f}")

    print("-" * 62)
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


def _analyze_file(path: str) -> int:
    from src.image_io import ImageValidationError, load_image_from_path
    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1
    print(f" File              : {path}")
    print(f" Original mode     : {loaded.original_mode}")
    print(analyze_image(loaded.image).summary())
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(_analyze_file(sys.argv[1]))
    sys.exit(_run_self_test())