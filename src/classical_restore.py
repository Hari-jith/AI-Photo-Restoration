"""
classical_restore.py

Classical (non-AI) photo cleanup with OpenCV:

  1. Denoising     - Non-Local Means removes grain while keeping edges.
  2. Scratch repair - find thin, long, isolated lines and fill them in
                      from the surrounding pixels ("inpainting").

    from src.classical_restore import restore_classical

    result = restore_classical(image, noise_sigma=report.noise_sigma)
    result.image            # cleaned Pillow RGB image
    result.scratch_mask     # white = pixels that were repaired
    print(result.notes)     # what was done, in plain English

HONEST LIMITS
  * Repairs THIN scratches and light grain only. Large torn or missing
    areas cannot be rebuilt by classical methods.
  * A scratch detector cannot truly tell a scratch from a real thin object
    (wire, hair, fence). On sharp, busy photos it wrongly flags roughly
    1-3% of pixels. Repairing those pixels changes the picture only
    slightly, but this is why scratch repair is optional in the app.
  * All thresholds are starting guesses; we tune them in Module 16.

Self-test:    python -m src.classical_restore
Try a file:   python -m src.classical_restore samples\\test_1.jpg
"""

import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from src.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings (starting guesses, tuned in Module 16)
# ---------------------------------------------------------------------------
# --- Denoising ---
# Below this estimated noise (see analysis.py) the photo is already clean,
# and denoising would only blur detail, so it is skipped.
SKIP_DENOISE_BELOW_SIGMA = 2.5
DEFAULT_DENOISE_STRENGTH = 5.0   # used when no noise estimate is supplied
MIN_DENOISE_STRENGTH = 3.0
MAX_DENOISE_STRENGTH = 10.0      # stronger than this smears real detail
NLM_TEMPLATE_WINDOW = 7          # OpenCV's recommended values
NLM_SEARCH_WINDOW = 21

# --- Scratch detection ---
# A pixel is a scratch candidate if it differs from the median-filtered
# picture by more than this many brightness levels (0-255).
SCRATCH_THRESHOLDS = {"low": 40, "medium": 28, "high": 20}
# The median filter erases lines up to about 3 pixels wide.
MEDIAN_KERNEL_SIZE = 7
# A scratch must be at least this long (pixels). Shorter marks are ignored.
MIN_SCRATCH_LENGTH = 25
# A scratch must be at most this wide (pixels), measured as area / length.
MAX_SCRATCH_WIDTH = 7.0
ANGLE_STEP_DEGREES = 10          # directions tested: 0, 10, 20 ... 170
# Candidates are thickened a little so a 1-pixel diagonal line still
# contains the test line. This also gives inpainting a safety margin.
CANDIDATE_DILATE_SIZE = 3
# Real scratches are ISOLATED: smooth picture around them. Texture (hair,
# foam, fabric) has many strong thin marks nearby. We measure the share of
# candidate pixels in a window around each line and reject lines in busy
# areas. (An isolated 2 px line alone gives about 0.10.)
NEIGHBOURHOOD_WINDOW = 21
MAX_NEIGHBOURHOOD_DENSITY = 0.15
# Safety net: if more than this fraction of the picture is flagged, the
# detector is probably reacting to texture, so repair is SKIPPED.
MAX_MASK_FRACTION = 0.05
# Radius (pixels) OpenCV looks around each damaged pixel when filling.
INPAINT_RADIUS = 3


@dataclass
class ClassicalRestoreResult:
    """Output of restore_classical."""

    image: Image.Image                 # cleaned RGB image
    scratch_mask: Image.Image          # mode "L": 255 = repaired pixel
    mask_fraction: float               # share of pixels repaired (0 if none)
    denoised: bool
    denoise_strength: float            # 0.0 if denoising was skipped
    scratches_repaired: bool
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "RGB")


def choose_denoise_strength(noise_sigma: float | None) -> float:
    """Pick the Non-Local-Means strength 'h' from an estimated noise level.

    Rule of thumb: h close to the noise level removes noise without
    erasing detail. Returns 0.0 when the image is clean enough to skip.
    """
    if noise_sigma is None:
        return DEFAULT_DENOISE_STRENGTH
    if noise_sigma < SKIP_DENOISE_BELOW_SIGMA:
        return 0.0
    return float(min(max(noise_sigma, MIN_DENOISE_STRENGTH),
                     MAX_DENOISE_STRENGTH))


def denoise_bgr(bgr: np.ndarray, strength: float) -> np.ndarray:
    """Non-Local-Means denoising of a BGR uint8 image."""
    return cv2.fastNlMeansDenoisingColored(
        bgr, None, strength, strength, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW)


def _line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """A square 0/1 kernel containing one 1-pixel line at the given angle."""
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = (length - 1) / 2.0
    radians = np.deg2rad(angle_degrees)
    dx = np.cos(radians) * center
    dy = -np.sin(radians) * center
    start = (int(round(center - dx)), int(round(center - dy)))
    end = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, start, end, 1, 1)
    return kernel


def detect_scratches(bgr: np.ndarray, threshold: int) -> np.ndarray:
    """Return a uint8 mask (255 = scratch pixel) of thin, long, isolated lines.

    Steps:
      1. 'Picture minus median-filtered picture' lights up only thin bright
         or dark structures (a median filter erases thin lines but keeps edges).
      2. For each direction, keep candidates that form a straight run at
         least MIN_SCRATCH_LENGTH long (morphological opening with a line).
      3. Keep only runs that are narrow (MAX_SCRATCH_WIDTH) and isolated
         (low candidate density around them).
    The shape tests are done one direction at a time so crossing scratches
    are judged separately instead of merging into one big blob.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(gray, MEDIAN_KERNEL_SIZE)
    difference = cv2.absdiff(gray, background)
    raw_candidates = (difference > threshold).astype(np.uint8)

    window = (NEIGHBOURHOOD_WINDOW, NEIGHBOURHOOD_WINDOW)
    density = cv2.boxFilter(raw_candidates.astype(np.float32), -1, window)

    kernel_size = (CANDIDATE_DILATE_SIZE, CANDIDATE_DILATE_SIZE)
    candidates = cv2.dilate(raw_candidates * 255,
                            np.ones(kernel_size, np.uint8))

    scratch_mask = np.zeros_like(candidates)
    for angle in range(0, 180, ANGLE_STEP_DEGREES):
        kernel = _line_kernel(MIN_SCRATCH_LENGTH, angle)
        opened = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
        if not opened.any():
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            opened, connectivity=8)
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area < MIN_SCRATCH_LENGTH:
                continue
            member = labels[y:y + height, x:x + width] == index
            ys, xs = np.nonzero(member)
            points = np.column_stack([xs, ys]).astype(np.float32)
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(points)
            length = max(rect_w, rect_h)
            if length < MIN_SCRATCH_LENGTH or area / length > MAX_SCRATCH_WIDTH:
                continue                      # too short, or too fat
            if density[y:y + height, x:x + width][member].mean() \
                    > MAX_NEIGHBOURHOOD_DENSITY:
                continue                      # inside busy texture
            scratch_mask[y:y + height, x:x + width][member] = 255
    return scratch_mask


def inpaint_bgr(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill the masked pixels using the surrounding picture (Telea method)."""
    return cv2.inpaint(bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def restore_classical(
    image: Image.Image,
    noise_sigma: float | None = None,
    denoise: bool = True,
    repair_scratches: bool = True,
    scratch_sensitivity: str = "medium",
) -> ClassicalRestoreResult:
    """Denoise and repair thin scratches.

    Args:
        image: RGB Pillow image (from image_io / preprocessing).
        noise_sigma: noise estimate from analysis.py. If None, a moderate
            default strength is used.
        denoise: turn denoising on or off.
        repair_scratches: turn scratch repair on or off.
        scratch_sensitivity: "low", "medium" or "high". Higher finds fainter
            scratches but flags more real thin details by mistake.
    """
    if scratch_sensitivity not in SCRATCH_THRESHOLDS:
        raise ValueError(
            f"scratch_sensitivity must be one of "
            f"{list(SCRATCH_THRESHOLDS)}, got '{scratch_sensitivity}'.")

    bgr = _to_bgr(image)
    notes = []
    denoise_strength = 0.0

    # --- 1. Denoise first, so grain is not mistaken for scratches ---
    if denoise:
        denoise_strength = choose_denoise_strength(noise_sigma)
        if denoise_strength > 0:
            bgr = denoise_bgr(bgr, denoise_strength)
            notes.append(f"Removed grain (strength {denoise_strength:.1f}).")
        else:
            notes.append("Photo is already clean; denoising skipped.")

    # --- 2. Scratches ---
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    repaired = False
    mask_fraction = 0.0
    if repair_scratches:
        found = detect_scratches(bgr, SCRATCH_THRESHOLDS[scratch_sensitivity])
        found_fraction = float(np.count_nonzero(found)) / found.size
        if found_fraction == 0.0:
            notes.append("No scratches found.")
        elif found_fraction > MAX_MASK_FRACTION:
            notes.append(
                f"Too much of the photo ({found_fraction:.0%}) looked like "
                "scratches (probably texture), so scratch repair was skipped "
                "to avoid smearing the picture.")
        else:
            bgr = inpaint_bgr(bgr, found)
            mask = found
            mask_fraction = found_fraction
            repaired = True
            notes.append(f"Repaired scratches covering "
                         f"{found_fraction:.2%} of the photo.")

    logger.info("Classical restore: %s", " ".join(notes))
    return ClassicalRestoreResult(
        image=_to_pil(bgr),
        scratch_mask=Image.fromarray(mask, "L"),
        mask_fraction=mask_fraction,
        denoised=denoise_strength > 0,
        denoise_strength=denoise_strength,
        scratches_repaired=repaired,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Self-test and file trial
# ---------------------------------------------------------------------------
def _make_test_picture(size: int = 400) -> np.ndarray:
    """A smooth BGR picture with a few soft shapes (a stand-in for a photo)."""
    y, x = np.mgrid[0:size, 0:size]
    picture = np.stack([60 + x * 0.3, 90 + y * 0.25, 140 - x * 0.1],
                       axis=-1).astype(np.uint8)
    cv2.rectangle(picture, (60, 60), (160, 180), (200, 180, 90), -1)
    cv2.circle(picture, (280, 250), 60, (40, 60, 200), -1)
    return cv2.GaussianBlur(picture, (0, 0), 1.0)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity of two images in dB (higher = more alike)."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))


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
    clean = _make_test_picture()

    # Scratched copy: vertical white, diagonal dark, near-horizontal 1 px
    # white, and a slightly curved white one. The first three cross.
    scratched = clean.copy()
    truth = np.zeros(clean.shape[:2], np.uint8)
    lines = [((200, 40), (200, 360), 2, 255), ((30, 350), (370, 120), 2, 10),
             ((20, 20), (380, 40), 1, 255)]
    for start, end, thickness, value in lines:
        cv2.line(scratched, start, end, (value,) * 3, thickness)
        cv2.line(truth, start, end, 255, thickness)
    curve = np.array([[330, 60], [335, 150], [345, 230], [340, 320]], np.int32)
    cv2.polylines(scratched, [curve], False, (255, 255, 255), 2)
    cv2.polylines(truth, [curve], False, 255, 2)
    scratch_pixels = truth > 0

    # --- detection ---
    mask = detect_scratches(scratched, SCRATCH_THRESHOLDS["medium"])
    coverage = np.count_nonzero((mask > 0) & scratch_pixels) / np.count_nonzero(
        scratch_pixels)
    check("finds at least 90% of scratch pixels (incl. crossing + curved)",
          coverage >= 0.90, f"(found {coverage:.0%})")

    clean_fraction = np.count_nonzero(
        detect_scratches(clean, SCRATCH_THRESHOLDS["medium"])) / clean[:, :, 0].size
    check("scratch-free picture: less than 0.5% flagged",
          clean_fraction < 0.005, f"(got {clean_fraction:.2%})")

    short = clean.copy()
    cv2.line(short, (100, 300), (110, 300), (255, 255, 255), 2)   # 10 px mark
    check("a short 10 px mark is ignored",
          not detect_scratches(short, SCRATCH_THRESHOLDS["medium"]).any())

    texture = rng.integers(0, 256, clean.shape, dtype=np.uint8)
    check("random texture is not mistaken for scratches",
          not detect_scratches(texture, SCRATCH_THRESHOLDS["medium"]).any())

    # --- repair ---
    repaired = inpaint_bgr(scratched, mask)
    error_before = np.abs(scratched.astype(int) - clean.astype(int))[
        scratch_pixels].mean()
    error_after = np.abs(repaired.astype(int) - clean.astype(int))[
        scratch_pixels].mean()
    check("repair removes most of the error on scratch pixels",
          error_after < 0.25 * error_before,
          f"(error {error_before:.0f} -> {error_after:.0f})")

    scratched_rgb = Image.fromarray(
        cv2.cvtColor(scratched, cv2.COLOR_BGR2RGB), "RGB")
    result = restore_classical(scratched_rgb, denoise=False)
    check("restore_classical repairs and reports it",
          result.scratches_repaired and 0 < result.mask_fraction < 0.05)
    check("output size, mode and mask size are correct",
          result.image.size == scratched_rgb.size and result.image.mode == "RGB"
          and result.scratch_mask.mode == "L"
          and result.scratch_mask.size == scratched_rgb.size)

    # --- denoise ---
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)
    denoised = denoise_bgr(noisy, 10.0)
    gain = _psnr(denoised, clean) - _psnr(noisy, clean)
    check("denoising improves similarity to the clean picture by 3+ dB",
          gain >= 3.0, f"(gain {gain:.1f} dB)")

    clean_rgb = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB), "RGB")
    skipped = restore_classical(clean_rgb, noise_sigma=1.0,
                                repair_scratches=False)
    check("clean photo: denoising skipped, pixels unchanged",
          not skipped.denoised
          and np.array_equal(np.array(skipped.image), np.array(clean_rgb)))

    # --- safety and input handling ---
    global MAX_MASK_FRACTION
    saved_cap = MAX_MASK_FRACTION
    MAX_MASK_FRACTION = 0.0001           # pretend the cap is tiny
    try:
        capped = restore_classical(scratched_rgb, denoise=False)
    finally:
        MAX_MASK_FRACTION = saved_cap
    check("safety cap: repair skipped and picture left unchanged",
          not capped.scratches_repaired
          and np.array_equal(np.array(capped.image), np.array(scratched_rgb)))

    try:
        restore_classical(clean_rgb, scratch_sensitivity="extreme")
        check("invalid sensitivity is rejected", False)
    except ValueError:
        check("invalid sensitivity is rejected", True)

    try:
        tiny = restore_classical(Image.new("RGB", (20, 20), "gray"))
        check("tiny image does not crash", tiny.image.size == (20, 20))
    except Exception as error:
        check("tiny image does not crash", False, str(error))

    print("-" * 62)
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


def _try_file(path: str) -> int:
    """Run the cleanup on a real file and save the result and the mask."""
    from src import config
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
    result = restore_classical(working, noise_sigma=report.noise_sigma)

    config.ensure_directories()
    result_path = config.OUTPUTS_DIR / "classical_result.png"
    mask_path = config.OUTPUTS_DIR / "classical_mask.png"
    working.save(config.OUTPUTS_DIR / "classical_before.png")
    result.image.save(result_path)
    result.scratch_mask.save(mask_path)

    print(f" Estimated noise : {report.noise_sigma:.1f}")
    for note in result.notes:
        print(f" - {note}")
    print(f" Saved: {config.OUTPUTS_DIR / 'classical_before.png'}")
    print(f" Saved: {result_path}")
    print(f" Saved: {mask_path}  (white = repaired pixels)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(_try_file(sys.argv[1]))
    sys.exit(_run_self_test())