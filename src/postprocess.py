"""
postprocess.py

Final touch-ups applied after restoration. Deliberately small: we only
include adjustments that measurements on real photos showed to be useful.

  1. CONTRAST RESTORATION (for faded photos only)
     Stretches the lightness so the darkest and brightest parts use more of
     the available range. Colour is left alone. In "auto" mode it does
     NOTHING to photos whose contrast is already fine (the same threshold as
     analysis.py), so it cannot make a good photo worse.
  2. FILM GRAIN (optional, off by default, cosmetic)
     AI upscalers and face restorers remove most of the original grain, which
     can make a photo look smooth and "airbrushed". Measured on one real photo,
     the estimated grain level fell from 4.8 (original) to 0.6-0.9 (after
     upscaling). This adds fine, repeatable grain back to lightness only.

    from src.postprocess import postprocess

    result = postprocess(image)                       # contrast "auto", no grain
    result = postprocess(image, grain=0.5)            # about the original's grain
    result.image, result.notes

Not included on purpose: extra sharpening (the upscaler and GFPGAN already
sharpen) and colour-cast removal (no evidence yet that it is needed).

Self-test:       python -m src.postprocess
Try a file:      python -m src.postprocess samples\\test_1.jpg --grain 0.5
Test on a good photo by FADING it first:   add  --simulate-fade
"""

import argparse
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from src import config
from src.logging_setup import get_logger

logger = get_logger(__name__)

# --- Settings ----------------------------------------------------------------
# Same definition and value as analysis.LOW_CONTRAST_RANGE (a self-test checks
# that they match): brightness spread between the 1st and 99th percentile.
LOW_CONTRAST_RANGE = 150
LOW_PERCENTILE, HIGH_PERCENTILE = 1.0, 99.0

# Lightness (Lab "L", 0 = black .. 100 = white) the stretched range is mapped
# to. Staying a little inside 0..100 avoids crushed blacks and burnt highlights.
TARGET_LOW_L, TARGET_HIGH_L = 3.0, 97.0
MAX_GAIN = 3.0               # never amplify contrast more than 3x (noise!)
MIN_SOURCE_RANGE_L = 5.0     # a picture flatter than this is left alone

# Grain: standard deviation (in L units, 0..100) at strength 1.0. The original
# photo's grain was about 2 L units, so strength 0.5 roughly restores it.
GRAIN_MAX_SIGMA_L = 4.0
GRAIN_SOFTNESS = 0.6         # slight blur so grain looks like film, not digital noise
CONTRAST_MODES = ("auto", "on", "off")


@dataclass
class PostprocessResult:
    image: Image.Image
    contrast_applied: bool
    grain_applied: bool
    contrast_range_before: float     # 0..255 scale, p99 - p1 of brightness
    contrast_range_after: float
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_lab(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def _from_lab(lab: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    # clip BEFORE converting to bytes, or out-of-range values wrap around
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8), "RGB")


def contrast_range(image: Image.Image) -> float:
    """Brightness spread (p99 - p1, scale 0..255). Same measure as analysis.py."""
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    low, high = np.percentile(gray, [LOW_PERCENTILE, HIGH_PERCENTILE])
    return float(high - low)


# ---------------------------------------------------------------------------
# 1. Contrast
# ---------------------------------------------------------------------------
def stretch_contrast(image: Image.Image, strength: float = 1.0):
    """Stretch the lightness of an image. Returns (image, note or None).

    Colour channels are untouched. `strength` blends between the original
    (0.0, returned untouched) and the fully stretched picture (1.0). A picture
    that is almost flat is returned unchanged, and the gain is capped at
    MAX_GAIN. Note: converting to Lab and back can change a pixel value by 1
    (out of 255) on a fraction of a percent of values; invisible, but it is
    why we skip the conversion whenever there is nothing to do.
    """
    if strength <= 0.0:                 # nothing to do: do not touch the pixels at all
        return image.convert("RGB"), None
    lab = _to_lab(image)
    lightness = lab[:, :, 0]
    low, high = np.percentile(lightness, [LOW_PERCENTILE, HIGH_PERCENTILE])
    if high - low < MIN_SOURCE_RANGE_L:
        return image.convert("RGB"), "The picture is almost flat, so contrast was left unchanged."
    gain = min(MAX_GAIN, (TARGET_HIGH_L - TARGET_LOW_L) / (high - low))
    stretched = TARGET_LOW_L + (lightness - low) * gain
    lab[:, :, 0] = np.clip(lightness + strength * (stretched - lightness), 0.0, 100.0)
    note = f"Contrast stretched (gain {gain:.2f}x" + (", capped" if gain >= MAX_GAIN else "") + ")."
    return _from_lab(lab), note


# ---------------------------------------------------------------------------
# 2. Grain
# ---------------------------------------------------------------------------
def add_grain(image: Image.Image, strength: float, seed: int = 0) -> Image.Image:
    """Add fine, repeatable film grain to the lightness only.

    The same seed always gives the same grain (so results can be compared).
    Strength 1.0 gives a standard deviation of GRAIN_MAX_SIGMA_L lightness units.
    """
    if strength <= 0.0:
        return image.convert("RGB")
    lab = _to_lab(image)
    noise = np.random.default_rng(seed).normal(0.0, 1.0, lab.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), GRAIN_SOFTNESS)
    noise = noise / max(float(noise.std()), 1e-6)            # back to exactly unit spread
    lab[:, :, 0] = np.clip(lab[:, :, 0] + noise * GRAIN_MAX_SIGMA_L * strength, 0.0, 100.0)
    return _from_lab(lab)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def postprocess(image: Image.Image, contrast: str = "auto", contrast_strength: float = 1.0,
                grain: float = 0.0, seed: int = 0) -> PostprocessResult:
    """Apply the final touch-ups.

    Args:
        contrast: "auto" (only if the photo is low-contrast), "on" (always),
            or "off".
        contrast_strength: 0.0 to 1.0, how much of the stretch to apply.
        grain: 0.0 (none) to 1.0. About 0.5 matches typical old-photo grain.
        seed: makes the grain repeatable.
    """
    if contrast not in CONTRAST_MODES:
        raise ValueError(f"contrast must be one of {CONTRAST_MODES}.")
    if not 0.0 <= contrast_strength <= 1.0:
        raise ValueError("contrast_strength must be between 0.0 and 1.0.")
    if not 0.0 <= grain <= 1.0:
        raise ValueError("grain must be between 0.0 and 1.0.")

    current = image.convert("RGB")
    before = contrast_range(current)
    notes = []
    contrast_applied = False

    if contrast == "off":
        pass
    elif contrast == "auto" and before >= LOW_CONTRAST_RANGE:
        notes.append(f"Contrast is already good ({before:.0f}/255), so it was left unchanged.")
    else:
        current, note = stretch_contrast(current, contrast_strength)
        contrast_applied = note is not None and note.startswith("Contrast stretched")
        if note:
            notes.append(note)

    grain_applied = grain > 0.0
    if grain_applied:
        current = add_grain(current, grain, seed)
        notes.append(f"Film grain added (strength {grain:g}).")

    after = contrast_range(current)
    logger.info("Postprocess: %s", " ".join(notes) if notes else "nothing to do")
    return PostprocessResult(current, contrast_applied, grain_applied, before, after, notes)


def simulate_fade(image: Image.Image, factor: float = 0.55, offset: float = 60.0) -> Image.Image:
    """Make a photo look faded (compressed levels, lifted blacks). For demos/tests."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) * factor + offset
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# Self-test
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

    def raises(expected, function) -> bool:
        try:
            function()
        except expected:
            return True
        except Exception:
            return False
        return False

    def chroma(image):
        lab = _to_lab(image)
        return float(np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2).mean())

    rng = np.random.default_rng(0)
    # a colourful, full-range test picture (smooth, so noise tests are clean)
    y, x = np.mgrid[0:200, 0:300].astype(np.float32)
    colourful = Image.fromarray(np.stack(
        [x / 300 * 255, y / 200 * 255, (1 - x / 300) * 255], -1).astype(np.uint8), "RGB")
    faded = simulate_fade(colourful)

    check("consistent with analysis.py: same low-contrast threshold",
          _analysis_threshold() == LOW_CONTRAST_RANGE)

    # ---- contrast ----
    check("the faded test picture really is low contrast",
          contrast_range(faded) < LOW_CONTRAST_RANGE, f"({contrast_range(faded):.0f})")
    result = postprocess(faded)
    check("auto: a faded picture is stretched to a much wider range",
          result.contrast_applied and result.contrast_range_after > 190
          and result.contrast_range_after > result.contrast_range_before + 60,
          f"({result.contrast_range_before:.0f} -> {result.contrast_range_after:.0f})")
    check("stretching keeps the colour (mean chroma within 15%)",
          abs(chroma(result.image) - chroma(faded)) / chroma(faded) < 0.15,
          f"({chroma(faded):.1f} -> {chroma(result.image):.1f})")

    good = postprocess(colourful)
    check("auto: a picture with good contrast is left completely unchanged",
          not good.contrast_applied
          and np.array_equal(np.array(good.image), np.array(colourful))
          and "already good" in good.notes[0])
    check("contrast 'off' changes nothing even on a faded picture",
          np.array_equal(np.array(postprocess(faded, contrast="off").image), np.array(faded)))
    check("contrast 'on' applies even when the picture is already fine",
          postprocess(colourful, contrast="on").contrast_applied is True)

    half = postprocess(faded, contrast_strength=0.5)
    check("strength 0.5 lands between the original and the full stretch",
          result.contrast_range_before < half.contrast_range_after < result.contrast_range_after)
    check("strength 0 leaves the picture unchanged",
          np.array_equal(np.array(postprocess(faded, contrast_strength=0.0).image),
                         np.array(faded)))

    dull = Image.fromarray(np.tile(np.linspace(100, 120, 300), (100, 1)).astype(np.uint8), "L")
    dull_result = postprocess(dull)
    check("a very dull picture: gain is capped (no runaway amplification)",
          dull_result.contrast_range_after <= 3.2 * dull_result.contrast_range_before + 2,
          f"({dull_result.contrast_range_before:.0f} -> {dull_result.contrast_range_after:.0f})")
    flat = postprocess(Image.new("RGB", (80, 80), (120, 120, 120)), contrast="on")
    check("a perfectly flat picture is left unchanged (no divide-by-zero)",
          not flat.contrast_applied and "flat" in flat.notes[0])

    # ---- grain ----
    mid_grey = Image.new("RGB", (200, 200), (128, 128, 128))
    base_l = _to_lab(mid_grey)[:, :, 0]
    grained = postprocess(mid_grey, contrast="off", grain=1.0, seed=1).image
    spread = float((_to_lab(grained)[:, :, 0] - base_l).std())
    check("grain 1.0 adds noise with the documented strength (4 L units, within 15%)",
          abs(spread - GRAIN_MAX_SIGMA_L) / GRAIN_MAX_SIGMA_L < 0.15, f"(got {spread:.2f})")
    g = np.array(grained).astype(int)
    check("grain affects lightness only: a gray picture stays gray",
          np.abs(g[..., 0] - g[..., 2]).max() <= 2)
    check("average brightness is preserved",
          abs(float(np.array(grained).mean()) - 128.0) < 1.5)
    check("same seed gives identical grain, a different seed gives different grain",
          np.array_equal(np.array(add_grain(mid_grey, 0.5, 7)), np.array(add_grain(mid_grey, 0.5, 7)))
          and not np.array_equal(np.array(add_grain(mid_grey, 0.5, 7)),
                                 np.array(add_grain(mid_grey, 0.5, 8))))
    check("grain 0 returns the picture unchanged",
          np.array_equal(np.array(postprocess(colourful, grain=0.0).image), np.array(colourful)))

    # ---- inputs ----
    check("grayscale and RGBA inputs are accepted, output is RGB of the same size",
          postprocess(Image.new("L", (60, 40), 90), grain=0.3).image.size == (60, 40)
          and postprocess(Image.new("RGBA", (60, 40), (90, 90, 90, 255))).image.mode == "RGB")
    check("invalid arguments are rejected",
          raises(ValueError, lambda: postprocess(colourful, contrast="maybe"))
          and raises(ValueError, lambda: postprocess(colourful, grain=1.5))
          and raises(ValueError, lambda: postprocess(colourful, contrast_strength=-0.1)))
    extreme = Image.fromarray(np.where(rng.random((50, 50, 3)) > 0.5, 255, 0).astype(np.uint8), "RGB")
    check("extreme black/white input does not wrap around or crash",
          postprocess(extreme, contrast="on", grain=1.0).image.size == (50, 50))

    print("-" * 62)
    print(" RESULT:", "all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


def _analysis_threshold() -> float:
    from src.analysis import LOW_CONTRAST_RANGE as analysis_value
    return analysis_value


def _process_file(path: str, contrast: str, grain: float, simulate: bool) -> int:
    from src.image_io import ImageValidationError, load_image_from_path
    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1
    source = simulate_fade(loaded.image) if simulate else loaded.image
    if simulate:
        print(" (--simulate-fade: the photo was artificially faded first)")
    result = postprocess(source, contrast=contrast, grain=grain)
    for note in result.notes:
        print(f" - {note}")
    print(f" Contrast range: {result.contrast_range_before:.0f} -> "
          f"{result.contrast_range_after:.0f} (of 255; below {LOW_CONTRAST_RANGE} counts as faded)")
    config.ensure_directories()
    out_path = config.OUTPUTS_DIR / "postprocess_result.png"
    cmp_path = config.OUTPUTS_DIR / "postprocess_comparison.png"
    result.image.save(out_path)
    sheet = Image.new("RGB", (source.width * 2, source.height))
    sheet.paste(source.convert("RGB"), (0, 0))
    sheet.paste(result.image, (source.width, 0))
    sheet.save(cmp_path)
    print(f" Saved: {out_path}\n Saved: {cmp_path}  (before | after)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-processing (contrast, grain)")
    parser.add_argument("image", nargs="?", help="image to process (omit to run the self-test)")
    parser.add_argument("--contrast", default="auto", choices=CONTRAST_MODES)
    parser.add_argument("--grain", type=float, default=0.0, help="0.0 to 1.0 (about 0.5 matches old-photo grain)")
    parser.add_argument("--simulate-fade", action="store_true",
                        help="fade the photo first, to see the contrast fix at work")
    arguments = parser.parse_args()
    if arguments.image:
        sys.exit(_process_file(arguments.image, arguments.contrast, arguments.grain,
                               arguments.simulate_fade))
    sys.exit(_run_self_test())