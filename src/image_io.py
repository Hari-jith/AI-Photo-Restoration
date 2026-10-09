"""
image_io.py

Safe loading and validation of images.

Main entry points:

    from src.image_io import load_image_from_path, load_image_from_bytes

    loaded = load_image_from_path("samples/old_photo.jpg")
    loaded.image            # a Pillow image, always 8-bit RGB
    loaded.original_mode    # e.g. "L" (grayscale) or "RGB"

Every problem raises an ImageValidationError (or a subclass) whose message
is written for a beginner, so the web interface can show it directly.

Self-test:      python -m src.image_io
Inspect a file: python -m src.image_io samples/old_photo.jpg
"""

import io
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src import config
from src.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors. All inherit from ImageValidationError so callers can catch them
# with one "except ImageValidationError".
# ---------------------------------------------------------------------------
class ImageValidationError(Exception):
    """Base class: the image cannot be accepted. The message is user-friendly."""


class UnsupportedFormatError(ImageValidationError):
    """File type is not one we accept."""


class FileTooLargeError(ImageValidationError):
    """File size on disk is over the limit."""


class CorruptImageError(ImageValidationError):
    """File is empty, damaged, or not really an image."""


class ImageTooSmallError(ImageValidationError):
    """Image has too few pixels for the AI models."""


class ImageTooLargeError(ImageValidationError):
    """Image has too many pixels and could exhaust memory."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class LoadedImage:
    """A validated image plus information about what the original file was."""

    image: Image.Image          # always 8-bit RGB
    original_format: str        # e.g. "JPEG", "PNG"
    original_mode: str          # e.g. "L", "RGB", "RGBA", "P", "CMYK"
    original_size: tuple        # (width, height) of the file as stored
    file_size_bytes: int
    had_transparency: bool      # True if transparency was flattened onto white

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------
def validate_extension(filename: str) -> str:
    """Return the lowercase extension (like '.jpg') or raise an error."""
    extension = Path(filename).suffix.lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Files of type '{extension or 'unknown'}' are not supported. "
            f"Please upload one of: {allowed}."
        )
    return extension


def make_safe_filename(original_filename: str) -> str:
    """Create a random, safe filename that keeps only the original extension.

    The uploaded name is never used on disk, so names like '..\\..\\x.jpg'
    or names with strange characters cannot cause harm.
    """
    extension = validate_extension(original_filename)
    return f"{uuid.uuid4().hex}{extension}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_image_from_path(path) -> LoadedImage:
    """Load and validate an image file from disk."""
    file_path = Path(path)
    if not file_path.is_file():
        raise CorruptImageError(
            "The image file could not be found. Check the path and try again."
        )
    # Check size BEFORE reading, so a huge file is never loaded into memory.
    size_on_disk = file_path.stat().st_size
    _check_file_size(size_on_disk)
    return load_image_from_bytes(file_path.read_bytes(), file_path.name)


def load_image_from_bytes(data: bytes, filename: str = "") -> LoadedImage:
    """Load and validate an image from raw bytes (e.g. a web upload).

    Args:
        data: the complete file content.
        filename: original name, used only to check the extension.
    """
    _check_file_size(len(data))
    if filename:
        validate_extension(filename)

    # Image.open reads only the file header, so we can inspect the size
    # BEFORE the expensive full decode.
    try:
        with warnings.catch_warnings():
            # Treat Pillow's "decompression bomb" warning as an error.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            raw_image = Image.open(io.BytesIO(data))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageTooLargeError(_too_large_message()) from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise CorruptImageError(
            "This file could not be read as an image. It may be damaged, "
            "or it may not be a real image file."
        ) from None

    with raw_image:
        _check_format_and_dimensions(raw_image)
        original_format = raw_image.format
        original_mode = raw_image.mode
        original_size = raw_image.size

        # Full decode. Truncated or damaged files usually fail here.
        try:
            raw_image.load()
        except (OSError, SyntaxError, ValueError):
            raise CorruptImageError(
                "This image appears to be damaged or incomplete and could "
                "not be fully read. Try re-saving or re-scanning it."
            ) from None

        rgb_image, had_transparency = _normalize_to_rgb(raw_image)

    logger.info(
        "Loaded image: format=%s mode=%s size=%dx%d bytes=%d",
        original_format, original_mode, original_size[0], original_size[1],
        len(data),
    )
    return LoadedImage(
        image=rgb_image,
        original_format=original_format,
        original_mode=original_mode,
        original_size=original_size,
        file_size_bytes=len(data),
        had_transparency=had_transparency,
    )


# ---------------------------------------------------------------------------
# Internal checks
# ---------------------------------------------------------------------------
def _too_large_message() -> str:
    return (
        f"This image has too many pixels (limit: {config.MAX_IMAGE_PIXELS:,}, "
        "roughly 5000 x 5000). Please resize it and try again."
    )


def _check_file_size(size_bytes: int) -> None:
    if size_bytes == 0:
        raise CorruptImageError("The file is empty.")
    if size_bytes > config.MAX_UPLOAD_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise FileTooLargeError(
            f"The file is {size_mb:.1f} MB, but the maximum is "
            f"{config.MAX_UPLOAD_MB} MB. Please upload a smaller file."
        )


def _check_format_and_dimensions(image: Image.Image) -> None:
    """Check the REAL format (from file content) and the pixel dimensions."""
    if image.format not in config.ALLOWED_FORMATS:
        raise UnsupportedFormatError(
            f"The file content is '{image.format}', which is not supported. "
            f"Supported formats: {', '.join(sorted(config.ALLOWED_FORMATS))}."
        )

    width, height = image.size
    if min(width, height) < config.MIN_IMAGE_SIDE:
        raise ImageTooSmallError(
            f"The image is {width}x{height} pixels. Each side must be at "
            f"least {config.MIN_IMAGE_SIDE} pixels for restoration to work."
        )
    if width * height > config.MAX_IMAGE_PIXELS:
        raise ImageTooLargeError(_too_large_message())


def _normalize_to_rgb(image: Image.Image):
    """Convert any supported image to 8-bit RGB.

    Returns (rgb_image, had_transparency).
    """
    # Apply the camera's "rotate me" metadata so the image is upright.
    image = ImageOps.exif_transpose(image)

    has_transparency = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_transparency:
        # Flatten onto white. A plain convert("RGB") would make
        # transparent areas black.
        rgba = image.convert("RGBA")
        white_background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flattened = Image.alpha_composite(white_background, rgba)
        return flattened.convert("RGB"), True

    if image.mode.startswith("I"):
        # 16-bit (or 32-bit integer) grayscale, common in professional scans.
        # Pillow's own convert would clip values above 255, so scale first.
        values = np.asarray(image).astype(np.float32)
        if values.max() > 255:
            values = values / 257.0   # 65535 / 257 = 255
        eight_bit = np.clip(values, 0, 255).astype(np.uint8)
        return Image.fromarray(eight_bit).convert("RGB"), False

    # Covers RGB, L (grayscale), P (palette), CMYK, 1 (black and white).
    return image.convert("RGB"), False


# ---------------------------------------------------------------------------
# Self-test and file inspector
# ---------------------------------------------------------------------------
def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _run_self_test() -> int:
    """Check good and bad inputs. Returns 0 if all pass, 1 otherwise."""
    # (description, bytes, filename, expected error class or None)
    gradient = np.tile(np.arange(0, 256, dtype=np.uint8), (200, 1))
    transparent = Image.new("RGBA", (200, 200), (255, 0, 0, 0))   # fully clear

    cases = [
        ("valid RGB PNG", _png_bytes(Image.new("RGB", (200, 200), "tan")),
         "a.png", None),
        ("valid grayscale PNG", _png_bytes(Image.fromarray(gradient, "L")),
         "b.png", None),
        ("RGBA with transparency", _png_bytes(transparent), "c.png", None),
        ("16-bit grayscale PNG",
         _png_bytes(Image.fromarray(
             np.tile(np.arange(0, 65536, 256, dtype=np.uint16), (200, 1)))),
         "d.png", None),
        ("wrong extension (.exe)", b"MZ fake program", "virus.exe",
         UnsupportedFormatError),
        ("text file named .jpg", b"hello, I am not an image" * 50, "e.jpg",
         CorruptImageError),
        ("truncated PNG",
         _png_bytes(Image.new("RGB", (300, 300), "red"))[:100], "f.png",
         CorruptImageError),
        ("empty file", b"", "g.png", CorruptImageError),
        ("image too small", _png_bytes(Image.new("RGB", (10, 10))), "h.png",
         ImageTooSmallError),
        ("file too large", b"\x00" * (config.MAX_UPLOAD_BYTES + 1), "i.png",
         FileTooLargeError),
        ("too many pixels", _png_bytes(Image.new("L", (6000, 6000))), "j.png",
         ImageTooLargeError),
    ]

    failures = 0
    for description, data, filename, expected_error in cases:
        try:
            loaded = load_image_from_bytes(data, filename)
            if expected_error is None and loaded.image.mode == "RGB":
                outcome = f"accepted ({loaded.original_mode} -> RGB)"
                print(f" [PASS] {description}: {outcome}")
            else:
                failures += 1
                print(f" [FAIL] {description}: should have been rejected")
        except ImageValidationError as error:
            if expected_error is not None and isinstance(error, expected_error):
                print(f" [PASS] {description}: rejected -> {error}")
            else:
                failures += 1
                print(f" [FAIL] {description}: unexpected "
                      f"{type(error).__name__}: {error}")

    # Transparent pixels must become WHITE, not black.
    flattened = load_image_from_bytes(_png_bytes(transparent), "c.png").image
    if flattened.getpixel((5, 5)) == (255, 255, 255):
        print(" [PASS] transparent area became white")
    else:
        failures += 1
        print(" [FAIL] transparent area is not white")

    # Safe filenames must not contain the uploaded name.
    safe_name = make_safe_filename("..\\..\\evil name.JPG")
    if safe_name.endswith(".jpg") and "evil" not in safe_name \
            and "\\" not in safe_name:
        print(f" [PASS] safe filename generated: {safe_name}")
    else:
        failures += 1
        print(f" [FAIL] unsafe filename: {safe_name}")

    print("-" * 62)
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


def _inspect_file(path: str) -> int:
    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1
    print(f" File            : {path}")
    print(f" Original format : {loaded.original_format}")
    print(f" Original mode   : {loaded.original_mode}")
    print(f" Original size   : {loaded.original_size[0]} x "
          f"{loaded.original_size[1]} px")
    print(f" File size       : {loaded.file_size_bytes / 1024:.0f} KB")
    print(f" Had transparency: {loaded.had_transparency}")
    print(f" Loaded as       : RGB {loaded.width} x {loaded.height} px")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(_inspect_file(sys.argv[1]))
    sys.exit(_run_self_test())