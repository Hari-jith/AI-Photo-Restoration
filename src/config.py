"""
config.py

Central settings for the AI Old Photo Restoration project.

Every folder path, size limit and option lives here, so the rest of the
code never needs hard-coded values. Other modules use it like this:

    from src.config import MODELS_DIR, MAX_UPLOAD_BYTES

A few values can be overridden with environment variables (for example
PHOTO_MAX_UPLOAD_MB). If the variable is not set, the default is used.
"""

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default.

    If the variable is missing or is not a valid whole number, the default
    is used instead of crashing.
    """
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# This file is at <project>/src/config.py, so the project root is two
# levels up. Using the file's own location means the project works no
# matter where on your computer the folder is placed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"      # downloaded model weights
OUTPUTS_DIR = PROJECT_ROOT / "outputs"    # restored images
UPLOADS_DIR = PROJECT_ROOT / "uploads"    # temporary copies of user uploads
SAMPLES_DIR = PROJECT_ROOT / "samples"    # small test images (tracked in Git)
LOGS_DIR = PROJECT_ROOT / "logs"          # log files

# Folders the application must be able to write to.
# (samples/ is not listed: it is only read from, and we create it by hand.)
WRITABLE_DIRS = [MODELS_DIR, OUTPUTS_DIR, UPLOADS_DIR, LOGS_DIR]


# ---------------------------------------------------------------------------
# Upload and image limits
# ---------------------------------------------------------------------------
# Largest uploaded file we accept. Prevents someone (or an accident)
# from filling memory or disk with a huge file.
MAX_UPLOAD_MB = _env_int("PHOTO_MAX_UPLOAD_MB", 10)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# File extensions accepted at upload time (lowercase, with the dot).
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Image formats as Pillow names them. We check the REAL content of the
# file against this list, because a file's extension can be faked.
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}

# Images with a shorter side than this are rejected: too little detail
# for the AI models to work with.
MIN_IMAGE_SIDE = _env_int("PHOTO_MIN_IMAGE_SIDE", 64)

# Images with more total pixels than this are rejected. 25,000,000 is
# about 5000 x 5000. Bigger images can exhaust the RAM of a normal PC
# and are also a known way to crash image software ("decompression bomb").
MAX_IMAGE_PIXELS = _env_int("PHOTO_MAX_IMAGE_PIXELS", 25_000_000)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("PHOTO_LOG_LEVEL", "INFO").upper()
LOG_FILE = LOGS_DIR / "app.log"
LOG_FILE_MAX_BYTES = 1 * 1024 * 1024   # start a new log file after 1 MB
LOG_FILE_BACKUP_COUNT = 3              # keep 3 old log files, delete older


def ensure_directories() -> None:
    """Create all writable folders if they do not exist yet."""
    for folder in WRITABLE_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


def print_summary() -> None:
    """Print the current settings (useful to check that everything is right)."""
    print("\n" + "=" * 62)
    print(" CONFIGURATION SUMMARY")
    print("=" * 62)
    print(f" Project root      : {PROJECT_ROOT}")
    print(f" Models folder     : {MODELS_DIR}")
    print(f" Outputs folder    : {OUTPUTS_DIR}")
    print(f" Uploads folder    : {UPLOADS_DIR}")
    print(f" Logs folder       : {LOGS_DIR}")
    print(f" Max upload size   : {MAX_UPLOAD_MB} MB")
    print(f" Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    print(f" Min image side    : {MIN_IMAGE_SIDE} px")
    print(f" Max image pixels  : {MAX_IMAGE_PIXELS:,}")
    print(f" Log level         : {LOG_LEVEL}")
    print("=" * 62)


if __name__ == "__main__":
    ensure_directories()
    print_summary()
    print(" Folders created/verified.")