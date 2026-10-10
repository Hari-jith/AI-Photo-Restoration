"""
download_weights.py

Downloads the AI model weights from official, non-Hugging-Face sources into
the models/ folder, and checks that each file looks complete.

    python scripts/download_weights.py --list        show what exists / is missing
    python scripts/download_weights.py               download everything missing
    python scripts/download_weights.py --only gfpgan-v1.4 realesrgan-x4plus

Uses only Python's standard library (no extra packages). Files are saved as
".part" first and renamed only when complete, so an interrupted download
never leaves a broken file behind.

NOTE: file sizes below are approximate (from memory). A file smaller than
'min_bytes' is treated as a failed download (usually an error page, or a
link that has moved).
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Allow "python scripts/download_weights.py" to find the src package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

MB = 1024 * 1024
CHUNK_SIZE = 1 * MB
CONNECT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class WeightFile:
    key: str            # short name used with --only
    folder: str         # subfolder inside models/
    filename: str
    urls: tuple         # tried in order until one works
    min_bytes: int      # smaller than this = failed download
    approx_mb: int
    used_by: str
    manual_help: str = ""


GITHUB_REALESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download"
GITHUB_GFPGAN = "https://github.com/TencentARC/GFPGAN/releases/download"
GITHUB_FACEXLIB = "https://github.com/xinntao/facexlib/releases/download"

WEIGHTS = [
    WeightFile(
        key="realesrgan-x4plus", folder="realesrgan",
        filename="RealESRGAN_x4plus.pth",
        urls=(f"{GITHUB_REALESRGAN}/v0.1.0/RealESRGAN_x4plus.pth",),
        min_bytes=50 * MB, approx_mb=64,
        used_by="Super-resolution (best quality, slower) - Module 12",
    ),
    WeightFile(
        key="realesrgan-general-x4v3", folder="realesrgan",
        filename="realesr-general-x4v3.pth",
        urls=(f"{GITHUB_REALESRGAN}/v0.2.5.0/realesr-general-x4v3.pth",),
        min_bytes=3 * MB, approx_mb=5,
        used_by="Super-resolution (small, CPU-friendly) - Module 12",
    ),
    WeightFile(
        key="gfpgan-v1.4", folder="gfpgan", filename="GFPGANv1.4.pth",
        urls=(f"{GITHUB_GFPGAN}/v1.3.0/GFPGANv1.4.pth",
              f"{GITHUB_GFPGAN}/v1.3.4/GFPGANv1.4.pth"),
        min_bytes=250 * MB, approx_mb=333,
        used_by="Face restoration - Module 11",
    ),
    WeightFile(
        key="facexlib-detection", folder="facexlib",
        filename="detection_Resnet50_Final.pth",
        urls=(f"{GITHUB_FACEXLIB}/v0.1.0/detection_Resnet50_Final.pth",),
        min_bytes=80 * MB, approx_mb=104,
        used_by="Face detection inside GFPGAN - Module 11",
    ),
    WeightFile(
        key="facexlib-parsing", folder="facexlib",
        filename="parsing_parsenet.pth",
        urls=(f"{GITHUB_FACEXLIB}/v0.2.2/parsing_parsenet.pth",),
        min_bytes=70 * MB, approx_mb=85,
        used_by="Face blending inside GFPGAN - Module 11",
    ),
    WeightFile(
        key="ddcolor", folder="ddcolor", filename="pytorch_model.pt",
        urls=("https://www.modelscope.cn/models/damo/"
              "cv_ddcolor_image-colorization/resolve/master/pytorch_model.pt",),
        min_bytes=100 * MB, approx_mb=0,   # size unconfirmed: several hundred MB
        used_by="Colorization (DDColor, official ModelScope release) - Module 9",
        manual_help=(
            "Manual download (works in any browser):\n"
            "   1. Open https://www.modelscope.cn/models/damo/"
            "cv_ddcolor_image-colorization\n"
            "   2. Click the 'Files' / model-files tab and download "
            "pytorch_model.pt\n"
            "   3. Save it as:  models\\ddcolor\\pytorch_model.pt\n"
            "   (ModelScope servers are in China, so the download can be slow.)"
        ),
    ),
]


# ---------------------------------------------------------------------------
def destination_of(weight: WeightFile) -> Path:
    return config.MODELS_DIR / weight.folder / weight.filename


def is_present(weight: WeightFile) -> bool:
    path = destination_of(weight)
    return path.is_file() and path.stat().st_size >= weight.min_bytes


def download_one_url(url: str, destination: Path, min_bytes: int) -> None:
    """Download url to destination. Raises an exception on any problem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "photo-restorer-weights-downloader/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=CONNECT_TIMEOUT_SECONDS) as response:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            with open(partial, "wb") as out_file:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    received += len(chunk)
                    if total:
                        print(f"\r    {received / MB:7.1f} / {total / MB:.1f} MB "
                              f"({100 * received / total:3.0f}%)", end="", flush=True)
                    else:
                        print(f"\r    {received / MB:7.1f} MB", end="", flush=True)
        print()
        if received < min_bytes:
            raise ValueError(
                f"downloaded only {received / MB:.1f} MB, expected at least "
                f"{min_bytes / MB:.0f} MB (the link may have moved)")
        os.replace(partial, destination)     # atomic: file appears only when complete
    finally:
        if partial.exists():
            partial.unlink()                 # never leave a broken .part behind


def download_weight(weight: WeightFile) -> bool:
    destination = destination_of(weight)
    print(f"\n* {weight.key}  ({weight.used_by})")
    if is_present(weight):
        print("    already present, skipping")
        return True
    for url in weight.urls:
        print(f"    from {url}")
        try:
            download_one_url(url, destination, weight.min_bytes)
            print(f"    saved: {destination}")
            return True
        except urllib.error.HTTPError as error:
            print(f"\n    FAILED: server answered {error.code} {error.reason}")
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"\n    FAILED: could not connect ({error}). "
                  "Check your internet connection or firewall.")
        except (OSError, ValueError) as error:
            print(f"\n    FAILED: {error}")
    if weight.manual_help:
        print("    " + weight.manual_help.replace("\n", "\n    "))
    else:
        print("    Automatic download failed. Open the URL above in a browser; "
              f"if it works, save the file as:\n    {destination}")
    return False


def print_status() -> None:
    print("\n" + "=" * 70)
    print(" MODEL WEIGHTS STATUS")
    print("=" * 70)
    for weight in WEIGHTS:
        state = "present" if is_present(weight) else "MISSING"
        size = f"~{weight.approx_mb} MB" if weight.approx_mb else "size varies"
        print(f" [{state:>7}] {weight.key:<24} {size:<12} {weight.folder}/{weight.filename}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights.")
    parser.add_argument("--list", action="store_true",
                        help="only show which files are present or missing")
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        help="download only these keys (see --list)")
    args = parser.parse_args()

    if args.list:
        print_status()
        return 0

    selected = WEIGHTS
    if args.only:
        known = {weight.key for weight in WEIGHTS}
        unknown = [key for key in args.only if key not in known]
        if unknown:
            print(f"Unknown key(s): {', '.join(unknown)}")
            print(f"Valid keys: {', '.join(sorted(known))}")
            return 1
        selected = [weight for weight in WEIGHTS if weight.key in args.only]

    results = {weight.key: download_weight(weight) for weight in selected}
    print_status()
    failed = [key for key, ok in results.items() if not ok]
    if failed:
        print(f" {len(failed)} file(s) need attention: {', '.join(failed)}")
        print(" Fix them (see messages above) and run this script again.")
        return 1
    print(" All requested weights are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())