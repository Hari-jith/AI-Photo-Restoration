"""
super_resolution.py

Enlarges photos with Real-ESRGAN (neural-network upscaling that adds
plausible detail, unlike a plain enlargement which only blurs).

    from src.super_resolution import Upscaler

    upscaler = Upscaler()                       # nothing is loaded yet
    result = upscaler.upscale(image, scale=2)   # loads the model on first use
    result.image       # Pillow RGB image, 2x larger
    result.model_name  # "general" or "x4plus"
    print(result.notes)

Two models (both are 4x networks; other scales are made by resizing):
  "general"  realesr-general-x4v3  small and FAST (about 5 s for a 0.1 MP photo
             on one CPU thread) but gives a smoother, "airbrushed" look and can
             add glow to eyes.
  "x4plus"   RealESRGAN_x4plus     much more natural-looking but about 14x
             SLOWER on CPU (about a minute for the same photo).
  "auto"     x4plus on a GPU with 4 GB+ memory, otherwise general.
These timings were measured on one CPU thread; yours will differ.

How tiling works: a big image is cut into tiles, each tile is enlarged
separately (with extra context around it), and the tiles are joined. Smaller
tiles need less memory. We use 32 pixels of context; the library's usual 10
produced visible differences from the untiled result in our measurements.
Unlike the original Real-ESRGAN code, a failing tile raises an error instead
of silently leaving a black square.

Honest limits
  * Upscaling sharpens EVERYTHING, including cracks, scratches and dust.
    Repair damage before upscaling.
  * Added detail is invented, not recovered.

Self-test (no weights needed):   python -m src.super_resolution
Upscale a file:                  python -m src.super_resolution samples\\test_1.jpg --scale 2
"""

import argparse
import math
import sys
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
from src.device import (free_memory, get_device_info, is_out_of_memory_error,
                        recommended_tile_size)
from src.logging_setup import get_logger

logger = get_logger(__name__)

# --- Settings ----------------------------------------------------------------
REALESRGAN_DIR = config.MODELS_DIR / "realesrgan"
MODEL_FILES = {
    "general": REALESRGAN_DIR / "realesr-general-x4v3.pth",
    "x4plus": REALESRGAN_DIR / "RealESRGAN_x4plus.pth",
}
NATIVE_SCALE = 4              # both networks enlarge 4x
TILE_PAD = 32                 # context pixels around each tile (see docstring)
BORDER_PAD = 10               # reflected border added around the whole image
MIN_TILE_SIZE = 64
MIN_SCALE, MAX_SCALE = 1.0, 4.0           # scale must be above 1.0, at most 4.0
MAX_NATIVE_OUTPUT_PIXELS = 40_000_000     # about 6300 x 6300 at 4x
GPU_MIN_VRAM_FOR_X4PLUS_GB = 4


class UpscaleError(Exception):
    """Upscaling could not be completed (message is user-friendly)."""


class UpscaleModelMissingError(UpscaleError):
    """A weights file is missing."""


@dataclass
class UpscaleResult:
    image: Image.Image
    scale: float
    model_name: str
    tile_size: int
    notes: list = field(default_factory=list)


def _resolve_model_name(name: str, device_kind: str, vram_gb) -> str:
    """Turn "auto" into a concrete model name (pure function, easy to test)."""
    if name not in ("auto", "general", "x4plus"):
        raise ValueError("model must be 'auto', 'general' or 'x4plus'.")
    if name != "auto":
        return name
    if device_kind == "cuda" and vram_gb is not None and vram_gb >= GPU_MIN_VRAM_FOR_X4PLUS_GB:
        return "x4plus"
    return "general"


def _build_network(model_name: str):
    """Create the network for a model name (random weights, evaluation mode)."""
    try:
        if model_name == "x4plus":
            from src.vendor.realesrgan.rrdbnet_arch import RRDBNet
            return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                           num_grow_ch=32, scale=4).eval()
        from src.vendor.realesrgan.srvgg_arch import SRVGGNetCompact
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                               upscale=4, act_type="prelu").eval()
    except ImportError as error:
        raise UpscaleError(
            "The upscaler code is missing. Run this once from the project "
            f"folder:  python scripts\\setup_upscaler_code.py  (details: {error})"
        ) from error


# ---------------------------------------------------------------------------
# Tiled enlargement of one image (the heart of the module)
# ---------------------------------------------------------------------------
def enlarge_native(model, rgb: np.ndarray, device, tile_size: int, progress=None) -> np.ndarray:
    """Enlarge an RGB uint8 array (H, W, 3) by the network's native factor (4).

    The image is reflected-padded, cut into tiles with TILE_PAD pixels of
    context, enlarged tile by tile, and joined. Errors are NOT swallowed.
    `progress(done, total)` is called after every tile.
    """
    height, width = rgb.shape[:2]
    x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    border = min(BORDER_PAD, height - 1, width - 1)   # reflect needs pad < size
    if border > 0:
        x = F.pad(x, (border,) * 4, mode="reflect")
    _, _, padded_h, padded_w = x.shape

    scale = NATIVE_SCALE
    output = np.zeros((padded_h * scale, padded_w * scale, 3), dtype=np.uint8)
    tiles_y = math.ceil(padded_h / tile_size)
    tiles_x = math.ceil(padded_w / tile_size)
    total, done = tiles_y * tiles_x, 0

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0, x0 = ty * tile_size, tx * tile_size
            y1, x1 = min(y0 + tile_size, padded_h), min(x0 + tile_size, padded_w)
            # the same tile, plus context pixels on every side (clipped at the edges)
            cy0, cx0 = max(y0 - TILE_PAD, 0), max(x0 - TILE_PAD, 0)
            cy1, cx1 = min(y1 + TILE_PAD, padded_h), min(x1 + TILE_PAD, padded_w)

            tile_in = x[:, :, cy0:cy1, cx0:cx1].to(device)
            with torch.no_grad():
                tile_out = model(tile_in)
            # keep only the part that belongs to the un-padded tile
            oy, ox = (y0 - cy0) * scale, (x0 - cx0) * scale
            tile_out = tile_out[:, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
            # clip BEFORE bytes: the network overshoots (about -0.1 .. 1.2)
            tile_bytes = (tile_out.clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
                          * 255.0).round().astype(np.uint8)
            output[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = tile_bytes
            done += 1
            if progress is not None:
                progress(done, total)

    if border > 0:
        cut = border * scale
        output = output[cut:output.shape[0] - cut, cut:output.shape[1] - cut]
    return np.ascontiguousarray(output)


# ---------------------------------------------------------------------------
# The upscaler
# ---------------------------------------------------------------------------
class Upscaler:
    """Loads Real-ESRGAN models lazily and enlarges images."""

    def __init__(self, model: str = "auto", models: dict | None = None):
        """
        Args:
            model: "auto", "general" or "x4plus" (see the module docstring).
            models: ready-made networks by name (used by the self-test).
        """
        self.requested_model = model
        _resolve_model_name(model, "cpu", None)        # validates the name early
        self._networks = dict(models) if models else {}
        self._device = None
        self._force_cpu = False
        self._lock = threading.Lock()

    def weights_available(self, model: str | None = None) -> bool:
        name = _resolve_model_name(model or self.requested_model,
                                   get_device_info().kind, get_device_info().vram_gb)
        return name in self._networks or MODEL_FILES[name].is_file()

    def _device_and_name(self):
        info = get_device_info()
        kind, vram = ("cpu", None) if self._force_cpu else (info.kind, info.vram_gb)
        device = torch.device("cpu") if self._force_cpu else info.device
        return device, _resolve_model_name(self.requested_model, kind, vram)

    def _get_network(self, name: str, device):
        if name in self._networks:
            return self._networks[name].to(device)
        path = MODEL_FILES[name]
        if not path.is_file():
            raise UpscaleModelMissingError(
                f"The upscaling weights {path.name} were not found at {path}. "
                "Download them with:  python scripts\\download_weights.py")
        started = time.time()
        network = _build_network(name)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            key = "params_ema" if "params_ema" in checkpoint else "params"
            network.load_state_dict(checkpoint[key], strict=True)
        except Exception as error:
            raise UpscaleError(
                f"The weights file {path.name} could not be loaded (damaged "
                "download, or not the right file). Run python "
                "scripts\\download_weights.py --verify, delete the file if it "
                f"reports a problem, and download it again. ({str(error)[:150]})"
            ) from error
        self._networks[name] = network.to(device)
        logger.info("Upscaling model '%s' loaded on %s in %.1f s",
                    name, device.type, time.time() - started)
        return self._networks[name]

    def upscale(self, image: Image.Image, scale: float = 2.0,
                tile_size: int | None = None, progress=None) -> UpscaleResult:
        """Enlarge an image.

        Args:
            image: Pillow image.
            scale: enlargement factor, above 1.0 and at most 4.0.
            tile_size: tile side in pixels (default: chosen from your hardware;
                smaller = less memory, slower).
            progress: optional function progress(done_tiles, total_tiles).
        """
        if not MIN_SCALE < scale <= MAX_SCALE:
            raise ValueError(f"scale must be above {MIN_SCALE} and at most {MAX_SCALE}.")
        if tile_size is not None and tile_size < MIN_TILE_SIZE:
            raise ValueError(f"tile_size must be at least {MIN_TILE_SIZE}.")
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        if height * width * NATIVE_SCALE ** 2 > MAX_NATIVE_OUTPUT_PIXELS:
            limit_side = int(math.sqrt(MAX_NATIVE_OUTPUT_PIXELS) / NATIVE_SCALE)
            raise UpscaleError(
                f"This image ({width}x{height}) is too large to enlarge. Shrink it "
                f"to about {limit_side} pixels on its longest side first.")

        with self._lock:
            started = time.time()
            try:
                result = self._upscale(rgb, scale, tile_size, progress)
            except UpscaleError:
                raise
            except Exception as error:
                if (self._device is not None and self._device.type == "cuda"
                        and is_out_of_memory_error(error)):
                    logger.warning("GPU ran out of memory; retrying on the CPU.")
                    self._force_cpu = True
                    free_memory()
                    try:
                        result = self._upscale(rgb, scale, tile_size, progress)
                    except Exception as retry_error:
                        raise UpscaleError(
                            f"Upscaling failed even on the CPU: {retry_error}") from retry_error
                    result.notes.insert(0, "The GPU ran out of memory, so the CPU was "
                                           f"used instead (slower). Model: '{result.model_name}'.")
                elif is_out_of_memory_error(error):
                    raise UpscaleError(
                        "Not enough memory to upscale this image. Close other "
                        "programs, use a smaller image, or a smaller scale.") from error
                else:
                    raise UpscaleError(f"Upscaling failed: {error}") from error
            logger.info("Upscaled %dx%d -> %dx%d with '%s' in %.1f s", width, height,
                        result.image.width, result.image.height, result.model_name,
                        time.time() - started)
            return result

    def _upscale(self, rgb, scale, tile_size, progress) -> UpscaleResult:
        device, name = self._device_and_name()
        self._device = device
        network = self._get_network(name, device)
        tile = tile_size or recommended_tile_size()
        notes = []

        big = enlarge_native(network, rgb, device, tile, progress)
        height, width = rgb.shape[:2]
        if scale != float(NATIVE_SCALE):
            target = (max(1, round(width * scale)), max(1, round(height * scale)))
            # AREA gives clean results when shrinking (4x network output -> smaller)
            big = cv2.resize(big, target, interpolation=cv2.INTER_AREA)
            notes.append(f"The network enlarges 4x; the result was resized to {scale:g}x.")
        return UpscaleResult(Image.fromarray(big, "RGB"), scale, name, tile, notes)


# ---------------------------------------------------------------------------
# Self-test (needs the vendored network code, but no weights)
# ---------------------------------------------------------------------------
class _NearestModel(torch.nn.Module):
    """Stand-in 4x network: repeats each pixel 4x4 (no receptive field)."""

    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode="nearest")


class _BlurModel(torch.nn.Module):
    """Stand-in 4x network with a receptive field: 3x3 box blur, then 4x repeat."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1.0 / 9.0)

    def forward(self, x):
        return F.interpolate(self.conv(x), scale_factor=4, mode="nearest")


class _ConstantModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4), self.value)


class _FailingModel(torch.nn.Module):
    def forward(self, x):
        raise ValueError("simulated tile failure")


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

    rng = np.random.default_rng(0)
    photo_array = rng.integers(0, 256, (100, 150, 3), dtype=np.uint8)     # 150 x 100
    photo = Image.fromarray(photo_array, "RGB")
    nearest = Upscaler(model="general", models={"general": _NearestModel()})

    # ---- geometry ----
    result = nearest.upscale(photo, scale=4, tile_size=64)
    expected = np.repeat(np.repeat(photo_array, 4, axis=0), 4, axis=1)
    check("4x: size is exactly 4 times larger, mode RGB",
          result.image.size == (600, 400) and result.image.mode == "RGB")
    check("4x: every pixel is in the right place (border padding removed correctly)",
          np.array_equal(np.array(result.image), expected))
    result = nearest.upscale(photo, scale=2)
    check("2x: result is resized to exactly 2x", result.image.size == (300, 200)
          and "resized" in result.notes[0])
    result = nearest.upscale(photo, scale=1.5)
    check("1.5x: non-integer scale gives rounded size", result.image.size == (225, 150))
    check("scale must be above 1 and at most 4",
          raises(ValueError, lambda: nearest.upscale(photo, scale=1.0))
          and raises(ValueError, lambda: nearest.upscale(photo, scale=4.5)))

    # ---- tiling ----
    blur = Upscaler(model="general", models={"general": _BlurModel()})
    tiled = np.array(blur.upscale(photo, scale=4, tile_size=64).image).astype(int)
    untiled = np.array(blur.upscale(photo, scale=4, tile_size=4096).image).astype(int)
    check("tiled result equals the untiled result (model with a receptive field)",
          int(np.abs(tiled - untiled).max()) <= 1, f"(max diff {np.abs(tiled - untiled).max()})")

    calls = []
    nearest.upscale(photo, scale=4, tile_size=64, progress=lambda d, t: calls.append((d, t)))
    padded_tiles = math.ceil(120 / 64) * math.ceil(170 / 64)       # padded 150x100 -> 170x120
    check("progress is reported once per tile and ends at 100%",
          len(calls) == padded_tiles and calls[-1][0] == calls[-1][1]
          and [c[0] for c in calls] == sorted(c[0] for c in calls))

    # ---- values and errors ----
    over = Upscaler(model="general", models={"general": _ConstantModel(1.3)}).upscale(photo, 4)
    under = Upscaler(model="general", models={"general": _ConstantModel(-0.2)}).upscale(photo, 4)
    check("values outside 0..1 are clipped, not wrapped around",
          np.array(over.image).min() == 255 and np.array(under.image).max() == 0)
    check("a failing tile raises an error (no silent black tiles)",
          raises(UpscaleError, lambda: Upscaler(
              model="general", models={"general": _FailingModel()}).upscale(photo, 2)))
    with mock.patch.dict(MODEL_FILES, {"x4plus": Path("no_such_folder/none.pth")}):
        check("missing weights give UpscaleModelMissingError",
              raises(UpscaleModelMissingError,
                     lambda: Upscaler(model="x4plus").upscale(photo, 2)))
    tiny = Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8), "RGB")
    check("a tiny 8x8 image works (border padding adapts)",
          nearest.upscale(tiny, scale=4, tile_size=64).image.size == (32, 32))
    check("a grayscale image is accepted",
          nearest.upscale(Image.new("L", (80, 80), 100), scale=2).image.mode == "RGB")
    huge = Image.new("RGB", (3200, 3200))
    check("an image too large to enlarge gives a clear error",
          raises(UpscaleError, lambda: nearest.upscale(huge, scale=2)))

    # ---- model choice and the real network definitions ----
    check("auto: CPU -> general, GPU with 8 GB -> x4plus, GPU with 2 GB -> general",
          _resolve_model_name("auto", "cpu", None) == "general"
          and _resolve_model_name("auto", "cuda", 8) == "x4plus"
          and _resolve_model_name("auto", "cuda", 2) == "general")
    check("an unknown model name is rejected", raises(ValueError, lambda: Upscaler(model="best")))
    try:
        for name in ("general", "x4plus"):
            with torch.no_grad():
                out = _build_network(name)(torch.rand(1, 3, 16, 16))
            check(f"real '{name}' network builds and enlarges 16x16 -> 64x64",
                  tuple(out.shape) == (1, 3, 64, 64))
    except UpscaleError as error:
        failures += 1
        print(f" [FAIL] {error}")

    print("-" * 62)
    present = {n: p.is_file() for n, p in MODEL_FILES.items()}
    print(f" Weights present: general={'yes' if present['general'] else 'NO'}, "
          f"x4plus={'yes' if present['x4plus'] else 'NO'}")
    print(" (The self-test does not load the real weights. To test them, run\n"
          "  python -m src.super_resolution samples\\test_1.jpg --scale 2)")
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


def _upscale_file(path: str, scale: float, model: str, tile: int | None) -> int:
    from src.image_io import ImageValidationError, load_image_from_path

    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1

    def show_progress(done, total):
        print(f"\r Tile {done}/{total}", end="", flush=True)

    print(f" Input {loaded.image.width}x{loaded.image.height}. Loading model and upscaling "
          f"(model '{model}'; 'x4plus' is slow on a CPU)...")
    started = time.time()
    try:
        result = Upscaler(model=model).upscale(
            loaded.image, scale=scale, tile_size=tile, progress=show_progress)
    except (UpscaleError, ValueError) as error:
        print(f"\n ERROR: {error}")
        return 1
    print()
    config.ensure_directories()
    out_path = config.OUTPUTS_DIR / f"upscaled_{result.model_name}_result.png"
    result.image.save(out_path)
    print(f" Model used: {result.model_name}, tile size {result.tile_size}")
    print(f" Output: {result.image.width}x{result.image.height} ({result.scale:g}x)")
    for note in result.notes:
        print(f" - {note}")
    print(f" Done in {time.time() - started:.1f} s on {get_device_info().kind.upper()}")
    print(f" Saved: {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-ESRGAN super-resolution")
    parser.add_argument("image", nargs="?", help="image to upscale (omit to run the self-test)")
    parser.add_argument("--scale", type=float, default=2.0, help="1.0 < scale <= 4.0 (default 2)")
    parser.add_argument("--model", default="auto", choices=["auto", "general", "x4plus"])
    parser.add_argument("--tile", type=int, default=None, help="tile size in pixels")
    arguments = parser.parse_args()
    if arguments.image:
        sys.exit(_upscale_file(arguments.image, arguments.scale, arguments.model, arguments.tile))
    sys.exit(_run_self_test())