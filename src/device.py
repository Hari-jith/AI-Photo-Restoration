"""
device.py

Decides whether the AI models run on the GPU ("cuda") or the CPU, and
recommends safe settings for each.

    from src.device import get_device_info

    info = get_device_info()
    model.to(info.device)

You can force the CPU for testing:   $env:PHOTO_DEVICE = "cpu"
Accepted values: auto (default), cpu, cuda.

Self-test / report:   python -m src.device
"""

import gc
import os
from dataclasses import dataclass

try:
    import torch
except ImportError as error:
    raise ImportError(
        "PyTorch is not installed in this environment. Activate the "
        "virtual environment (.venv\\Scripts\\Activate.ps1) and run "
        "python scripts/check_environment.py for installation help."
    ) from error

from src.logging_setup import get_logger

logger = get_logger(__name__)

BYTES_PER_GB = 1024 ** 3

# --- Settings ---------------------------------------------------------------
# Tile sizes (pixels) for models that process big images in pieces.
CPU_TILE_SIZE = 256
GPU_TILE_SIZE_8GB_PLUS = 512
GPU_TILE_SIZE_4GB_PLUS = 256
GPU_TILE_SIZE_SMALL = 128

# Longest image side we recommend working on (see preprocessing.py).
CPU_MAX_WORKING_SIDE = 1024
GPU_MAX_WORKING_SIDE_8GB_PLUS = 2048
GPU_MAX_WORKING_SIDE_4GB_PLUS = 1536
GPU_MAX_WORKING_SIDE_SMALL = 1024


@dataclass
class DeviceInfo:
    """What we will run on."""

    device: "torch.device"
    kind: str                   # "cuda" or "cpu"
    name: str                   # GPU model name, or "CPU"
    vram_gb: float | None       # GPU memory, None on CPU
    forced_cpu: bool            # True if PHOTO_DEVICE=cpu was set

    def summary(self) -> str:
        if self.kind == "cuda":
            return f"GPU: {self.name} ({self.vram_gb:.1f} GB VRAM)"
        if self.forced_cpu:
            return "CPU (forced by PHOTO_DEVICE=cpu)"
        return "CPU (no usable NVIDIA GPU found; processing will be slower)"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _detect() -> DeviceInfo:
    requested = os.environ.get("PHOTO_DEVICE", "auto").strip().lower()
    if requested not in ("auto", "cpu", "cuda"):
        logger.warning("Unknown PHOTO_DEVICE value '%s'; using 'auto'.", requested)
        requested = "auto"

    if requested != "cpu" and torch.cuda.is_available():
        try:
            # A tiny real calculation proves the GPU actually works.
            _ = torch.zeros(1, device="cuda") + 1
            properties = torch.cuda.get_device_properties(0)
            return DeviceInfo(
                device=torch.device("cuda"),
                kind="cuda",
                name=torch.cuda.get_device_name(0),
                vram_gb=properties.total_memory / BYTES_PER_GB,
                forced_cpu=False,
            )
        except Exception as error:   # driver problems raise many error types
            logger.warning("GPU detected but not usable (%s). Using CPU.", error)
    elif requested == "cuda":
        logger.warning("PHOTO_DEVICE=cuda was set but no GPU is available. "
                       "Using CPU.")

    return DeviceInfo(
        device=torch.device("cpu"), kind="cpu", name="CPU",
        vram_gb=None, forced_cpu=(requested == "cpu"),
    )


_cached_info: DeviceInfo | None = None


def get_device_info(refresh: bool = False) -> DeviceInfo:
    """Return the device to use. The result is cached after the first call;
    pass refresh=True to detect again (used by the self-test)."""
    global _cached_info
    if _cached_info is None or refresh:
        _cached_info = _detect()
        logger.info("Device selected: %s", _cached_info.summary())
    return _cached_info


# ---------------------------------------------------------------------------
# Recommendations (pure functions, so they are easy to test)
# ---------------------------------------------------------------------------
def _tile_size_for(kind: str, vram_gb: float | None) -> int:
    if kind != "cuda" or vram_gb is None:
        return CPU_TILE_SIZE
    if vram_gb >= 8:
        return GPU_TILE_SIZE_8GB_PLUS
    if vram_gb >= 4:
        return GPU_TILE_SIZE_4GB_PLUS
    return GPU_TILE_SIZE_SMALL


def _max_side_for(kind: str, vram_gb: float | None) -> int:
    if kind != "cuda" or vram_gb is None:
        return CPU_MAX_WORKING_SIDE
    if vram_gb >= 8:
        return GPU_MAX_WORKING_SIDE_8GB_PLUS
    if vram_gb >= 4:
        return GPU_MAX_WORKING_SIDE_4GB_PLUS
    return GPU_MAX_WORKING_SIDE_SMALL


def recommended_tile_size() -> int:
    """Tile size (pixels) for tiled models such as Real-ESRGAN."""
    info = get_device_info()
    return _tile_size_for(info.kind, info.vram_gb)


def recommended_max_working_side() -> int:
    """Longest image side we suggest processing on this machine."""
    info = get_device_info()
    return _max_side_for(info.kind, info.vram_gb)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def is_out_of_memory_error(error: BaseException) -> bool:
    """True if the error means 'not enough memory' (GPU or normal RAM)."""
    if isinstance(error, MemoryError):
        return True
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(error, cuda_oom):
        return True
    # Older PyTorch versions and some CPU paths raise a plain RuntimeError.
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def free_memory() -> None:
    """Release memory that is no longer needed (call after a model is done)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    original_setting = os.environ.get("PHOTO_DEVICE")
    try:
        os.environ.pop("PHOTO_DEVICE", None)
        info = get_device_info(refresh=True)
        print(f"\n Detected: {info.summary()}")
        print(f" Recommended tile size        : {recommended_tile_size()} px")
        print(f" Recommended max working side : "
              f"{recommended_max_working_side()} px\n")

        a = torch.rand(64, 64, device=info.device)
        check(f"calculation works on {info.kind}",
              (a @ a).device.type == info.kind)

        os.environ["PHOTO_DEVICE"] = "cpu"
        forced = get_device_info(refresh=True)
        check("PHOTO_DEVICE=cpu forces the CPU",
              forced.kind == "cpu" and forced.forced_cpu)

        os.environ["PHOTO_DEVICE"] = "banana"
        fallback = get_device_info(refresh=True)
        check("invalid PHOTO_DEVICE value falls back to auto",
              fallback.kind in ("cpu", "cuda"))
    finally:
        if original_setting is None:
            os.environ.pop("PHOTO_DEVICE", None)
        else:
            os.environ["PHOTO_DEVICE"] = original_setting
        get_device_info(refresh=True)

    check("CPU tile size is 256", _tile_size_for("cpu", None) == 256)
    check("12 GB GPU -> 512 tiles", _tile_size_for("cuda", 12) == 512)
    check("6 GB GPU -> 256 tiles", _tile_size_for("cuda", 6) == 256)
    check("2 GB GPU -> 128 tiles", _tile_size_for("cuda", 2) == 128)
    check("CPU working side is 1024", _max_side_for("cpu", None) == 1024)
    check("12 GB GPU working side is 2048", _max_side_for("cuda", 12) == 2048)

    check("recognizes 'CUDA out of memory' RuntimeError",
          is_out_of_memory_error(RuntimeError("CUDA out of memory. Tried to allocate")))
    check("recognizes MemoryError", is_out_of_memory_error(MemoryError()))
    check("ignores unrelated errors", not is_out_of_memory_error(ValueError("x")))

    try:
        free_memory()
        check("free_memory runs without error", True)
    except Exception as error:
        check("free_memory runs without error", False, str(error))

    print("-" * 62)
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_self_test())