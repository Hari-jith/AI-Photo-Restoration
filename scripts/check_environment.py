"""
check_environment.py

Checks that your computer is correctly set up for the AI Old Photo
Restoration project. Run it from the project root:

    python scripts/check_environment.py

It prints PASS / WARN / FAIL for each check, with a suggested fix for
anything that is not PASS. It uses only the Python standard library at the
top level, so it still runs (and tells you what is missing) even when
packages are not installed yet.
"""

import ctypes
import importlib.metadata as metadata
import platform
import shutil
import sys
from pathlib import Path

# ---- Settings (change here if our requirements change) ----
RECOMMENDED_PYTHON = (3, 11)       # the version we test against
MIN_FREE_DISK_GB = 10              # models + outputs need space
MIN_RAM_GB = 8                     # below this, big images may crash
MAX_NUMPY_MAJOR = 1                # numpy 2.x breaks older AI libraries
MAX_TORCHVISION_MINOR = 16         # torchvision 0.17+ breaks basicsr

# Each result is (status, check name, message)
results = []


def add(status: str, name: str, message: str) -> None:
    """Record one check result."""
    results.append((status, name, message))


def package_version(package_name: str):
    """Return installed version of a package, or None if not installed."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def parse_version(version: str) -> tuple:
    """Turn '1.26.4' or '2.1.2+cpu' into (1, 26, 4) for comparison."""
    clean = version.split("+")[0]
    parts = []
    for piece in clean.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def total_ram_gb():
    """Total RAM in GB on Windows (standard library only). None elsewhere."""
    if platform.system() != "Windows":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullTotalPhys / (1024 ** 3)


def check_python() -> None:
    current = sys.version_info[:2]
    text = f"Python {platform.python_version()}"
    if current == RECOMMENDED_PYTHON:
        add("PASS", "Python version", text)
    elif (3, 10) <= current <= (3, 12):
        add("WARN", "Python version",
            f"{text}. It may work, but we test with 3.11. "
            "Recreate the venv with: py -3.11 -m venv .venv")
    else:
        add("FAIL", "Python version",
            f"{text} is not supported. Install Python 3.11 from python.org "
            "and recreate the venv with: py -3.11 -m venv .venv")


def check_virtual_environment() -> None:
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        add("PASS", "Virtual environment", "Active")
    else:
        add("FAIL", "Virtual environment",
            "Not active. Run: .venv\\Scripts\\Activate.ps1 "
            "(your prompt should start with (.venv))")


def check_system() -> None:
    add("PASS", "Operating system", f"{platform.system()} {platform.release()}")

    ram = total_ram_gb()
    if ram is None:
        add("WARN", "RAM", "Could not detect RAM on this OS (not a problem)")
    elif ram >= MIN_RAM_GB:
        add("PASS", "RAM", f"{ram:.1f} GB")
    else:
        add("WARN", "RAM",
            f"{ram:.1f} GB (below {MIN_RAM_GB} GB). Large photos may fail; "
            "we will limit image size.")

    project_root = Path(__file__).resolve().parent.parent
    free_gb = shutil.disk_usage(project_root).free / (1024 ** 3)
    if free_gb >= MIN_FREE_DISK_GB:
        add("PASS", "Free disk space", f"{free_gb:.1f} GB")
    else:
        add("WARN", "Free disk space",
            f"{free_gb:.1f} GB (want at least {MIN_FREE_DISK_GB} GB)")


def check_packages() -> None:
    """Check that base packages are installed and actually import."""
    # (pip name, import name, install hint)
    required = [
        ("numpy", "numpy", "pip install -r requirements.txt"),
        ("Pillow", "PIL", "pip install -r requirements.txt"),
        ("opencv-python", "cv2", "pip install -r requirements.txt"),
    ]
    for pip_name, import_name, hint in required:
        version = package_version(pip_name)
        if version is None:
            add("FAIL", pip_name, f"Not installed. Run: {hint}")
            continue
        try:
            __import__(import_name)
        except Exception as error:  # broken install can raise many error types
            add("FAIL", pip_name,
                f"Installed ({version}) but failed to import: {error}")
            continue
        add("PASS", pip_name, version)

    # numpy 2.x causes problems with the libraries we add later.
    numpy_version = package_version("numpy")
    if numpy_version and parse_version(numpy_version)[0] > MAX_NUMPY_MAJOR:
        add("FAIL", "numpy version",
            f"{numpy_version} is too new. Run: pip install numpy==1.26.4")


def check_pytorch() -> None:
    torch_version = package_version("torch")
    vision_version = package_version("torchvision")

    if torch_version is None:
        add("FAIL", "PyTorch",
            "Not installed. See the PyTorch install commands in the "
            "Module 1 instructions (CPU or NVIDIA).")
        return
    add("PASS", "PyTorch", torch_version)

    if vision_version is None:
        add("FAIL", "torchvision",
            "Not installed. Install it together with torch (see instructions).")
    else:
        minor = parse_version(vision_version)[1]
        if minor > MAX_TORCHVISION_MINOR:
            add("WARN", "torchvision",
                f"{vision_version} is newer than our tested 0.16.x. "
                "It will likely break basicsr later. Reinstall using the "
                "pinned commands in the instructions.")
        else:
            add("PASS", "torchvision", vision_version)

    try:
        import torch
    except Exception as error:
        add("FAIL", "PyTorch import", f"Could not import torch: {error}")
        return

    # A tiny calculation proves PyTorch really works, not just that it exists.
    try:
        matrix = torch.rand(256, 256)
        _ = matrix @ matrix
        add("PASS", "PyTorch CPU test", "Matrix multiplication works on CPU")
    except Exception as error:
        add("FAIL", "PyTorch CPU test", str(error))
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        try:
            gpu_matrix = torch.rand(256, 256, device="cuda")
            _ = gpu_matrix @ gpu_matrix
            add("PASS", "GPU (CUDA)",
                f"{gpu_name}, {vram_gb:.1f} GB VRAM. The app will use the GPU.")
        except Exception as error:
            add("WARN", "GPU (CUDA)",
                f"GPU detected but a test calculation failed: {error}. "
                "The app will fall back to CPU.")
    else:
        add("WARN", "GPU (CUDA)",
            "No usable GPU found. The app will run on CPU: this works, "
            "but is slower. (If you DO have an NVIDIA GPU, you probably "
            "installed the CPU version of PyTorch. See instructions.)")


def print_report() -> int:
    print("\n" + "=" * 62)
    print(" ENVIRONMENT CHECK: AI Old Photo Restoration")
    print("=" * 62)
    for status, name, message in results:
        print(f" [{status}] {name}: {message}")
    print("=" * 62)

    failures = sum(1 for s, _, _ in results if s == "FAIL")
    warnings = sum(1 for s, _, _ in results if s == "WARN")
    if failures:
        print(f" RESULT: {failures} FAIL, {warnings} WARN. "
              "Fix the FAIL items above, then run this script again.")
        return 1
    print(f" RESULT: Ready to continue ({warnings} warning(s)).")
    return 0


def main() -> None:
    check_python()
    check_virtual_environment()
    check_system()
    check_packages()
    check_pytorch()
    sys.exit(print_report())


if __name__ == "__main__":
    main()