"""
setup_face_code.py

Fetches the MODEL CODE (not the weights) needed for face restoration and
places it under src/vendor/:

    src/vendor/gfpgan/     GFPGAN v1.4 network   (TencentARC/GFPGAN)
    src/vendor/facexlib/   face detection, alignment and face-blending
                           helpers              (xinntao/facexlib, MIT)

    python scripts/setup_face_code.py            # fetch what is missing
    python scripts/setup_face_code.py --force    # fetch everything again

Why not "pip install gfpgan facexlib"?
  * gfpgan needs the "basicsr" package, which fails to install or import on
    many setups (known build and torchvision-version problems).
  * facexlib's pip package also installs filterpy, numba and scipy (and
    through them matplotlib) that we do not use.
  The GFPGAN v1.4 network needs only two tiny helpers from basicsr, which
  are re-implemented in a small compatibility file written by this script
  (src/vendor/gfpgan/_compat.py). facexlib needs only PyTorch, NumPy, OpenCV
  and torchvision, so just the needed files are copied.

Safety
  * Commits are pinned and every file is checked against a SHA-256 checksum.
  * Nothing is written for a package unless ALL its files passed the checks.
  * The only edits to upstream files are a few import lines (listed in each
    folder's NOTICE.txt).

LICENSE NOTE (important)
  GFPGAN says "Apache 2.0", but its LICENSE file lists third-party
  components, including StyleGAN2-derived code under NVIDIA's license, which
  limits use to research or evaluation purposes, and DFDNet-derived code
  under CC BY-NC-SA 4.0. Treat GFPGAN as NON-COMMERCIAL unless you get legal
  advice. For that reason src/vendor/gfpgan/ is kept OUT of Git (this script
  puts a .gitignore inside it). facexlib is MIT-licensed.

Uses only Python's standard library.
"""

import hashlib
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "src" / "vendor"


@dataclass
class Package:
    name: str
    repo: str                     # "owner/name" on GitHub
    commit: str
    target: Path
    files: list                   # (path in repo, local path, sha256)
    notice: str
    extra_files: dict = field(default_factory=dict)   # local path -> text we write
    keep_out_of_git: bool = False


# ---------------------------------------------------------------------------
# GFPGAN (clean architecture used by v1.4)
# ---------------------------------------------------------------------------
GFPGAN_COMMIT = "7552a7791caad982045a7bbe5634bbf1cd5c8679"

GFPGAN_REWRITES = {
    "gfpganv1_clean_arch.py": [
        ("from basicsr.utils.registry import ARCH_REGISTRY",
         "from ._compat import ARCH_REGISTRY"),
    ],
    "stylegan2_clean_arch.py": [
        ("from basicsr.archs.arch_util import default_init_weights",
         "from ._compat import default_init_weights"),
        ("from basicsr.utils.registry import ARCH_REGISTRY",
         "from ._compat import ARCH_REGISTRY"),
    ],
}

COMPAT_TEXT = '''"""
_compat.py  (written by scripts/setup_face_code.py; not part of GFPGAN)

GFPGAN's network files import two tiny helpers from the "basicsr" package.
Re-implemented here so that basicsr does not have to be installed.

  * ARCH_REGISTRY.register(): in basicsr this records a class in a registry.
    We only need the class to stay unchanged, so it returns it as it is.
  * default_init_weights(): random initialisation of layers, following
    basicsr (Apache 2.0). It only matters when TRAINING from scratch. For
    inference every weight is overwritten by the downloaded checkpoint.
"""

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm


class _Registry:
    def register(self, obj=None):
        if obj is None:
            return lambda cls: cls      # used as @ARCH_REGISTRY.register()
        return obj                      # used as @ARCH_REGISTRY.register


ARCH_REGISTRY = _Registry()


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialise Conv2d / Linear / BatchNorm layers (see basicsr)."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for layer in module.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(layer.weight, **kwargs)
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.fill_(bias_fill)
            elif isinstance(layer, _BatchNorm):
                init.constant_(layer.weight, 1)
                if layer.bias is not None:
                    layer.bias.data.fill_(bias_fill)
'''

GFPGAN_NOTICE = f"""GFPGAN network code (vendored)

Source:  https://github.com/TencentARC/GFPGAN
Commit:  {GFPGAN_COMMIT}
Files:   gfpgan/archs/gfpganv1_clean_arch.py, gfpgan/archs/stylegan2_clean_arch.py
Licence: see the LICENSE file in this folder. It is Apache 2.0 PLUS a list of
         third-party components, including StyleGAN2-derived code under NVIDIA's
         license (research or evaluation use only) and DFDNet-derived code under
         CC BY-NC-SA 4.0. Treat this code and the GFPGAN weights as NON-COMMERCIAL.

Changes made by this project:
  * gfpganv1_clean_arch.py, stylegan2_clean_arch.py: imports of
    "basicsr.utils.registry" and "basicsr.archs.arch_util" were replaced by
    "from ._compat import ...".
  * _compat.py was written by this project (small stand-ins for two basicsr helpers).

This folder is generated by scripts/setup_face_code.py and is not committed to Git.
"""

GFPGAN_PACKAGE = Package(
    name="GFPGAN",
    repo="TencentARC/GFPGAN",
    commit=GFPGAN_COMMIT,
    target=VENDOR_DIR / "gfpgan",
    files=[
        ("gfpgan/archs/gfpganv1_clean_arch.py", "gfpganv1_clean_arch.py",
         "576f44f3aae39e21a8138fe51f8ec96c0b94559e08fc10c78416e8658ac290a9"),
        ("gfpgan/archs/stylegan2_clean_arch.py", "stylegan2_clean_arch.py",
         "274f6a66c0c56ee565e0df9e77a90d2a475c766419e15026273ca10f0dfd0ba0"),
        ("LICENSE", "LICENSE",
         "b916d19bf894eb43409866adf17c9f71cbb17aefd9cd105117c4a32598ef6a46"),
    ],
    notice=GFPGAN_NOTICE,
    extra_files={
        "_compat.py": COMPAT_TEXT,
        ".gitignore": ("# Generated by scripts/setup_face_code.py. GFPGAN's license lists\n"
                       "# non-commercial third-party components, so this folder is not\n"
                       "# committed. Run scripts/setup_face_code.py after cloning.\n*\n"),
    },
    keep_out_of_git=True,
)

# ---------------------------------------------------------------------------
# facexlib (detection, parsing, alignment and the restoration helper)
# ---------------------------------------------------------------------------
FACEXLIB_COMMIT = "260620ae93990a300f4b16448df9bb459f1caba9"

FACEXLIB_NOTICE = f"""facexlib code (vendored subset)

Source:  https://github.com/xinntao/facexlib
Commit:  {FACEXLIB_COMMIT}
Licence: MIT (see the LICENSE file in this folder)
Files:   detection/*, parsing/*, utils/*  (only what face restoration needs;
         the tracking, headpose, matting, recognition and assessment parts
         and their heavy dependencies are not included)

Changes made by this project:
  * Absolute imports "from facexlib.X import ..." were changed to
    "from src.vendor.facexlib.X import ..." (11 lines) so the code works from here.
  * No other edits.

This folder is generated by scripts/setup_face_code.py.
"""

_FX = "facexlib/"
FACEXLIB_PACKAGE = Package(
    name="facexlib",
    repo="xinntao/facexlib",
    commit=FACEXLIB_COMMIT,
    target=VENDOR_DIR / "facexlib",
    files=[
        ("LICENSE", "LICENSE", "db131d6ba4392941108e6adea4216475221b30accd1f59c20f69de619fd39fd8"),
        (_FX + "detection/__init__.py", "detection/__init__.py", "b55639aee235130302a5770bd2a6e94dd7210353f2ea243b5b5dae965780c6b0"),
        (_FX + "detection/align_trans.py", "detection/align_trans.py", "418c4ecabb5278b6178f2fcc5539d4a75ec36bfc9f7455f153e7e476aa7c7070"),
        (_FX + "detection/matlab_cp2tform.py", "detection/matlab_cp2tform.py", "cc875b69dbaf31ae9aeee293538a31d3308db65fd257d0961a0be7d281a845ec"),
        (_FX + "detection/retinaface.py", "detection/retinaface.py", "13800a85446ee250781ed8091311bb6658490efe4fa7a40124014f0033364299"),
        (_FX + "detection/retinaface_net.py", "detection/retinaface_net.py", "ba8d0df7b79e34a016df0260e0454a43f0fdff61962ad8ecc39551b0ddac5d60"),
        (_FX + "detection/retinaface_utils.py", "detection/retinaface_utils.py", "26bdc6bd8f214fa76621ddfda3fd7571390847b403a1c7a93ad931f11b70d714"),
        (_FX + "parsing/__init__.py", "parsing/__init__.py", "08501ab312dee8495b55f9ae395bcf6372c4cfb1dd5587ffccc10b6321b56587"),
        (_FX + "parsing/bisenet.py", "parsing/bisenet.py", "470194b0566913e6daf500cd382d2e0060392fa8b8d944eda36f5ee65e3e3ce9"),
        (_FX + "parsing/parsenet.py", "parsing/parsenet.py", "1e952436140ef558eefe65e44e6be758abdf95c23be46c06b9c465c52dda8d30"),
        (_FX + "parsing/resnet.py", "parsing/resnet.py", "60bcc64e354b17c292a4ae5e5a94c74761e23c27c6d36908d4f152806428cdea"),
        (_FX + "utils/__init__.py", "utils/__init__.py", "8d73d9f1cb094c82fb9e418759d57f68a9fa4c3af43332b356a5e2425f8dc189"),
        (_FX + "utils/face_restoration_helper.py", "utils/face_restoration_helper.py", "0a2387b0faa3729d7a13176b4e53cab4ad6ed2c92ec75ea24b5d59aaeabb69bc"),
        (_FX + "utils/face_utils.py", "utils/face_utils.py", "6fd4b7d8d65f55ac0c234b9781130cfa3f3c0eba046a0e054c33801f7c353c81"),
        (_FX + "utils/misc.py", "utils/misc.py", "e5fac68f493d709eaa8fc6af6f9f399626898d4eee124e932353ab2abf604a01"),
    ],
    notice=FACEXLIB_NOTICE,
)

FACEXLIB_IMPORT = re.compile(r"^(\s*)from facexlib\.", re.MULTILINE)
EXPECTED_FACEXLIB_REWRITES = 11


# ---------------------------------------------------------------------------
def fetch(package: Package, remote_path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{package.repo}/{package.commit}/{remote_path}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "photo-restorer-setup/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def rewrite(package: Package, local_path: str, text: str):
    """Apply the import edits for one file. Returns (new_text, error or None)."""
    if package.name == "GFPGAN":
        for old, new in GFPGAN_REWRITES.get(local_path, []):
            if text.count(old) != 1:
                return text, f"expected exactly one line '{old}' in {local_path}"
            text = text.replace(old, new)
        if re.search(r"^\s*(from|import)\s+basicsr", text, re.MULTILINE):
            return text, f"a basicsr import is still present in {local_path}"
    elif package.name == "facexlib" and local_path.endswith(".py"):
        text, _ = FACEXLIB_IMPORT.subn(r"\1from src.vendor.facexlib.", text)
        if re.search(r"^\s*(from|import)\s+facexlib\b", text, re.MULTILINE):
            return text, f"an unrewritten facexlib import remains in {local_path}"
    return text, None


def install(package: Package, force: bool) -> bool:
    names = [local for _, local, _ in package.files] + ["NOTICE.txt"]
    names += list(package.extra_files)
    if not force and all((package.target / name).is_file() for name in names):
        print(f"{package.name}: already installed in {package.target} (use --force to refetch)")
        return True

    print(f"{package.name}: fetching commit {package.commit[:10]} ...")
    downloaded = {}
    for remote_path, local_path, expected_hash in package.files:
        try:
            data = fetch(package, remote_path)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"  FAILED to download {remote_path}: {error}")
            print("  Check your internet connection and try again.")
            return False
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            print(f"  CHECKSUM MISMATCH for {remote_path}")
            print(f"    expected {expected_hash}\n    got      {actual_hash}")
            print("  The file differs from the version this project was tested "
                  "with, so nothing was installed for this package.")
            return False
        downloaded[local_path] = data
    print(f"  {len(downloaded)} files downloaded, all checksums match")

    # Prepare all edited texts first; write only if everything succeeded.
    prepared = {}
    total_facexlib_rewrites = 0
    for local_path, data in downloaded.items():
        text = data.decode("utf-8")
        if package.name == "facexlib" and local_path.endswith(".py"):
            total_facexlib_rewrites += len(FACEXLIB_IMPORT.findall(text))
        text, problem = rewrite(package, local_path, text)
        if problem:
            print(f"  ERROR: {problem}. Nothing was installed.")
            return False
        prepared[local_path] = text
    if package.name == "facexlib" and total_facexlib_rewrites != EXPECTED_FACEXLIB_REWRITES:
        print(f"  ERROR: rewrote {total_facexlib_rewrites} imports, expected "
              f"{EXPECTED_FACEXLIB_REWRITES}. Nothing was installed.")
        return False

    for local_path, text in prepared.items():
        destination = package.target / local_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
    (package.target / "NOTICE.txt").write_text(package.notice, encoding="utf-8")
    for local_path, text in package.extra_files.items():
        (package.target / local_path).write_text(text, encoding="utf-8", newline="\n")
    print(f"  installed in {package.target}")
    if package.keep_out_of_git:
        print("  (this folder is excluded from Git on purpose; see the LICENSE NOTE)")
    return True


def main() -> int:
    force = "--force" in sys.argv[1:]
    ok = install(GFPGAN_PACKAGE, force) & install(FACEXLIB_PACKAGE, force)
    if ok:
        print("\nDone. Face restoration code is ready.")
        return 0
    print("\nSome packages were not installed. Fix the problem above and run again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())