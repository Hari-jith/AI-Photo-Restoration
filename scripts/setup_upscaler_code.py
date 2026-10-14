"""
setup_upscaler_code.py

Fetches the network definitions needed for Real-ESRGAN super-resolution and
places them in  src/vendor/realesrgan/ :

    rrdbnet_arch.py   RRDBNet, used by RealESRGAN_x4plus.pth
                      (from BasicSR, Apache 2.0)
    srvgg_arch.py     SRVGGNetCompact, used by realesr-general-x4v3.pth
                      (from Real-ESRGAN, BSD 3-Clause)

    python scripts/setup_upscaler_code.py            # fetch if missing
    python scripts/setup_upscaler_code.py --force    # fetch again

Why not "pip install realesrgan basicsr"?
  * basicsr is hard to install and import on many setups (known problems).
  * Real-ESRGAN's own inference class swallows errors: if a tile fails (for
    example when memory runs out), it prints a message and leaves that tile
    BLACK, then carries on. We write our own tiling that reports errors.
  So only the two small network files are copied. Three tiny helpers they
  import from basicsr are re-implemented in _compat.py (copied from the
  basicsr source, which we read and compared).

Safety: pinned commits, SHA-256 check of every file, nothing written unless
all files pass. Uses only Python's standard library.

Both upstream projects are permissively licensed (Apache 2.0, BSD 3-Clause),
so this folder is safe to commit to Git.
"""

import hashlib
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "src" / "vendor" / "realesrgan"

BASICSR_COMMIT = "8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a"
REALESRGAN_COMMIT = "a4abfb2979a7bbff3f69f58f58ae324608821e27"

# (GitHub repo, commit, path in repo, local name, sha256 of the original)
FILES = [
    ("XPixelGroup/BasicSR", BASICSR_COMMIT, "basicsr/archs/rrdbnet_arch.py",
     "rrdbnet_arch.py",
     "6aa8eb3f77e918b90eeff4ebfaf8745abe0c40eb8fcc03abbc1928693d70ab58"),
    ("xinntao/Real-ESRGAN", REALESRGAN_COMMIT, "realesrgan/archs/srvgg_arch.py",
     "srvgg_arch.py",
     "7419a75e789ca09295266563d74a6d7c08d6c24502af6b0a919b1811d0a0be59"),
    ("xinntao/Real-ESRGAN", REALESRGAN_COMMIT, "LICENSE", "LICENSE",
     "4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2"),
]

# Exact import lines to replace, per file (each must occur exactly once).
REWRITES = {
    "rrdbnet_arch.py": [
        ("from basicsr.utils.registry import ARCH_REGISTRY",
         "from ._compat import ARCH_REGISTRY"),
        ("from .arch_util import default_init_weights, make_layer, pixel_unshuffle",
         "from ._compat import default_init_weights, make_layer, pixel_unshuffle"),
    ],
    "srvgg_arch.py": [
        ("from basicsr.utils.registry import ARCH_REGISTRY",
         "from ._compat import ARCH_REGISTRY"),
    ],
}

COMPAT_TEXT = '''"""
_compat.py  (written by scripts/setup_upscaler_code.py; not part of Real-ESRGAN)

Stand-ins for the helpers that the network files import from "basicsr", so
that basicsr does not have to be installed. The three functions below are
copies of the basicsr implementations (Apache 2.0, BasicSR project).
"""

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm


class _Registry:
    def register(self, obj=None):
        if obj is None:
            return lambda cls: cls      # used as @ARCH_REGISTRY.register()
        return obj


ARCH_REGISTRY = _Registry()


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights (only matters for training from scratch)."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    """Pixel unshuffle: (b, c, hh, hw) -> (b, c*scale^2, hh/scale, hw/scale)."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)
'''

NOTICE_TEXT = f"""Real-ESRGAN network definitions (vendored)

rrdbnet_arch.py
  Source:  https://github.com/XPixelGroup/BasicSR  (commit {BASICSR_COMMIT})
  Licence: Apache License 2.0, as stated in BasicSR's LICENSE/README.md
           (full text: https://www.apache.org/licenses/LICENSE-2.0).
           BasicSR's list of third-party components (StyleGAN2, DFDNet, ...)
           concerns other files, not this one.
  The RRDB architecture is from ESRGAN (Wang et al., 2018).

srvgg_arch.py and LICENSE
  Source:  https://github.com/xinntao/Real-ESRGAN  (commit {REALESRGAN_COMMIT})
  Licence: BSD 3-Clause (see the LICENSE file in this folder)

Changes made by this project:
  * rrdbnet_arch.py, srvgg_arch.py: imports of "basicsr.utils.registry" and
    "basicsr.archs.arch_util" were replaced by "from ._compat import ...".
  * _compat.py was written by this project (copies of three small basicsr helpers).

This folder is generated by scripts/setup_upscaler_code.py.
"""


def fetch(repo: str, commit: str, remote_path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{remote_path}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "photo-restorer-setup/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main() -> int:
    force = "--force" in sys.argv[1:]
    expected = [local for _, _, _, local, _ in FILES] + ["_compat.py", "NOTICE.txt"]
    if not force and all((TARGET_DIR / name).is_file() for name in expected):
        print(f"Upscaler code is already in {TARGET_DIR}. (Use --force to refetch.)")
        return 0

    print("Fetching Real-ESRGAN network code ...")
    downloaded = {}
    for repo, commit, remote_path, local_name, expected_hash in FILES:
        try:
            data = fetch(repo, commit, remote_path)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"  FAILED to download {repo}:{remote_path}: {error}")
            print("  Check your internet connection and try again.")
            return 1
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            print(f"  CHECKSUM MISMATCH for {repo}:{remote_path}")
            print(f"    expected {expected_hash}\n    got      {actual_hash}")
            print("  The file differs from the tested version; nothing was installed.")
            return 1
        downloaded[local_name] = data.decode("utf-8")
        print(f"  ok  {local_name}")

    for local_name, edits in REWRITES.items():
        text = downloaded[local_name]
        for old, new in edits:
            if text.count(old) != 1:
                print(f"  ERROR: expected exactly one line '{old}' in {local_name}. "
                      "Nothing was installed.")
                return 1
            text = text.replace(old, new)
        if re.search(r"^\s*(from|import)\s+basicsr", text, re.MULTILINE):
            print(f"  ERROR: a basicsr import remains in {local_name}. Nothing installed.")
            return 1
        downloaded[local_name] = text

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, text in downloaded.items():
        (TARGET_DIR / local_name).write_text(text, encoding="utf-8", newline="\n")
    (TARGET_DIR / "_compat.py").write_text(COMPAT_TEXT, encoding="utf-8", newline="\n")
    (TARGET_DIR / "NOTICE.txt").write_text(NOTICE_TEXT, encoding="utf-8")
    print(f"\nDone. Upscaler code is in {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())