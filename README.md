# AI Old Photo Restoration & Colorization

> **Work in progress.** This README describes the project as built through
> **Module 10 of 25**. There is **no website yet**. What exists today is a
> tested set of command-line building blocks (loading, analysis, cleanup,
> colorization). Sections marked *Planned* do not exist yet. This file is
> updated as modules are added.
>
> Last updated: October 2026

A locally-run, zero-cost tool that restores old photographs with open-source
models: it will clean damage, restore faces, colorize black-and-white photos
and upscale the result, then show a before/after comparison on a simple
website. No paid services, no cloud APIs, and **no Hugging Face** (models come
from official GitHub repositories and other official project hosts).

The project is also a learning build: it is developed one small, tested module
at a time, with the reasoning for each decision kept in the code comments.

---

## Contents

1. [Current status](#1-current-status)
2. [Features](#2-features)
3. [Planned architecture](#3-planned-architecture)
4. [Technology stack](#4-technology-stack)
5. [Prerequisites and hardware](#5-prerequisites-and-hardware)
6. [Installation](#6-installation)
7. [Model weights](#7-model-weights)
8. [Running and testing what exists](#8-running-and-testing-what-exists)
9. [How the implemented modules work](#9-how-the-implemented-modules-work)
10. [Configuration](#10-configuration)
11. [Project structure](#11-project-structure)
12. [Models and licenses](#12-models-and-licenses)
13. [Known limitations](#13-known-limitations)
14. [Troubleshooting](#14-troubleshooting)
15. [Roadmap](#15-roadmap)
16. [Git and GitHub notes](#16-git-and-github-notes)
17. [Credits](#17-credits)

---

## 1. Current status

| # | Module | Status |
|---|---|---|
| 1 | Environment setup and check (`scripts/check_environment.py`) | Done |
| 2 | Git setup and `.gitignore` | Done |
| 3 | Central config and logging (`src/config.py`, `src/logging_setup.py`) | Done |
| 4 | Safe image loading and validation (`src/image_io.py`) | Done |
| 5 | Image analysis (`src/analysis.py`) | Done |
| 6 | Preprocessing (`src/preprocessing.py`) | Done |
| 7 | Device manager and weights downloader (`src/device.py`, `scripts/download_weights.py`) | Done |
| 8 | Classical restoration: denoise and scratch repair (`src/classical_restore.py`) | Done |
| 9 | Colorization with DDColor (`src/colorize.py`) | Done (code and self-test); real-photo quality not yet evaluated |
| 10 | Backup colorizer and fallback chain (`src/colorize_backup.py`) | Done (code and self-test); real-weights run not yet confirmed |
| 11-25 | Face restoration, super-resolution, pipeline, evaluation, website, tests, packaging | **Planned** (see [Roadmap](#15-roadmap)) |

**Not measured yet:** there are no quality benchmarks, no PSNR/SSIM results and
no timing table for the full pipeline. Evaluation is planned for Module 16.
Nothing in this README should be read as a quality claim.

---

## 2. Features

### Implemented

- **Safe image intake:** extension and real-content format check, size limits,
  corrupt or truncated file detection, "decompression bomb" protection, EXIF
  rotation, transparency flattened onto white, 16-bit grayscale handling, safe
  random filenames. All errors have beginner-readable messages.
- **Photo analysis:** detects grayscale vs sepia-tinted vs colour photos (by
  measuring pixels, not the file mode), and estimates contrast, sharpness,
  noise, and a rough face hint.
- **Preprocessing:** conversions between Pillow, RGB NumPy and BGR NumPy
  (a classic source of swapped-colour bugs, handled in one place) and a working
  size limit that never enlarges an image.
- **Classical cleanup:** Non-Local-Means denoising and thin-scratch repair
  (OpenCV morphology + inpainting), with a safety cap that skips repair when the
  detector is probably reacting to texture.
- **Colorization (main):** DDColor, run locally, preserving the original
  lightness channel exactly and predicting only colour. Adjustable saturation.
- **Colorization (backup):** Zhang et al. 2016 via OpenCV, plus a
  `ColorizerChain` that falls back to it automatically if DDColor fails, and
  reports why.
- **Hardware handling:** automatic CPU/GPU detection with CPU fallback, an
  out-of-memory retry on CPU, tile-size recommendations by available GPU memory.
- **Weights management:** one script downloads weights from official sources
  with size checks, SHA-256 verification where a checksum is known, and a
  `--verify` command.

### Planned

Face restoration (GFPGAN), super-resolution (Real-ESRGAN), a complete pipeline
and command-line runner, a FastAPI backend, a plain HTML/CSS/JS website
(drag-and-drop upload, progress status, before/after slider, download), tests,
performance tuning, and packaging.

---

## 3. Planned architecture

```
Browser (HTML/CSS/JS)                                   [Planned]
   |  upload image + options
   v
FastAPI server --> background job (in memory)           [Planned]
   |
   v
Restoration pipeline
   validate -> analyze -> clean -> colorize -> faces -> upscale -> post-process
     done       done      done      done      planned    planned     planned
   |
   v
outputs/ -> status polling -> before/after slider + download
```

Design principles: the pipeline works from the command line first and the
website is a thin layer on top; no database, Redis or Docker; models are
loaded lazily (only when a feature is requested); every step has a CPU
fallback.

---

## 4. Technology stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Deep learning | PyTorch 2.1.2 (CPU or CUDA 12.1 build) |
| Image handling | Pillow 10.4.0, OpenCV 4.10 (`opencv-python`) |
| Numerics | NumPy 1.26.4 (pinned below 2 for compatibility with later libraries) |
| Backend *(planned)* | FastAPI + Uvicorn |
| Frontend *(planned)* | Plain HTML, CSS, JavaScript (no Node.js needed) |
| Testing *(planned)* | pytest (each module currently has a built-in self-test) |

---

## 5. Prerequisites and hardware

**Software:** Windows 10/11 (developed and tested on Windows; other systems are
untested), Python **3.11** (3.12+ can fail to install some pinned packages),
Git, and an editor such as VS Code. No WSL needed.

| | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 6+ cores |
| RAM | 8 GB | 16 GB |
| GPU | none (CPU works) | NVIDIA, 6 GB+ VRAM |
| Free disk | 10 GB | 15 GB |

Rough memory use: loading DDColor needs about 2-3 GB of RAM at its peak. The
GPU code path is implemented but has **not yet been tested on real GPU
hardware** (development so far used a CPU-only machine).

---

## 6. Installation

Run these in **Windows PowerShell**.

```powershell
# 1. Get the code (replace YOUR-USERNAME)
git clone https://github.com/YOUR-USERNAME/photo-restorer.git
cd photo-restorer

# 2. Create and activate a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation says scripts are disabled, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

```powershell
# 3. Install PyTorch FIRST (pick ONE)
#    CPU only:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
#    NVIDIA GPU:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 4. Install the other packages
pip install -r requirements.txt

# 5. Check the environment
python scripts\check_environment.py
```

A GPU **WARN** on a computer without an NVIDIA GPU is expected. Only **FAIL**
lines matter.

The DDColor model code is already included in `src/vendor/ddcolor/` when you
clone the repository. If that folder is ever missing, run
`python scripts\setup_ddcolor_code.py`.

---

## 7. Model weights

Weights are **not** stored in Git (they are large and are not source code).
Download them once:

```powershell
python scripts\download_weights.py            # download everything missing
python scripts\download_weights.py --list     # show present / missing
python scripts\download_weights.py --verify   # check SHA-256 checksums
```

Files are saved to `models/<name>/`:

| Key | File | Size | Source | Checksum | Used by |
|---|---|---|---|---|---|
| `ddcolor` | `ddcolor/pytorch_model.pt` | about 911 MB | ModelScope (DDColor authors' official host) | none known | Module 9 |
| `backup-model` | `colorizer_backup/colorization_release_v2.caffemodel` | about 123 MB | UC Berkeley (authors' server, plain http) | none known | Module 10 |
| `backup-prototxt` | `colorizer_backup/colorization_deploy_v2.prototxt` | about 10 KB | GitHub (richzhang/colorization) | yes | Module 10 |
| `backup-points` | `colorizer_backup/pts_in_hull.npy` | about 5 KB | GitHub (richzhang/colorization) | yes | Module 10 |
| `gfpgan-v1.4` | `gfpgan/GFPGANv1.4.pth` | about 333 MB | GitHub release | yes | Module 11 (planned) |
| `facexlib-detection` | `facexlib/detection_Resnet50_Final.pth` | about 104 MB | GitHub release | yes | Module 11 (planned) |
| `facexlib-parsing` | `facexlib/parsing_parsenet.pth` | about 85 MB | GitHub release | yes | Module 11 (planned) |
| `realesrgan-x4plus` | `realesrgan/RealESRGAN_x4plus.pth` | about 64 MB | GitHub release | yes | Module 12 (planned) |
| `realesrgan-general-x4v3` | `realesrgan/realesr-general-x4v3.pth` | about 5 MB | GitHub release | yes | Module 12 (planned) |

Notes:
- ModelScope's servers are in China and downloads can be slow. If the automatic
  download fails, the script prints manual browser steps.
- The `ddcolor` and `backup-model` files have no known checksum, so their
  integrity is only checked by size and by the loader (which rejects damaged or
  mismatched files with a clear error).
- No weights are downloaded from Hugging Face, by design.

---

## 8. Running and testing what exists

There is no website yet. Each module has a built-in self-test, and several can
process a real image. Run everything from the project root with the virtual
environment active, using `python -m` (not `python src\file.py`).

### Self-tests

| Command | Needs model weights? |
|---|---|
| `python scripts\check_environment.py` | no |
| `python -m src.config` | no |
| `python -m src.logging_setup` | no |
| `python -m src.image_io` | no |
| `python -m src.analysis` | no |
| `python -m src.preprocessing` | no |
| `python -m src.device` | no |
| `python -m src.classical_restore` | no |
| `python -m src.colorize` | no (uses a tiny model and stand-ins) |
| `python -m src.colorize_backup` | no (uses stand-ins) |

Each self-test ends with `all checks passed` when healthy.

### Try the building blocks on your own photo

Put a photo in `samples\` (for example `samples\test_1.jpg`), then:

```powershell
python -m src.image_io samples\test_1.jpg          # load + validate, print details
python -m src.analysis samples\test_1.jpg          # colour type, noise, sharpness, faces
python -m src.preprocessing samples\test_1.jpg     # saves outputs\preview_working.png and preview_gray.png
python -m src.classical_restore samples\test_1.jpg # saves before / result / scratch mask to outputs\
python -m src.colorize outputs\preview_gray.png    # DDColor (needs the ddcolor weights)
python -m src.colorize_backup outputs\preview_gray.png          # backup colorizer
python -m src.colorize_backup outputs\preview_gray.png --chain  # DDColor, falling back to the backup
```

Add `--saturation 0.8` or `--saturation 1.3` to the colorization commands to
make colours more muted or more vivid (range 0.0 to 2.0).

Results are written to `outputs\` (ignored by Git). `outputs\preview_gray.png`
is a grayscale version of your photo, which makes a handy test because you can
compare the colorized result against the real original.

---

## 9. How the implemented modules work

**`image_io.py`** - The only way images enter the system. Checks size on disk
before reading, reads only the header to inspect dimensions before decoding,
verifies the real format from file content (not the filename), and returns a
plain 8-bit RGB image plus information about the original (mode, format,
whether transparency was flattened).

**`analysis.py`** - Converts the photo to Lab colour space and measures
*chroma* (how colourful pixels are) and *hue concentration* (whether all colour
points the same direction, as in sepia). That gives three kinds: `grayscale`,
`tinted_monochrome`, `color`. Also estimates noise (Immerkaer method),
sharpness (Laplacian variance), contrast range, and a face hint from OpenCV's
Haar detector. All thresholds are starting guesses to be tuned in Module 16.

**`preprocessing.py`** - Explicit, tested conversions between Pillow, RGB and
BGR arrays, a working-size limit (default longest side 1024 px, never
enlarges), and grayscale preparation.

**`device.py`** - Chooses CUDA if it genuinely works (a small test calculation
is run), otherwise CPU. Can be forced with an environment variable. Recommends
tile sizes and working sizes by GPU memory, and recognises out-of-memory errors.

**`classical_restore.py`** - Denoises first (so grain isn't mistaken for
scratches), then detects scratches: pixels that differ strongly from a
median-filtered copy, kept only if they form a straight run of at least 25 px,
are narrow, and are *isolated* (smooth surroundings, unlike hair or foam).
Detected pixels are filled by inpainting. If more than 5% of the photo is
flagged, repair is skipped to avoid damaging the image.

**`colorize.py`** - Splits the photo into lightness (L) and colour (a, b).
DDColor sees a 512 x 512 copy and predicts only a and b; those are stretched
back to the photo's size and combined with the original full-resolution L, so
brightness and detail are untouched and runtime does not grow with photo size.
Weights are loaded with PyTorch's `weights_only` mode where possible, and the
loader **verifies** that the checkpoint fits the model: DDColor's own code
loads with `strict=False`, which would silently produce garbage colours from a
mismatched file.

**`colorize_backup.py`** - The Zhang et al. (2016) network run through OpenCV
(224 x 224 input, no PyTorch needed). `ColorizerChain` tries DDColor first and
falls back to this on a colorization error, returning a plain-English note
explaining why. Errors caused by bugs (such as invalid arguments) are
deliberately *not* hidden by the fallback.

**`scripts/setup_ddcolor_code.py`** - DDColor's repository bundles its own copy
of a package named `basicsr`, which would clash with the real `basicsr`
needed by GFPGAN and Real-ESRGAN. The model only needs 5 small files, so
this script fetches exactly those from a pinned commit, verifies SHA-256
checksums, and rewrites 4 import lines to local ones. The Apache 2.0 license
and a NOTICE listing the changes are stored alongside in `src/vendor/ddcolor/`.

---

## 10. Configuration

Defaults live in `src/config.py`. A few can be overridden with environment
variables (PowerShell example: `$env:PHOTO_MAX_UPLOAD_MB = "20"`).

| Variable | Default | Meaning |
|---|---|---|
| `PHOTO_MAX_UPLOAD_MB` | 10 | Largest accepted file |
| `PHOTO_MIN_IMAGE_SIDE` | 64 | Smallest accepted image side (pixels) |
| `PHOTO_MAX_IMAGE_PIXELS` | 25,000,000 | Largest accepted image (about 5000 x 5000) |
| `PHOTO_LOG_LEVEL` | INFO | DEBUG, INFO, WARNING or ERROR |
| `PHOTO_DEVICE` | auto | `auto`, `cpu` or `cuda` |

Logs go to the console and to `logs\app.log` (rotating, about 1 MB each).

---

## 11. Project structure

Files that exist today:

```
photo-restorer/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- scripts/
|   |-- check_environment.py      # environment health check
|   |-- download_weights.py       # downloads and verifies model weights
|   `-- setup_ddcolor_code.py     # fetches DDColor model code (pinned, checksummed)
|-- src/
|   |-- config.py                 # paths, limits, settings
|   |-- logging_setup.py
|   |-- image_io.py               # safe loading and validation
|   |-- analysis.py               # grayscale detection, noise, sharpness, ...
|   |-- preprocessing.py          # format conversions, size limit
|   |-- device.py                 # GPU/CPU selection
|   |-- classical_restore.py      # denoise + scratch repair
|   |-- colorize.py               # DDColor (main colorizer)
|   |-- colorize_backup.py        # backup colorizer + fallback chain
|   `-- vendor/ddcolor/           # vendored DDColor model code + LICENSE + NOTICE
|-- samples/                      # your test images (small ones may be committed)
|-- models/                       # downloaded weights (NOT in Git)
|-- outputs/                      # generated images (NOT in Git)
|-- uploads/  logs/               # created at runtime (NOT in Git)
```

Planned additions: `src/face_restore.py`, `src/super_resolution.py`,
`src/postprocess.py`, `src/pipeline.py`, `src/api/`, `frontend/`, `tests/`,
`notebooks/`, `run.py`, `.env.example`.

---

## 12. Models and licenses

| Component | Role | Code license | Notes |
|---|---|---|---|
| [DDColor](https://github.com/piddnad/DDColor) (ICCV 2023) | Main colorizer | Apache 2.0 | Weights hosted by the authors on ModelScope. The weights' own license was **not independently verified**; check the ModelScope model page before redistributing. |
| [Colorful Image Colorization](https://github.com/richzhang/colorization) (Zhang et al., ECCV 2016) | Backup colorizer | BSD 2-Clause | Weights trained on ImageNet; check ImageNet terms for your use case. |
| GFPGAN v1.4 | Face restoration *(planned)* | Apache 2.0 | License to be re-verified when integrated |
| Real-ESRGAN | Super-resolution *(planned)* | BSD 3-Clause | License to be re-verified when integrated |
| facexlib | Face detection *(planned)* | MIT | License to be re-verified when integrated |
| OpenCV, Pillow, NumPy, PyTorch | Libraries | Their respective open-source licenses | |

**License of this project:** not chosen yet. Decide before publishing.

---

## 13. Known limitations

- **No end-to-end tool or website yet.** Only individual modules exist.
- **No quality evaluation yet.** There are no benchmark numbers. The scratch
  detector and analysis thresholds are untuned starting guesses.
- **Scratch repair is imperfect.** It handles thin, isolated scratches only. It
  cannot rebuild torn or missing regions, and on sharp, detailed photos it
  wrongly flags roughly 1-3% of pixels (measured on four clean sample photos).
  Repairing those pixels changes the picture only slightly, but it is why
  scratch repair will be a user option.
- **Colorization is a guess.** Colours are plausible, not historically
  accurate. Uniforms, flags and clothing can come out wrong. The backup
  colorizer is lower quality, usually more muted, and looks at only 224 x 224
  pixels.
- **Colorization re-colours colour photos.** Only lightness is used, so
  running it on a colour photo discards the original colours.
- **Face hint is crude.** The Haar detector in `analysis.py` only finds
  frontal faces and is just a hint. Face restoration will use a better detector.
- **Model hosting risks.** The main model comes from ModelScope (slow outside
  China) and the backup weights from an old university server over plain http
  with no known checksum.
- **Working size.** Photos are shrunk to a 1024 px longest side before
  processing on CPU.
- **Platform.** Developed and tested on Windows with a CPU-only machine. The
  GPU path is untested on real hardware.

---

## 14. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `No module named 'src'` | Wrong folder, or ran `python src\file.py` | `cd` to the project root and use `python -m src.<module>` |
| `No module named 'torch'` | PyTorch not installed | Do step 3 of [Installation](#6-installation) |
| `running scripts is disabled` | PowerShell policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Could not find a version that satisfies torch==2.1.2` | Python is not 3.11 | Recreate the venv with `py -3.11 -m venv .venv` |
| `(.venv)` missing from the prompt | New terminal window | `.venv\Scripts\Activate.ps1` |
| `The DDColor model code is missing` | `src\vendor\ddcolor` missing | `python scripts\setup_ddcolor_code.py` |
| `ModelWeightsMissingError` | Weights not downloaded | `python scripts\download_weights.py` |
| `weights file is missing N expected entries` | Weights file does not match the model code | Delete `models\ddcolor\pytorch_model.pt`, download again, and open an issue if it persists |
| `could not be read ... may be damaged` | Incomplete download | Delete the file and download it again |
| `[MISMATCH]` in `--verify` | Corrupted file | Delete it and download again |
| `Not enough memory` | Less than about 3 GB free RAM | Close other programs |
| Slow colorization | CPU-only processing | Expected; the first run also loads about 900 MB |
| Download from ModelScope or Berkeley fails | Server slow or blocked | Use the manual steps the script prints |

---

## 15. Roadmap

Done: Modules 1-10 (see [Current status](#1-current-status)).

| # | Planned module |
|---|---|
| 11 | Face restoration (GFPGAN v1.4) |
| 12 | Super-resolution (Real-ESRGAN, tiled) |
| 13 | Post-processing |
| 14 | Complete pipeline orchestrator |
| 15 | Command-line runner for the whole pipeline |
| 16 | Evaluation notebooks and metrics (PSNR/SSIM), threshold tuning |
| 17 | FastAPI backend (upload, job, status, download) |
| 18-19 | Frontend: upload, progress, before/after slider, download |
| 20 | Error handling, logging, temporary-file cleanup |
| 21 | Tests (pytest) |
| 22 | Performance tuning |
| 23 | Launch script and packaging |
| 24 | Final README |
| 25 | Optional free deployment (a fully free host is unlikely to have enough RAM; local use is the target) |

Possible stretch goal after everything works: a small transfer-learning
damage classifier. It would need labelled data and is not required for the app.

---

## 16. Git and GitHub notes

**Commit:** source code, `requirements.txt`, this README, small sample images
(public-domain or your own only), and `src/vendor/ddcolor/` including its
`LICENSE` and `NOTICE.txt`.

**Never commit:** `.venv/`, model weights (`models/`, `*.pth`, `*.pt`,
`*.caffemodel`), generated images (`outputs/`), uploads, logs, `.env` files,
caches. `.gitignore` already excludes these. Do not commit photos of private
individuals to a public repository.

---

## 17. Credits

- **DDColor:** Xiaoyang Kang et al., "DDColor: Towards Photo-Realistic Image
  Colorization via Dual Decoders", ICCV 2023.
- **Colorful Image Colorization:** Richard Zhang, Phillip Isola, Alexei A. Efros,
  ECCV 2016.
- **Planned components:** GFPGAN (Tencent ARC), Real-ESRGAN (xinntao),
  facexlib (xinntao).
