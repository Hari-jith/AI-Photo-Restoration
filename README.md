# AI Old Photo Restoration & Colorization

> **Work in progress.** Milestone 1 is reached: the complete restoration
> pipeline runs from the command line. **There is no website yet.** This README
> describes the project as built through Module 14 of 25 and separates what is
> confirmed from what is only planned. Last updated: October 2026.

A locally run, zero-cost tool that restores old photographs with open-source
models: it removes grain and scratches, colorizes black-and-white photos,
restores faces, enlarges the picture, and applies final touch-ups. No paid
services, no cloud APIs, and no Hugging Face: every model comes from an
official GitHub repository or another official project host.

The project is also a learning build. It is developed one small, tested module
at a time, and the reasoning behind each decision is kept in the code comments.

---

## Contents

1. [Status at a glance](#1-status-at-a-glance)
2. [How the pipeline works](#2-how-the-pipeline-works)
3. [Quick start](#3-quick-start)
4. [Using it](#4-using-it)
5. [What was measured, and what it decided](#5-what-was-measured-and-what-it-decided)
6. [Architecture and project structure](#6-architecture-and-project-structure)
7. [Models and licenses](#7-models-and-licenses)
8. [Known limitations](#8-known-limitations)
9. [Configuration](#9-configuration)
10. [Troubleshooting](#10-troubleshooting)
11. [Roadmap](#11-roadmap)
12. [Git and GitHub notes](#12-git-and-github-notes)
13. [Credits and license](#13-credits-and-license)

---

## 1. Status at a glance

**Works today (command line):** photo validation, analysis, grain and scratch
repair, colorization (with an automatic fallback model), face restoration,
upscaling, post-processing, and one command that runs all of it.

**Not built yet:** the website and its API, an automated pytest suite,
quantitative quality evaluation, packaging, deployment.

**Evaluated on very little data.** All visual judgements so far come from one
heavily cracked portrait. There are no PSNR/SSIM numbers and no benchmark set yet.

| # | Module | Main files | Confirmed on the author's machine |
|---|---|---|---|
| 1 | Environment check | `scripts/check_environment.py` | Yes |
| 2 | Git setup | `.gitignore` | n/a |
| 3 | Config and logging | `src/config.py`, `src/logging_setup.py` | Yes |
| 4 | Safe image loading and validation | `src/image_io.py` | Yes |
| 5 | Photo analysis | `src/analysis.py` | Yes |
| 6 | Preprocessing | `src/preprocessing.py` | Yes |
| 7 | Device manager, weight downloader | `src/device.py`, `scripts/download_weights.py` | Yes |
| 8 | Grain and scratch repair (OpenCV) | `src/classical_restore.py` | Yes |
| 9 | Colorization, DDColor | `src/colorize.py`, `scripts/setup_ddcolor_code.py` | Yes |
| 10 | Backup colorizer and fallback chain | `src/colorize_backup.py`, `scripts/setup_backup_colorizer.py` | Yes |
| 11 | Face restoration, GFPGAN | `src/face_restore.py`, `scripts/setup_face_code.py` | Yes |
| 12 | Upscaling, Real-ESRGAN | `src/super_resolution.py`, `scripts/setup_upscaler_code.py` | Yes |
| 13 | Post-processing | `src/postprocess.py` | Yes |
| 14 | Complete pipeline | `src/pipeline.py` | Yes |
| 15 | *(skipped: the command line is built into `pipeline.py`)* | | |
| 16-25 | Evaluation, website, tests, packaging | | Planned |

Every module has a built-in self-test that needs no model weights. The
"Confirmed" column records whether the author has run it on real hardware.

---

## 2. How the pipeline works

```
 prepare -> analyze -> repair -> colorize -> upscale -> faces -> finish
```

| Step | What it does | Default |
|---|---|---|
| prepare | Shrinks very large photos to a working size (never enlarges) | 1024 px longest side on CPU |
| analyze | Measures colour type, noise, contrast, sharpness | always |
| repair | Removes grain and thin scratches (OpenCV) | `light` repair, grain removal automatic |
| colorize | DDColor; if it fails, a smaller backup model | `auto`: only if the photo is detected as black-and-white |
| upscale | Real-ESRGAN enlargement | 2x, small fast model |
| faces | GFPGAN face restoration, pasted onto the **upscaled** picture | on, strength 0.6 |
| finish | Contrast restoration for faded photos; optional film grain | contrast `auto`, grain off |

**Why this order** (measured on one real portrait, details in section 5):
faces are restored *after* upscaling because restoring them first made the
small upscaler add a glowing eye and plastic skin; repair comes *before*
colorizing and upscaling because both would otherwise treat cracks as picture
content (the upscaler makes them sharper).

**Failures do not stop a run by default.** If a step fails (for example a
missing model file), the picture continues without it, the failure is listed in
the step table, and the program exits with code 1.

---

## 3. Quick start

**System requirements:** Windows 10/11 (developed and tested there; other
systems are untested), Python **3.11** (newer versions can fail to install some
pinned packages), Git. No WSL needed.

| | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 6+ cores |
| RAM | 8 GB | 16 GB |
| GPU | none (CPU works) | NVIDIA, 6 GB+ VRAM |
| Free disk | 10 GB | 15 GB |

The model downloads total about 1.6 GB. Loading DDColor needs roughly 2-3 GB of
RAM at its peak. The GPU code path is implemented but has **not been tested on
real GPU hardware**.

Run in **Windows PowerShell**:

```powershell
# 1. Get the code (replace YOUR-USERNAME)
git clone https://github.com/YOUR-USERNAME/photo-restorer.git
cd photo-restorer

# 2. Create and activate a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3. Install PyTorch FIRST (pick ONE)
#    CPU only:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
#    NVIDIA GPU:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 4. Install the other packages and check the environment
pip install -r requirements.txt
python scripts\check_environment.py

# 5. Fetch the model code (each script is safe to re-run; scripts whose code is
#    already present do nothing). GFPGAN's code is NOT stored in this repository
#    (see section 7), so setup_face_code.py always has work to do after a clone.
python scripts\setup_ddcolor_code.py
python scripts\setup_face_code.py
python scripts\setup_upscaler_code.py
python scripts\setup_backup_colorizer.py --skip-weights

# 6. Download and verify the model weights (about 1.6 GB)
python scripts\download_weights.py
python scripts\download_weights.py --verify
```

If PowerShell says scripts are disabled, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and activate
again. A GPU warning on a computer without an NVIDIA GPU is expected; only
`FAIL` lines matter.

Put a photo in `samples\` and restore it:

```powershell
python -m src.pipeline samples\my_photo.jpg
```

Results go to `outputs\`: `restored_my_photo.png` and
`restored_my_photo_comparison.png` (before on the left, after on the right).

---

## 4. Using it

### 4.1 The full pipeline

```powershell
python -m src.pipeline samples\my_photo.jpg --colorize always --upscale 2
```

| Option | Values (default first) | Meaning |
|---|---|---|
| `--repair` | `light`, `off`, `strong` | Scratch repair strength |
| `--denoise` | `auto`, `off` | Grain removal (only when grain is detected) |
| `--colorize` | `auto`, `always`, `never` | Whether to add colour |
| `--saturation` | `1.0` (0.0 to 2.0) | Colour strength |
| `--upscale` | `2` (1 = off, up to 4) | Enlargement factor |
| `--upscale-model` | `auto`, `general`, `x4plus` | `general` is fast; `x4plus` looks better but is much slower on CPU |
| `--faces` | `on`, `off` | Face restoration |
| `--face-strength` | `0.6` (0.0 to 1.0) | 1.0 = fully restored faces, 0.0 = original |
| `--contrast` | `auto`, `on`, `off` | Contrast restoration (`auto`: only for faded photos) |
| `--grain` | `0.0` (0.0 to 1.0) | Cosmetic film grain; about 0.4 matches typical old-photo grain |
| `--keep-steps` | flag | Also save the picture after each step |

The run prints a live percentage and then a table with the status and time of
every step, followed by plain-English notes. A black-and-white photo is
colorized automatically. A photo that is a toned print (very little colour
spread over different tints) is **not**, and you will get a note suggesting
`--colorize always`.

### 4.2 Single stages

Each stage can be run alone, which is useful for judging what each one does:

| Command | What it does |
|---|---|
| `python -m src.image_io FILE` | Load and validate, print the file's details |
| `python -m src.analysis FILE` | Colour type, noise, sharpness, contrast, face hint |
| `python -m src.preprocessing FILE` | Working-size preview and a grayscale copy |
| `python -m src.classical_restore FILE` | Grain and scratch repair; saves a mask of repaired pixels |
| `python -m src.colorize FILE [--saturation X]` | DDColor |
| `python -m src.colorize_backup FILE [--chain]` | Backup colorizer, or DDColor with automatic fallback |
| `python -m src.face_restore FILE [--strength X]` | GFPGAN faces |
| `python -m src.super_resolution FILE [--scale X] [--model M]` | Real-ESRGAN |
| `python -m src.postprocess FILE [--grain X] [--simulate-fade]` | Contrast and grain |

### 4.3 Self-tests (no model weights needed)

```powershell
python scripts\check_environment.py
python -m src.config
python -m src.logging_setup
python -m src.image_io
python -m src.analysis
python -m src.preprocessing
python -m src.device
python -m src.classical_restore
python -m src.colorize
python -m src.colorize_backup
python -m src.face_restore
python -m src.super_resolution
python -m src.postprocess
python -m src.pipeline
```

Each ends with `all checks passed` when healthy.

---

## 5. What was measured, and what it decided

### 5.1 Timings on the author's CPU-only Windows machine

*AMD Ryzen 5 7530U RAM - 16GB* Measured per stage on a 459x668 photo:

| Stage | Time | Source |
|---|---|---|
| DDColor: load model (once per process) | 12-16 s | measured |
| DDColor: colorize | 9.5-10.7 s | measured |
| Backup colorizer: load / colorize | 0.3 s / 1.3 s | measured |
| Face models: load (once per process) | 2.6-3.8 s | measured |
| Face restoration (1 face) | 8.3-9.7 s | measured |
| Upscaling, small model, 2x (918x1336) | 9.6 s | measured |

**The full pipeline's total time on this machine has not been recorded yet.**
Adding the stages gives an estimate of roughly 35-50 s per photo once the models
are loaded, plus about 15-20 s for loading them in a fresh process.

### 5.2 Design decisions backed by measurement (one portrait, development sandbox)

| Question | What was observed | Decision |
|---|---|---|
| Faces before or after upscaling? | Faces first, then the small upscaler: a glowing eye and plastic skin. Upscale first, then paste the 512 px GFPGAN face: natural skin | Upscale first, then paste faces |
| Which upscaler by default? | The pasted face is identical with either model. The best model only improves fabric and hands, at roughly 15-20x the compute (single CPU thread: about 5 s vs about 66 s for a 0.1 megapixel photo) | Small model by default, best model optional |
| Default repair level? | `low` removes the most prominent cracks and never touched the face. The `medium` level hit the safety cap and repaired nothing | "light" = `low`, "strong" = `high`; `medium` is not offered |
| Face strength? | At 1.0 one eye received an invented different colour; 0.6 looked consistent and still like the same person | Default 0.6 |
| Post-processing? | Contrast was already good at every stage (about 200 of 255). Upscaling removed most grain (estimated grain level 4.8 down to 0.6-0.9) | Contrast fix only for faded photos; grain optional and off by default |

### 5.3 Engineering lessons

- **No `basicsr` install.** It is hard to install and import on many setups. Only the
  network definitions needed are copied from the official repositories, pinned
  to a commit and checked against SHA-256 checksums.
- **Checkpoints are verified, not trusted.** DDColor's own loader uses
  `strict=False`, which would silently give garbage colours from a mismatched file.
  Ours raises a clear error.
- **Upstream pitfalls avoided:** Real-ESRGAN's tiling leaves a black square when a tile fails
  (ours raises an error); GFPGAN v1.4's `weight` option has no effect (we verified
  identical output, so we implemented a real blend instead); unclipped model output
  wraps around when converted to bytes (all stages clip first).
- **Dead download links happen.** The original backup model's address now answers 404; it
  was replaced by the authors' current PyTorch model, verified by its full SHA-256.
- **Self-tests can lie.** Several checks were found to pass vacuously during development
  and were rewritten to test what they claim.

---

## 6. Architecture and project structure

Planned shape: the browser talks to a small Python server, which runs the
pipeline. **Only the pipeline exists today.**

```
Browser (HTML/CSS/JS)                      [planned]
   | upload + options
   v
FastAPI server -> background job           [planned]
   |
   v
src/pipeline.py  (progress reporting, cancellation, graceful step failures)   [built]
   |
   v
outputs/  ->  status polling  ->  before/after slider + download               [planned]
```

```
photo-restorer/
|-- README.md  requirements.txt  .gitignore
|-- scripts/
|   |-- check_environment.py        environment health check
|   |-- download_weights.py         download and verify weights (checksums)
|   |-- setup_ddcolor_code.py       DDColor model code
|   |-- setup_face_code.py          GFPGAN (not committed) + facexlib code
|   |-- setup_upscaler_code.py      Real-ESRGAN network definitions
|   `-- setup_backup_colorizer.py   backup colorizer code and weights
|-- src/
|   |-- config.py  logging_setup.py  image_io.py  analysis.py  preprocessing.py
|   |-- device.py                   CPU/GPU selection, tile and size recommendations
|   |-- classical_restore.py        grain and scratch repair
|   |-- colorize.py                 DDColor
|   |-- colorize_backup.py          backup colorizer + fallback chain
|   |-- face_restore.py             GFPGAN
|   |-- super_resolution.py         Real-ESRGAN with our own error-safe tiling
|   |-- postprocess.py              contrast and grain
|   |-- pipeline.py                 the complete pipeline and its command line
|   `-- vendor/
|       |-- ddcolor/  siggraph17/  facexlib/  realesrgan/   committed (permissive licenses)
|       `-- gfpgan/                                         NOT committed (see section 7)
|-- samples/   your test images
|-- models/    downloaded weights  (NOT in Git)
|-- outputs/   results             (NOT in Git)
`-- uploads/  logs/                created at runtime (NOT in Git)
```

Planned additions: `src/api/`, `frontend/`, `tests/`, `notebooks/`, `run.py`.

---

## 7. Models and licenses

| Component | Role | Code license | Notes |
|---|---|---|---|
| [DDColor](https://github.com/piddnad/DDColor) (ICCV 2023) | Main colorizer | Apache 2.0 | Weights hosted by the authors on ModelScope; the weights' own license was **not independently verified**. No checksum is known for this file. |
| [Colorization, Zhang et al.](https://github.com/richzhang/colorization) (SIGGRAPH 2017) | Backup colorizer | BSD 2-Clause | Weights trained on ImageNet, hosted by the authors on S3, pinned by full SHA-256 |
| [GFPGAN](https://github.com/TencentARC/GFPGAN) v1.4 | Face restoration | Apache 2.0 headline, **with third-party exceptions** | Its LICENSE lists StyleGAN2-derived code under NVIDIA's license (research or evaluation use only) and DFDNet-derived code under CC BY-NC-SA 4.0. **Treat GFPGAN and its weights as non-commercial.** Its code is deliberately not committed here. |
| [facexlib](https://github.com/xinntao/facexlib) | Face detection and alignment | MIT | A small subset is vendored |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Upscaling | BSD 3-Clause | `RRDBNet` comes from BasicSR (Apache 2.0) |
| OpenCV, Pillow, NumPy, PyTorch | Libraries | their own open-source licenses | |

Because one component is non-commercial, treat the whole combination as
**for personal, educational and research use** unless you obtain legal advice.

**Hosting notes:** ModelScope servers are in China and downloads can be slow.
No weights come from Hugging Face.

---

## 8. Known limitations

- **No website or API yet**, and no automated pytest suite.
- **Very small evaluation.** One heavily damaged portrait; no quantitative
  metrics. Thresholds (scratch detection, colour classification, contrast) are
  untuned starting values.
- **Repair is partial.** On the test photo it removed the longest cracks but
  left the fine web of cracks and most cracks on the clothing. Filled areas look
  smooth, there is orange speckle residue along some crack edges, and one dark
  blotch appeared. A hypothesis (repair mask margin too small) is untested.
- **The repair safety cap is crude.** It skips repair when more than 5% of the
  picture is flagged. The test photo was flagged at 4.7%, just under the cap, so a
  slightly more damaged photo would skip repair (the pipeline reports this).
- **Toned prints are not auto-detected as black-and-white.** The test photo was
  classified as colour (very little colour, but two different tints). Use
  `--colorize always`.
- **Colorization is a guess.** Colours are plausible, not historically accurate.
  The backup colorizer is less colourful (on one photo, mean colourfulness 7.7
  vs 16.0 for DDColor).
- **Face restoration invents detail.** It can change features such as eye colour
  (seen at strength 1.0), and it smooths skin.
- **Upscaling enlarges damage too.** The small model gives a smooth, airbrushed
  look; the best model is much slower on CPU.
- **Contrast restoration** was tested only on an artificially faded copy. Real
  fading (yellowing, uneven dye loss) may need more. Film grain is cosmetic.
- **Untested:** the GPU path, non-Windows systems, real faded photos, photos with
  many faces, inputs near the size limits.
- **Limits:** photos are shrunk to a 1024 px working size on CPU; results are
  capped at 4096 px; upscaling is limited to 4x.

---

## 9. Configuration

Defaults live in `src/config.py`. These can be overridden with environment
variables (PowerShell example: `$env:PHOTO_MAX_UPLOAD_MB = "20"`):

| Variable | Default | Meaning |
|---|---|---|
| `PHOTO_MAX_UPLOAD_MB` | 10 | Largest accepted file |
| `PHOTO_MIN_IMAGE_SIDE` | 64 | Smallest accepted image side (pixels) |
| `PHOTO_MAX_IMAGE_PIXELS` | 25,000,000 | Largest accepted image (about 5000 x 5000) |
| `PHOTO_LOG_LEVEL` | INFO | DEBUG, INFO, WARNING or ERROR |
| `PHOTO_DEVICE` | auto | `auto`, `cpu` or `cuda` |

Logs go to the console and to `logs\app.log` (rotating, about 1 MB each).

---

## 10. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `No module named 'src'` | Wrong folder, or `python src\file.py` | Run from the project root with `python -m src.<module>` |
| `No module named 'torch'` | PyTorch not installed | Step 3 of the quick start |
| `Could not find a version that satisfies torch==2.1.2` | Python is not 3.11 | Recreate the venv with `py -3.11 -m venv .venv` |
| `(.venv)` missing from the prompt | New terminal | `.venv\Scripts\Activate.ps1` |
| `The ... code is missing` | A setup script was not run | Run the `setup_*` script named in the message |
| `... weights were not found` | Weights not downloaded | `python scripts\download_weights.py` |
| `weights file is missing N expected entries` | Weights file does not match the model code | Delete that file, download again, and open an issue if it persists |
| `[MISMATCH]` in `--verify` | Corrupted file | Delete it and download again |
| `CHECKSUM MISMATCH` in a setup script | Upstream file differs from the tested version | Nothing is installed; open an issue |
| A step shows `failed` in the table | Missing weights or out of memory | Read the NOTE under the table; the run still produced a result |
| `Not enough memory` | Less than about 3 GB free RAM | Close other programs, or use a smaller image |
| ModelScope download is slow or fails | Server in China | Use the manual browser steps the script prints |
| Slow runs | CPU-only processing | Expected; avoid `--upscale-model x4plus` on CPU |

---

## 11. Roadmap

| # | Planned work |
|---|---|
| 16 | Evaluation on a set of varied photos (PSNR/SSIM on synthetic damage, side-by-side review); tuning of the scratch detector, repair cap and colour classifier |
| 17 | FastAPI backend: upload, job, status, cancel, download |
| 18-19 | Frontend: drag-and-drop, options, progress, before/after slider, download |
| 20 | Error handling, logging and temporary-file cleanup for the web app |
| 21 | pytest suite |
| 22 | Performance tuning |
| 23 | Launch script and packaging |
| 24 | Final README with example images |
| 25 | Optional free deployment (a fully free host is unlikely to have enough RAM; local use is the target) |

Possible stretch goal: a learned scratch detector, which would address the
repair limits above. Its license and dependencies would need checking first.

---

## 12. Git and GitHub notes

**Commit:** source code, `requirements.txt`, this README, small sample images
(public domain or your own only), and the vendored folders under `src/vendor/`
that are committed (`ddcolor`, `siggraph17`, `facexlib`, `realesrgan`).

**Never commit:** `.venv/`, `models/`, `outputs/`, uploads, logs, `.env` files,
and `src/vendor/gfpgan/` (a `.gitignore` inside it already excludes it). Do not
commit photos of private individuals to a public repository.

```powershell
git add .
git commit -m "Module N: short description"
```

---

## 13. Credits and license

- **DDColor:** Xiaoyang Kang et al., ICCV 2023.
- **Colorization:** Richard Zhang, Phillip Isola, Alexei A. Efros, SIGGRAPH 2017.
- **GFPGAN:** Tencent ARC. **Real-ESRGAN, facexlib:** xinntao. **BasicSR:** XPixelGroup.
