"""
pipeline.py

The complete restoration pipeline. One call runs every step in the order that
our measurements on a real photo showed to work best:

    prepare -> analyze -> repair -> colorize -> upscale -> faces -> finish

  prepare   shrink very large photos to a working size (never enlarges)
  analyze   measure the photo (colour type, noise, contrast, ...)
  repair    remove grain and thin scratches (classical, OpenCV)
  colorize  DDColor, falling back to a simpler backup model if it fails
  upscale   Real-ESRGAN enlargement
  faces     GFPGAN face restoration, pasted onto the UPSCALED picture
  finish    contrast restoration for faded photos, optional film grain

Why this order (measured on one real portrait, see Module 14 notes):
  * Faces AFTER upscaling: restoring faces first and then upscaling the whole
    picture made the small upscaler add a glowing eye and plastic skin. Pasting
    the 512-pixel GFPGAN face onto the already-enlarged picture avoided that.
  * Repair BEFORE colorizing and upscaling: both would otherwise treat cracks
    as picture content (the upscaler makes them sharper).

    from src.pipeline import RestorationPipeline, PipelineOptions

    pipeline = RestorationPipeline()
    result = pipeline.run(image, PipelineOptions(upscale=2.0),
                          progress=lambda fraction, message: print(message))
    result.image, result.steps, result.notes

A step that fails (for example a missing model file) does NOT stop the run by
default: the picture continues without that step and the failure is reported.
Pass on_error="raise" to stop instead.

Run from the command line:
    python -m src.pipeline samples\\test_1.jpg --upscale 2
Self-test (no model weights needed):
    python -m src.pipeline
"""

import argparse
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from src import config
from src.analysis import FADED_COLOR_MEAN_CHROMA, analyze_image
from src.classical_restore import restore_classical
from src.colorize import ColorizationError
from src.device import free_memory, recommended_max_working_side
from src.face_restore import FaceRestorationError
from src.logging_setup import get_logger
from src.postprocess import postprocess
from src.preprocessing import limit_working_size
from src.super_resolution import UpscaleError

logger = get_logger(__name__)

# --- Allowed option values and defaults ----------------------------------------
# "medium" scratch sensitivity is deliberately NOT offered: on our test photo it
# hit the safety cap and silently repaired nothing (see classical_restore.py).
REPAIR_LEVELS = {"off": None, "light": "low", "strong": "high"}
COLORIZE_MODES = ("auto", "always", "never")
ONOFF = ("on", "off")
DENOISE_MODES = ("auto", "off")
MAX_UPSCALE = 4.0
DEFAULT_MAX_OUTPUT_SIDE = 4096     # never make a picture larger than this

# Relative cost of each step, used to turn step progress into overall progress.
STEP_WEIGHTS = {"prepare": 1, "analyze": 1, "repair": 2, "colorize": 6,
                "upscale": 6, "faces": 4, "finish": 1}
STEP_ORDER = list(STEP_WEIGHTS)

KNOWN_STEP_ERRORS = (ColorizationError, FaceRestorationError, UpscaleError)


class PipelineCancelled(Exception):
    """The run was cancelled (via the cancel event) between two steps."""


@dataclass
class PipelineOptions:
    """Everything a user can choose. Invalid values raise ValueError at once."""

    repair: str = "light"              # "off", "light" or "strong" (scratch repair)
    denoise: str = "auto"              # "auto" (only when grain is detected) or "off"
    colorize: str = "auto"             # "auto", "always" or "never"
    saturation: float = 1.0            # colour strength, 0.0-2.0
    upscale: float = 2.0               # 1.0 = no upscaling, otherwise up to 4.0
    upscale_model: str = "auto"        # "auto", "general" (fast) or "x4plus" (best, slow on CPU)
    faces: str = "on"                  # "on" or "off"
    face_strength: float = 0.6         # 1.0 = fully restored faces, 0.0 = original
    contrast: str = "auto"             # "auto", "on" or "off"
    grain: float = 0.0                 # 0.0-1.0, cosmetic film grain
    max_side: int | None = None        # working size limit (None = chosen from hardware)
    max_output_side: int = DEFAULT_MAX_OUTPUT_SIDE

    def __post_init__(self):
        def one_of(name, allowed):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {tuple(allowed)}, got '{getattr(self, name)}'.")

        def in_range(name, low, high):
            value = getattr(self, name)
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}, got {value}.")

        one_of("repair", REPAIR_LEVELS)
        one_of("denoise", DENOISE_MODES)
        one_of("colorize", COLORIZE_MODES)
        one_of("upscale_model", ("auto", "general", "x4plus"))
        one_of("faces", ONOFF)
        one_of("contrast", ("auto", "on", "off"))
        in_range("saturation", 0.0, 2.0)
        in_range("face_strength", 0.0, 1.0)
        in_range("grain", 0.0, 1.0)
        if not (self.upscale == 1.0 or 1.0 < self.upscale <= MAX_UPSCALE):
            raise ValueError(f"upscale must be 1.0 (off) or above 1.0 and at most {MAX_UPSCALE}.")
        if self.max_side is not None and self.max_side < 128:
            raise ValueError("max_side must be at least 128.")
        if self.max_output_side < 256:
            raise ValueError("max_output_side must be at least 256.")


@dataclass
class StepReport:
    name: str
    status: str                 # "done", "skipped" or "failed"
    seconds: float = 0.0
    message: str = ""


@dataclass
class PipelineResult:
    image: Image.Image                  # the final restored picture
    original: Image.Image               # the working-size input (for before/after)
    steps: list
    notes: list                         # plain-English remarks for the user
    analysis: dict
    total_seconds: float
    intermediates: dict = field(default_factory=dict)

    @property
    def failed_steps(self) -> list:
        return [step.name for step in self.steps if step.status == "failed"]

    def table(self) -> str:
        lines = [f" {'Step':<10} {'Status':<8} {'Time':>7}  Details"]
        for s in self.steps:
            lines.append(f" {s.name:<10} {s.status:<8} {s.seconds:>6.1f}s  {s.message}")
        lines.append(f" {'TOTAL':<10} {'':<8} {self.total_seconds:>6.1f}s")
        return "\n".join(lines)


@dataclass
class Components:
    """The heavy parts. Leave a field None to use the real one (loaded lazily)."""

    colorizer: object = None       # .colorize(image, saturation=) -> ChainResult
    face_restorer: object = None   # .restore(image, strength=, background=) -> FaceRestoreResult
    upscaler: object = None        # .upscale(image, scale=, progress=) -> UpscaleResult


class RestorationPipeline:
    def __init__(self, components: Components | None = None):
        self.components = components or Components()
        self._upscalers = {}                  # one real upscaler per model name
        self._creation_lock = threading.Lock()   # two requests must not both load a model

    # -- lazily created real components (injected ones are used as they are) ----
    def _colorizer(self):
        with self._creation_lock:
            if self.components.colorizer is None:
                from src.colorize_backup import ColorizerChain
                self.components.colorizer = ColorizerChain()
            return self.components.colorizer

    def _face_restorer(self):
        with self._creation_lock:
            if self.components.face_restorer is None:
                from src.face_restore import FaceRestorer
                self.components.face_restorer = FaceRestorer()
            return self.components.face_restorer

    def _upscaler(self, model: str):
        with self._creation_lock:
            if self.components.upscaler is not None:
                return self.components.upscaler
            if model not in self._upscalers:           # a later request may ask for another model
                from src.super_resolution import Upscaler
                self._upscalers[model] = Upscaler(model=model)
            return self._upscalers[model]

    # -- main entry point --------------------------------------------------------
    def run(self, image: Image.Image, options: PipelineOptions | None = None,
            progress=None, cancel=None, on_error: str = "continue",
            keep_intermediates: bool = False) -> PipelineResult:
        """Restore one photo.

        Args:
            image: any Pillow image (it is converted to RGB).
            options: what to do (see PipelineOptions).
            progress: optional function progress(fraction 0..1, message).
            cancel: optional threading.Event; if set, the run stops between steps.
            on_error: "continue" (default: skip a failed step and go on) or "raise".
            keep_intermediates: keep the picture after each step in the result.
        """
        if on_error not in ("continue", "raise"):
            raise ValueError("on_error must be 'continue' or 'raise'.")
        options = options or PipelineOptions()
        run = _Run(self, options, progress, cancel, on_error, keep_intermediates)
        return run.execute(image)


class _Run:
    """State of one pipeline run (kept separate so the pipeline object is reusable)."""

    def __init__(self, pipeline, options, progress, cancel, on_error, keep):
        self.pipeline, self.options = pipeline, options
        self.progress, self.cancel, self.on_error = progress, cancel, on_error
        self.keep = keep
        self.steps, self.notes, self.intermediates = [], [], {}
        self.total_weight = sum(STEP_WEIGHTS.values())
        self.done_weight = 0.0

    # -- reporting helpers -----------------------------------------------------
    def _report(self, within_step: float, message: str, step: str) -> None:
        if self.progress is not None:
            fraction = (self.done_weight + within_step * STEP_WEIGHTS[step]) / self.total_weight
            self.progress(min(max(fraction, 0.0), 1.0), message)

    def _finish_step(self, name, status, started, message=""):
        self.steps.append(StepReport(name, status, time.time() - started, message))
        self.done_weight += STEP_WEIGHTS[name]
        self._report(0.0, f"{name}: {status}" + (f" ({message})" if message else ""), name)
        logger.info("Step %-8s %-7s %5.1fs %s", name, status, time.time() - started, message)

    def _check_cancel(self):
        if self.cancel is not None and self.cancel.is_set():
            raise PipelineCancelled("The restoration was cancelled.")

    def _attempt(self, name, started, work):
        """Run `work()`. Known step errors are reported (or raised, per on_error)."""
        try:
            return work()
        except KNOWN_STEP_ERRORS as error:
            if self.on_error == "raise":
                raise
            self.notes.append(f"The '{name}' step could not be completed and was skipped: {error}")
            self._finish_step(name, "failed", started, str(error)[:200])
            return None

    # -- the whole run ---------------------------------------------------------
    def execute(self, image: Image.Image) -> PipelineResult:
        total_start = time.time()
        options = self.options
        keep = self.keep

        # ---- prepare ----
        self._check_cancel(); started = time.time()
        self._report(0.0, "Preparing the photo", "prepare")
        rgb = image.convert("RGB")
        max_side = options.max_side or recommended_max_working_side()
        resized = limit_working_size(rgb, max_side)
        working, message = resized.image, "unchanged size"
        if resized.scale < 1.0:
            message = f"{rgb.width}x{rgb.height} reduced to {working.width}x{working.height}"
            self.notes.append(f"The photo was reduced from {rgb.width}x{rgb.height} to "
                              f"{working.width}x{working.height} to keep processing practical.")
        original = working
        self._finish_step("prepare", "done", started, message)

        # ---- analyze ----
        self._check_cancel(); started = time.time()
        self._report(0.0, "Analyzing the photo", "analyze")
        report = analyze_image(working)
        self._finish_step("analyze", "done", started,
                          f"{report.color_kind}, noise {report.noise_sigma:.1f}")

        current = working

        # ---- repair ----
        self._check_cancel(); started = time.time()
        self._report(0.0, "Removing grain and scratches", "repair")
        if options.repair == "off" and options.denoise == "off":
            self._finish_step("repair", "skipped", started, "turned off")
        else:
            sensitivity = REPAIR_LEVELS[options.repair] or "medium"   # unused when repair is off
            result = self._attempt("repair", started, lambda: restore_classical(
                current, noise_sigma=report.noise_sigma, denoise=(options.denoise == "auto"),
                repair_scratches=(options.repair != "off"), scratch_sensitivity=sensitivity))
            if result is not None:
                current = result.image
                for note in result.notes:
                    if "skipped" in note and "scratch" in note:
                        self.notes.append(note)       # the user should know repair was skipped
                self._finish_step("repair", "done", started, " ".join(result.notes))
        if keep:
            self.intermediates["repaired"] = current

        # ---- colorize ----
        self._check_cancel(); started = time.time()
        self._report(0.0, "Colorizing", "colorize")
        want_color = (options.colorize == "always"
                      or (options.colorize == "auto" and report.needs_colorization))
        if options.colorize == "never":
            self._finish_step("colorize", "skipped", started, "turned off")
        elif not want_color:
            self._finish_step("colorize", "skipped", started, "the photo already has colour")
            if report.color_kind == "color" and report.mean_chroma < FADED_COLOR_MEAN_CHROMA:
                self.notes.append(
                    "This photo has very little colour and may be a toned black-and-white "
                    "print. It was NOT colorized automatically; choose Colorize = always "
                    "if you want colour added.")
        else:
            outcome = self._attempt("colorize", started, lambda: self.pipeline._colorizer().colorize(
                current, saturation=options.saturation))
            if outcome is not None:
                current = outcome.image
                self.notes.extend(outcome.notes)
                self._finish_step("colorize", "done", started, f"model: {outcome.method}")
            free_memory()
        if keep:
            self.intermediates["colorized"] = current

        # ---- upscale ----
        self._check_cancel(); started = time.time()
        self._report(0.0, "Enlarging", "upscale")
        before_upscale, upscaled = current, None
        scale = options.upscale
        if scale > 1.0:
            allowed = options.max_output_side / max(current.size)
            if scale > allowed:
                self.notes.append(f"The enlargement was reduced from {scale:g}x to {allowed:.2f}x "
                                  f"so the result stays within {options.max_output_side} pixels.")
                scale = allowed
        if scale <= 1.0:
            self._finish_step("upscale", "skipped", started,
                              "turned off" if options.upscale == 1.0 else "photo already at the size limit")
        else:
            def tile_progress(done, total):
                self._report(done / total, f"Enlarging (tile {done}/{total})", "upscale")

            result = self._attempt("upscale", started, lambda: self.pipeline._upscaler(
                options.upscale_model).upscale(current, scale=scale, progress=tile_progress))
            if result is not None:
                upscaled = current = result.image
                self.notes.extend(n for n in result.notes if "GPU" in n)
                self._finish_step("upscale", "done", started,
                                  f"{result.model_name} {scale:g}x -> {current.width}x{current.height}")
            free_memory()
        if keep:
            self.intermediates["upscaled"] = current

        # ---- faces (pasted onto the upscaled picture when there is one) ----
        self._check_cancel(); started = time.time()
        self._report(0.0, "Restoring faces", "faces")
        if options.faces == "off":
            self._finish_step("faces", "skipped", started, "turned off")
        else:
            result = self._attempt("faces", started, lambda: self.pipeline._face_restorer().restore(
                before_upscale, strength=options.face_strength, background=upscaled))
            if result is not None:
                current = result.image
                self.notes.extend(result.notes)
                self._finish_step("faces", "done", started,
                                  f"{result.faces_found} found, {result.faces_restored} restored")
            free_memory()
        if keep:
            self.intermediates["faces"] = current

        # ---- finish ----
        self._check_cancel(); started = time.time()
        self._report(0.0, "Final touch-ups", "finish")
        final = postprocess(current, contrast=options.contrast, grain=options.grain)
        for note in final.notes:
            if "already good" not in note:
                self.notes.append(note)
        self.steps.append(StepReport("finish", "done", time.time() - started,
                                     "; ".join(final.notes) or "nothing to do"))
        self.done_weight += STEP_WEIGHTS["finish"]
        if self.progress is not None:
            self.progress(1.0, "Done")

        return PipelineResult(final.image, original, self.steps, self.notes, report.to_dict(),
                              time.time() - total_start, self.intermediates)


# ---------------------------------------------------------------------------
# Self-test: real analysis, repair and post-processing; stand-ins for the
# heavy models, so no weights are needed.
# ---------------------------------------------------------------------------
class _StubColorizer:
    def __init__(self, error=None):
        self.error, self.calls = error, 0

    def colorize(self, image, saturation=1.0):
        self.calls += 1
        if self.error:
            raise self.error
        from src.colorize_backup import ChainResult
        return ChainResult(Image.new("RGB", image.size, (150, 110, 80)), "stub", [])


class _StubFaces:
    def __init__(self, error=None):
        self.error, self.calls, self.last_background = error, 0, "unset"

    def restore(self, image, strength=1.0, background=None):
        self.calls += 1
        self.last_background = background
        if self.error:
            raise self.error
        from src.face_restore import FaceRestoreResult
        return FaceRestoreResult((background or image).copy(), 1, 1, [])


class _StubUpscaler:
    def __init__(self, error=None):
        self.error, self.calls = error, 0

    def upscale(self, image, scale=2.0, progress=None):
        self.calls += 1
        if self.error:
            raise self.error
        for done in (1, 2):
            if progress:
                progress(done, 2)
        from src.super_resolution import UpscaleResult
        big = image.resize((round(image.width * scale), round(image.height * scale)))
        return UpscaleResult(big, scale, "stub", 256, [])


def _run_self_test() -> int:
    import threading

    import numpy as np

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
    y, x = np.mgrid[0:150, 0:100].astype(np.float32)
    gray_array = (np.clip(40 + y / 150 * 170 + rng.normal(0, 1, y.shape), 0, 255)).astype(np.uint8)
    gray_photo = Image.fromarray(gray_array, "L").convert("RGB")                 # 100 x 150, B/W
    colour_photo = Image.fromarray(np.stack(
        [x / 100 * 255, y / 150 * 255, (1 - x / 100) * 255], -1).astype(np.uint8), "RGB")

    def make(colorizer=None, faces=None, upscaler=None):
        parts = Components(colorizer or _StubColorizer(), faces or _StubFaces(),
                           upscaler or _StubUpscaler())
        return RestorationPipeline(parts), parts

    # ---- order, sizes ----
    pipeline, parts = make()
    result = pipeline.run(gray_photo)
    check("steps run in the documented order",
          [s.name for s in result.steps] == STEP_ORDER, str([s.name for s in result.steps]))
    check("default options: a B/W photo is colorized and enlarged 2x",
          parts.colorizer.calls == 1 and result.image.size == (200, 300)
          and result.original.size == (100, 150))
    check("faces are pasted onto the UPSCALED picture (background given)",
          parts.face_restorer.last_background is not None
          and parts.face_restorer.last_background.size == (200, 300))
    pipeline, parts = make()
    pipeline.run(gray_photo, PipelineOptions(upscale=1.0))
    check("no upscaling: faces get no background and the size is unchanged",
          parts.face_restorer.last_background is None and parts.upscaler.calls == 0)

    # ---- colorize decisions ----
    pipeline, parts = make()
    result = pipeline.run(colour_photo)
    check("auto: a colour photo is NOT colorized",
          parts.colorizer.calls == 0 and result.steps[3].status == "skipped")
    pipeline, parts = make(); pipeline.run(colour_photo, PipelineOptions(colorize="always"))
    check("colorize 'always' colorizes a colour photo", parts.colorizer.calls == 1)
    pipeline, parts = make(); pipeline.run(gray_photo, PipelineOptions(colorize="never"))
    check("colorize 'never' skips even a B/W photo", parts.colorizer.calls == 0)
    # A "toned print": different regions have different tints, so the analysis calls it
    # colour even though it has very little colour (the situation of our real test photo).
    left = (x < 50)[..., None]
    base = gray_array.astype(np.float32)[..., None]
    warm = base + np.array([14, 3, -12], np.float32)
    cool = base + np.array([-12, 0, 14], np.float32)
    toned_print = Image.fromarray(np.clip(np.where(left, warm, cool), 0, 255).astype(np.uint8), "RGB")
    pipeline, parts = make(); result = pipeline.run(toned_print, PipelineOptions(upscale=1.0))
    hints = [n for n in result.notes if "toned black-and-white" in n]
    check("a low-colour print the analysis calls 'colour': NOT colorized, but a hint is given",
          result.analysis["color_kind"] == "color" and result.analysis["mean_chroma"] < FADED_COLOR_MEAN_CHROMA
          and parts.colorizer.calls == 0 and len(hints) == 1,
          f"(kind={result.analysis['color_kind']}, chroma={result.analysis['mean_chroma']:.1f}, hints={len(hints)})")
    result = make()[0].run(colour_photo, PipelineOptions(upscale=1.0))
    check("a genuinely colourful photo gets no such hint",
          not any("toned black-and-white" in n for n in result.notes))
    pipeline, parts = make(); pipeline.run(toned_print, PipelineOptions(colorize="always", upscale=1.0))
    check("the user can override: colorize 'always' colorizes that print", parts.colorizer.calls == 1)

    # ---- repair / faces switches ----
    pipeline, parts = make()
    result = pipeline.run(gray_photo, PipelineOptions(repair="off", denoise="off", faces="off"))
    check("repair off + denoise off + faces off: those steps are skipped",
          [s.status for s in result.steps if s.name in ("repair", "faces")] == ["skipped", "skipped"]
          and parts.face_restorer.calls == 0)

    # ---- errors ----
    pipeline, parts = make(colorizer=_StubColorizer(ColorizationError("model broke")))
    result = pipeline.run(gray_photo)
    check("a failing step is reported and the run still finishes",
          result.failed_steps == ["colorize"] and result.image.size == (200, 300)
          and any("model broke" in n for n in result.notes))
    pipeline, parts = make(upscaler=_StubUpscaler(UpscaleError("no memory")))
    result = pipeline.run(gray_photo)
    check("failed upscaling: faces step gets no background and the size stays 1x",
          result.failed_steps == ["upscale"] and parts.face_restorer.last_background is None
          and result.image.size == (100, 150))
    pipeline, parts = make(faces=_StubFaces(FaceRestorationError("no weights")))
    check("on_error='raise' stops at the first failure",
          raises(FaceRestorationError, lambda: pipeline.run(gray_photo, on_error="raise")))
    pipeline, parts = make(colorizer=_StubColorizer(RuntimeError("a real bug")))
    check("unexpected errors (bugs) are NOT hidden",
          raises(RuntimeError, lambda: pipeline.run(gray_photo)))

    # ---- progress, cancel, limits ----
    events = []
    pipeline, parts = make()
    pipeline.run(gray_photo, progress=lambda f, m: events.append((f, m)))
    fractions = [f for f, _ in events]
    check("progress never goes backwards, stays within 0..1 and ends at 1.0",
          fractions == sorted(fractions) and 0.0 <= fractions[0] and fractions[-1] == 1.0)
    check("progress messages are readable text", all(isinstance(m, str) and m for _, m in events))
    stop = threading.Event(); stop.set()
    check("a cancelled run raises PipelineCancelled",
          raises(PipelineCancelled, lambda: make()[0].run(gray_photo, cancel=stop)))

    big = Image.fromarray(rng.integers(0, 256, (400, 600, 3), dtype=np.uint8), "RGB")
    result = make()[0].run(big, PipelineOptions(max_side=300, upscale=1.0))
    check("a large photo is reduced to the working size, with a note",
          max(result.original.size) == 300 and any("reduced" in n for n in result.notes))
    result = make()[0].run(gray_photo, PipelineOptions(upscale=4.0, max_output_side=256))
    check("the output size limit reduces the enlargement, with a note",
          max(result.image.size) <= 257 and any("reduced from 4x" in n for n in result.notes))

    # ---- inputs ----
    check("grayscale and RGBA inputs are accepted",
          make()[0].run(Image.new("L", (80, 80), 100), PipelineOptions(upscale=1.0)).image.mode == "RGB"
          and make()[0].run(Image.new("RGBA", (80, 80), (9, 9, 9, 255)),
                            PipelineOptions(upscale=1.0)).image.mode == "RGB")
    check("invalid options are rejected immediately",
          raises(ValueError, lambda: PipelineOptions(repair="medium"))
          and raises(ValueError, lambda: PipelineOptions(upscale=5.0))
          and raises(ValueError, lambda: PipelineOptions(face_strength=1.5))
          and raises(ValueError, lambda: PipelineOptions(colorize="maybe")))
    result = make()[0].run(gray_photo, keep_intermediates=True)
    check("intermediates can be kept for inspection",
          set(result.intermediates) == {"repaired", "colorized", "upscaled", "faces"})
    check("the result table lists every step", all(name in result.table() for name in STEP_ORDER))

    # a pipeline that serves several requests must give each request the model it asked for
    import src.super_resolution as sr_module
    created = []

    class _RecordingUpscaler(_StubUpscaler):
        def __init__(self, model="auto"):
            super().__init__()
            self.model = model
            created.append(model)

    original_class = sr_module.Upscaler
    sr_module.Upscaler = _RecordingUpscaler
    try:
        shared = RestorationPipeline(Components(_StubColorizer(), _StubFaces(), None))
        first = shared._upscaler("general"); second = shared._upscaler("x4plus")
        again = shared._upscaler("general")
    finally:
        sr_module.Upscaler = original_class
    check("each requested upscaler model is created once and reused (not stuck on the first)",
          first.model == "general" and second.model == "x4plus" and again is first
          and created == ["general", "x4plus"])

    print("-" * 62)
    print(" RESULT:", "all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
def _restore_file(path: str, options: PipelineOptions, keep_steps: bool) -> int:
    from src.image_io import ImageValidationError, load_image_from_path

    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1

    def show(fraction, message):
        print(f"\r [{fraction:4.0%}] {message:<60}", end="", flush=True)

    result = RestorationPipeline().run(loaded.image, options, progress=show,
                                       keep_intermediates=keep_steps)
    print("\r" + " " * 75)
    print(result.table())
    for note in result.notes:
        print(f" NOTE: {note}")

    config.ensure_directories()
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(path).stem)
    out_path = config.OUTPUTS_DIR / f"restored_{stem}.png"
    cmp_path = config.OUTPUTS_DIR / f"restored_{stem}_comparison.png"
    result.image.save(out_path)
    before = result.original.resize(result.image.size, Image.LANCZOS)
    sheet = Image.new("RGB", (result.image.width * 2, result.image.height))
    sheet.paste(before, (0, 0))
    sheet.paste(result.image, (result.image.width, 0))
    sheet.save(cmp_path)
    if keep_steps:
        for number, (name, picture) in enumerate(result.intermediates.items(), 1):
            picture.save(config.OUTPUTS_DIR / f"steps_{stem}_{number}_{name}.png")
    print(f" Saved: {out_path}\n Saved: {cmp_path}  (before | after)")
    return 1 if result.failed_steps else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete photo restoration pipeline")
    parser.add_argument("image", nargs="?", help="photo to restore (omit to run the self-test)")
    parser.add_argument("--repair", default="light", choices=list(REPAIR_LEVELS))
    parser.add_argument("--denoise", default="auto", choices=DENOISE_MODES)
    parser.add_argument("--colorize", default="auto", choices=COLORIZE_MODES)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--upscale", type=float, default=2.0, help="1 = off, up to 4")
    parser.add_argument("--upscale-model", default="auto", choices=["auto", "general", "x4plus"])
    parser.add_argument("--faces", default="on", choices=ONOFF)
    parser.add_argument("--face-strength", type=float, default=0.6)
    parser.add_argument("--contrast", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--grain", type=float, default=0.0)
    parser.add_argument("--keep-steps", action="store_true", help="also save each intermediate picture")
    arguments = parser.parse_args()
    if not arguments.image:
        sys.exit(_run_self_test())
    try:
        chosen = PipelineOptions(
            repair=arguments.repair, denoise=arguments.denoise, colorize=arguments.colorize,
            saturation=arguments.saturation, upscale=arguments.upscale,
            upscale_model=arguments.upscale_model, faces=arguments.faces,
            face_strength=arguments.face_strength, contrast=arguments.contrast,
            grain=arguments.grain)
    except ValueError as error:
        print(f" Invalid option: {error}")
        sys.exit(2)
    sys.exit(_restore_file(arguments.image, chosen, arguments.keep_steps))
