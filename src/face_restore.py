"""
face_restore.py

Restores faces in old photos with GFPGAN v1.4 and pastes them back.

    from src.face_restore import FaceRestorer

    restorer = FaceRestorer()                        # nothing is loaded yet
    result = restorer.restore(image)                 # loads on first use
    result.image            # Pillow RGB image, same size as the input
    result.faces_found      # how many faces were detected
    result.faces_restored   # how many were actually restored
    print(result.notes)     # plain-English remarks

How it works
  1. A face detector (RetinaFace) finds faces and 5 landmarks per face
     (eyes, nose, mouth corners).
  2. Each face is cut out, straightened and resized to 512 x 512.
  3. GFPGAN rebuilds the face at 512 x 512 (sharper skin, eyes, hair edges).
  4. A face-parsing network makes a soft mask and the restored face is
     pasted back, undoing the straightening.
The rest of the photo is NOT touched by this module.

Honest limits
  * GFPGAN "invents" plausible detail. The restored face can look different
    from the real person (especially in very small or very damaged faces).
    Use `strength` below 1.0 to keep more of the original face.
  * Cracks or scratches across a face may be turned into odd skin texture.
  * Faces smaller than a few dozen pixels, or turned sideways, are skipped
    or restored poorly.
  * GFPGAN's own `weight` option does nothing in the v1.4 network (we
    verified that the output is identical), so we do not offer it. `strength`
    is our own blend between the original and restored face.
  * GFPGAN and its weights are for NON-COMMERCIAL use (see
    scripts/setup_face_code.py).

Self-test (no weights needed):   python -m src.face_restore
Restore faces in a file:         python -m src.face_restore samples\\test_1.jpg
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src import config
from src.device import free_memory, get_device_info, is_out_of_memory_error
from src.logging_setup import get_logger
from src.preprocessing import bgr_array_to_pil, pil_to_bgr_array

logger = get_logger(__name__)

# --- Settings ----------------------------------------------------------------
GFPGAN_WEIGHTS_PATH = config.MODELS_DIR / "gfpgan" / "GFPGANv1.4.pth"
FACEXLIB_DIR = config.MODELS_DIR / "facexlib"
DETECTION_WEIGHTS_PATH = FACEXLIB_DIR / "detection_Resnet50_Final.pth"
PARSING_WEIGHTS_PATH = FACEXLIB_DIR / "parsing_parsenet.pth"

FACE_SIZE = 512                  # GFPGAN works on 512 x 512 faces
DETECTION_THRESHOLD = 0.97       # face-detector confidence (upstream default)
MIN_EYE_DISTANCE = 5             # pixels; skip faces with eyes closer than this
DEFAULT_MAX_FACES = 6            # each face costs several seconds on a CPU
MAX_SCALE_MISMATCH_PIXELS = 2    # tolerance when checking a background image


class FaceRestorationError(Exception):
    """Face restoration could not be completed (message is user-friendly)."""


class FaceModelMissingError(FaceRestorationError):
    """Model weights are missing."""


@dataclass
class FaceRestoreResult:
    image: Image.Image
    faces_found: int
    faces_restored: int
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversions between a face picture and the tensor GFPGAN expects
# ---------------------------------------------------------------------------
def face_to_tensor(face_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H, W, 3) -> float tensor (1, 3, H, W), RGB, range -1..1."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)


def tensor_to_face(output: torch.Tensor) -> np.ndarray:
    """Tensor (1, 3, H, W) in about -1..1 (RGB) -> BGR uint8 (H, W, 3).

    Values are clipped first: the network sometimes overshoots slightly
    (we measured 1.01), and unclipped values would wrap around when
    converted to bytes.
    """
    array = output.squeeze(0).detach().float().cpu().clamp(-1.0, 1.0).numpy()
    array = ((array + 1.0) / 2.0 * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(array.transpose(1, 2, 0).copy(), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# The restorer
# ---------------------------------------------------------------------------
class FaceRestorer:
    """Loads GFPGAN and the face tools lazily (on first use)."""

    def __init__(self, gfpgan_path=None, facexlib_dir=None,
                 detection_threshold: float = DETECTION_THRESHOLD,
                 model=None, helper=None):
        """
        Args:
            gfpgan_path: GFPGANv1.4.pth (default: models/gfpgan/).
            facexlib_dir: folder with the two facexlib weights files.
            detection_threshold: face-detector confidence, 0..1. Lower finds
                more faces in damaged photos but risks false detections.
            model, helper: ready-made stand-ins (used by the self-test).
        """
        self.gfpgan_path = Path(gfpgan_path) if gfpgan_path else GFPGAN_WEIGHTS_PATH
        self.facexlib_dir = Path(facexlib_dir) if facexlib_dir else FACEXLIB_DIR
        self.detection_threshold = detection_threshold
        self._model = model
        self._helper = helper
        self._device = torch.device("cpu")
        self._force_cpu = False
        self._lock = threading.Lock()   # models keep internal state: one image at a time

    # -- weights ------------------------------------------------------------
    def _required_files(self):
        return [self.gfpgan_path,
                self.facexlib_dir / DETECTION_WEIGHTS_PATH.name,
                self.facexlib_dir / PARSING_WEIGHTS_PATH.name]

    def weights_available(self) -> bool:
        return all(path.is_file() for path in self._required_files())

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._helper is not None:
            return
        missing = [p.name for p in self._required_files() if not p.is_file()]
        if missing:
            raise FaceModelMissingError(
                f"Face restoration files missing: {', '.join(missing)}. Download "
                "them with:  python scripts\\download_weights.py")
        try:
            from src.vendor.facexlib.utils.face_restoration_helper import FaceRestoreHelper
            from src.vendor.gfpgan.gfpganv1_clean_arch import GFPGANv1Clean
        except ImportError as error:
            raise FaceRestorationError(
                "The face restoration code is missing. Run this once from the "
                "project folder:  python scripts\\setup_face_code.py"
                f"  (details: {error})") from error

        started = time.time()
        info = get_device_info()
        self._device = torch.device("cpu") if self._force_cpu else info.device
        try:
            model = GFPGANv1Clean(
                out_size=FACE_SIZE, num_style_feat=512, channel_multiplier=2,
                decoder_load_path=None, fix_decoder=False, num_mlp=8,
                input_is_latent=True, different_w=True, narrow=1, sft_half=True)
            checkpoint = torch.load(self.gfpgan_path, map_location="cpu",
                                    weights_only=True)
            key = "params_ema" if "params_ema" in checkpoint else "params"
            model.load_state_dict(checkpoint[key], strict=True)
        except Exception as error:
            raise FaceRestorationError(
                f"The GFPGAN weights file {self.gfpgan_path.name} could not be "
                "loaded (damaged download, or not the right file). Run "
                "python scripts\\download_weights.py --verify, delete it if it "
                f"reports a problem, and download it again. ({str(error)[:150]})"
            ) from error
        self._model = model.eval().to(self._device)

        try:
            # model_rootpath: the library finds its weights here and never downloads.
            helper = FaceRestoreHelper(
                1, face_size=FACE_SIZE, crop_ratio=(1, 1),
                det_model="retinaface_resnet50", save_ext="png", use_parse=True,
                device=self._device, model_rootpath=str(self.facexlib_dir))
        except Exception as error:
            raise FaceRestorationError(
                "The face detection files could not be loaded (they may be "
                "damaged). Run python scripts\\download_weights.py --verify. "
                f"({str(error)[:150]})") from error

        # The library hard-codes its detection confidence; replace it with ours.
        original_detect = helper.face_det.detect_faces
        threshold = self.detection_threshold
        helper.face_det.detect_faces = lambda image, *args, **kwargs: original_detect(
            image, threshold)
        self._helper = helper
        logger.info("Face restoration loaded on %s in %.1f s",
                    self._device.type, time.time() - started)

    def _unload(self) -> None:
        self._model = None
        self._helper = None
        free_memory()

    # -- main API -----------------------------------------------------------
    def restore(self, image: Image.Image, strength: float = 1.0,
                background: Image.Image | None = None,
                only_center_face: bool = False,
                max_faces: int = DEFAULT_MAX_FACES,
                min_eye_distance: float = MIN_EYE_DISTANCE) -> FaceRestoreResult:
        """Restore the faces in a photo.

        Args:
            image: RGB Pillow image.
            strength: 1.0 = fully restored faces; 0.0 = original faces;
                in between blends the two (more natural, keeps identity).
            background: optional LARGER version of the same photo (for
                example from the super-resolution step). Faces are pasted
                onto it, so the result has the background's size.
            only_center_face: restore only the face nearest the centre.
            max_faces: restore at most this many faces (the largest ones).
            min_eye_distance: ignore faces whose eyes are closer than this
                many pixels (too small to restore sensibly).
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be between 0.0 and 1.0.")
        if max_faces < 1:
            raise ValueError("max_faces must be at least 1.")

        scale = 1.0
        if background is not None:
            scale = background.width / image.width
            if abs(background.height - image.height * scale) > MAX_SCALE_MISMATCH_PIXELS:
                raise ValueError(
                    "background must be a scaled version of the same photo "
                    f"(got {background.size} for an image of {image.size}).")

        with self._lock:
            self._ensure_loaded()
            started = time.time()
            try:
                result = self._restore(image, strength, background, scale,
                                       only_center_face, max_faces, min_eye_distance)
            except FaceRestorationError:
                raise
            except Exception as error:
                if self._device.type == "cuda" and is_out_of_memory_error(error):
                    logger.warning("GPU ran out of memory; retrying on the CPU.")
                    self._force_cpu = True
                    self._unload()
                    self._ensure_loaded()
                    try:
                        result = self._restore(image, strength, background, scale,
                                               only_center_face, max_faces,
                                               min_eye_distance)
                    except Exception as retry_error:
                        raise FaceRestorationError(
                            f"Face restoration failed even on the CPU: {retry_error}"
                        ) from retry_error
                elif is_out_of_memory_error(error):
                    raise FaceRestorationError(
                        "Not enough memory for face restoration. Close other "
                        "programs, or try a smaller image.") from error
                else:
                    raise FaceRestorationError(
                        f"Face restoration failed: {error}") from error
            logger.info("Face restoration: %d found, %d restored, %.1f s",
                        result.faces_found, result.faces_restored,
                        time.time() - started)
            return result

    def _restore(self, image, strength, background, scale, only_center_face,
                 max_faces, min_eye_distance) -> FaceRestoreResult:
        helper = self._helper
        notes = []
        bgr = pil_to_bgr_array(image)
        background_bgr = pil_to_bgr_array(background) if background is not None else None

        helper.clean_all()
        helper.set_upscale_factor(scale)
        helper.read_image(bgr)
        helper.get_face_landmarks_5(only_center_face=only_center_face,
                                    eye_dist_threshold=min_eye_distance)
        found = len(helper.det_faces)
        if found == 0:
            notes.append("No faces were detected, so the faces were left unchanged.")
            # convert() always returns a NEW RGB image (also fixes grayscale input)
            unchanged = (background if background is not None else image).convert("RGB")
            return FaceRestoreResult(unchanged, 0, 0, notes)

        if found > max_faces:
            def box_area(box):
                return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            keep = sorted(range(found), key=lambda i: box_area(helper.det_faces[i]),
                          reverse=True)[:max_faces]
            helper.det_faces = [helper.det_faces[i] for i in keep]
            helper.all_landmarks_5 = [helper.all_landmarks_5[i] for i in keep]
            notes.append(f"{found} faces found; only the {max_faces} largest were "
                         "restored (to keep processing time reasonable).")

        helper.align_warp_face()
        failed = 0
        for cropped_face in helper.cropped_faces:
            try:
                restored_face = self._restore_one_face(cropped_face, strength)
            except Exception as error:
                if is_out_of_memory_error(error):
                    raise                    # handled by the caller (CPU retry)
                logger.warning("One face could not be restored: %s", error)
                restored_face = cropped_face
                failed += 1
            helper.add_restored_face(restored_face)
        if failed:
            notes.append(f"{failed} face(s) could not be restored and were left as they were.")

        helper.get_inverse_affine(None)
        pasted = helper.paste_faces_to_input_image(upsample_img=background_bgr)
        result_image = bgr_array_to_pil(pasted)
        target_size = background.size if background is not None else image.size
        if result_image.size != target_size:         # rounding of a fractional scale
            result_image = result_image.resize(target_size, Image.LANCZOS)
        restored_count = len(helper.cropped_faces) - failed
        return FaceRestoreResult(result_image, found, restored_count, notes)

    def _restore_one_face(self, face_bgr: np.ndarray, strength: float) -> np.ndarray:
        tensor = face_to_tensor(face_bgr).to(self._device)
        with torch.no_grad():
            output = self._model(tensor, return_rgb=False)[0]
        restored = tensor_to_face(output)
        if strength < 1.0:
            restored = cv2.addWeighted(restored, strength, face_bgr, 1.0 - strength, 0)
        return restored


# ---------------------------------------------------------------------------
# Self-test (needs no weights)
# ---------------------------------------------------------------------------
class _FakeHelper:
    """Stand-in for facexlib's FaceRestoreHelper with a chosen list of faces."""

    def __init__(self, boxes):
        self.boxes = boxes                  # list of (x0, y0, x1, y1)
        self.scale_calls = []
        self.upsample_seen = "unset"
        self.clean_all()

    def clean_all(self):
        self.det_faces, self.all_landmarks_5 = [], []
        self.cropped_faces, self.restored_faces = [], []

    def set_upscale_factor(self, scale):
        self.scale_calls.append(scale)

    def read_image(self, img):
        self.input_img = img

    def get_face_landmarks_5(self, only_center_face=False, eye_dist_threshold=None):
        for x0, y0, x1, y1 in self.boxes:
            self.det_faces.append(np.array([x0, y0, x1, y1, 0.99]))
            self.all_landmarks_5.append(np.zeros((5, 2)))
        return len(self.det_faces)

    def align_warp_face(self):
        # Each face gets a distinct flat colour, so tests can tell them apart.
        for index in range(len(self.det_faces)):
            self.cropped_faces.append(np.full((512, 512, 3), 40 + 60 * index, np.uint8))

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def get_inverse_affine(self, path):
        pass

    def paste_faces_to_input_image(self, save_path=None, upsample_img=None):
        self.upsample_seen = upsample_img
        return (upsample_img if upsample_img is not None else self.input_img).copy()


class _ConstantModel:
    """Stand-in network: always outputs mid-grey (0.0 in -1..1), and counts calls."""

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, tensor, return_rgb=False):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise ValueError("simulated failure on one face")
        return torch.zeros_like(tensor), None


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

    photo = Image.fromarray(np.random.default_rng(0).integers(
        0, 256, (300, 200, 3), dtype=np.uint8), "RGB")

    # ---- conversions ----
    face = np.random.default_rng(1).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    tensor = face_to_tensor(face)
    check("face -> tensor has shape (1, 3, 512, 512) and range -1..1",
          tuple(tensor.shape) == (1, 3, 512, 512)
          and float(tensor.min()) >= -1.0 and float(tensor.max()) <= 1.0)
    check("face -> tensor -> face returns the same pixels (within 1)",
          int(np.abs(tensor_to_face(tensor).astype(int) - face.astype(int)).max()) <= 1)
    check("out-of-range values are clipped, not wrapped around",
          tensor_to_face(torch.full((1, 3, 4, 4), 1.2)).min() == 255)

    # ---- argument and weight handling ----
    restorer = FaceRestorer(model=_ConstantModel(), helper=_FakeHelper([]))
    check("strength outside 0..1 is rejected",
          raises(ValueError, lambda: restorer.restore(photo, strength=1.5)))
    check("missing weights give FaceModelMissingError",
          raises(FaceModelMissingError, lambda: FaceRestorer(
              gfpgan_path="nope/a.pth", facexlib_dir="nope").restore(photo)))

    # ---- behaviour with stand-in components ----
    result = FaceRestorer(model=_ConstantModel(), helper=_FakeHelper([])).restore(photo)
    check("no face: photo returned unchanged, with a note",
          result.faces_found == 0 and result.faces_restored == 0
          and np.array_equal(np.array(result.image), np.array(photo))
          and "No faces" in result.notes[0])

    model, helper = _ConstantModel(), _FakeHelper([(10, 10, 60, 70), (100, 100, 190, 220)])
    result = FaceRestorer(model=model, helper=helper).restore(photo)
    check("two faces: both found, both restored, model run twice",
          result.faces_found == 2 and result.faces_restored == 2 and model.calls == 2)
    check("restored faces are the model output (mid-grey = 128)",
          all(abs(int(f.mean()) - 128) <= 1 for f in helper.restored_faces))

    helper = _FakeHelper([(10, 10, 60, 70)])
    FaceRestorer(model=_ConstantModel(), helper=helper).restore(photo, strength=0.5)
    blended = int(helper.restored_faces[0].mean())
    check("strength 0.5 blends original (40) and restored (128): about 84",
          abs(blended - 84) <= 2, f"(got {blended})")

    helper = _FakeHelper([(10, 10, 60, 70)])
    FaceRestorer(model=_ConstantModel(), helper=helper).restore(photo, strength=0.0)
    check("strength 0 keeps the original face",
          int(helper.restored_faces[0].mean()) == 40)

    helper = _FakeHelper([(0, 0, 20, 20), (0, 0, 100, 100), (0, 0, 50, 50)])
    result = FaceRestorer(model=_ConstantModel(), helper=helper).restore(photo, max_faces=2)
    sizes = sorted(int((box[2] - box[0])) for box in helper.det_faces)
    check("more faces than max_faces: only the largest are restored, with a note",
          result.faces_found == 3 and result.faces_restored == 2
          and sizes == [50, 100] and "largest" in result.notes[0])

    helper = _FakeHelper([(10, 10, 60, 70), (100, 100, 190, 220)])
    result = FaceRestorer(model=_ConstantModel(fail_on_call=1), helper=helper).restore(photo)
    check("one face failing keeps the original crop and reports it",
          result.faces_restored == 1 and result.faces_found == 2
          and int(helper.restored_faces[0].mean()) == 40 and "could not" in result.notes[0])

    # ---- background (larger version of the same photo) ----
    big = photo.resize((400, 600))
    helper = _FakeHelper([(10, 10, 60, 70)])
    result = FaceRestorer(model=_ConstantModel(), helper=helper).restore(photo, background=big)
    check("with a 2x background: scale 2.0 is used and the result has its size",
          helper.scale_calls[-1] == 2.0 and result.image.size == big.size)
    check("a background with the wrong shape is rejected",
          raises(ValueError, lambda: FaceRestorer(
              model=_ConstantModel(), helper=_FakeHelper([])).restore(
                  photo, background=photo.resize((300, 300)))))

    gray = Image.new("L", (200, 300), 128)
    result = FaceRestorer(model=_ConstantModel(), helper=_FakeHelper([])).restore(gray)
    check("a grayscale image is accepted", result.image.mode == "RGB")

    print("-" * 62)
    present = FaceRestorer().weights_available()
    print(f" Face weights present: {'yes' if present else 'NO'}")
    print(" (The self-test does not load the real models. To test them, run\n"
          "  python -m src.face_restore samples\\test_1.jpg)")
    print(" RESULT:", "all checks passed" if failures == 0
          else f"{failures} check(s) failed")
    return 1 if failures else 0


def _restore_file(path: str, strength: float) -> int:
    from src.image_io import ImageValidationError, load_image_from_path
    from src.preprocessing import limit_working_size

    try:
        loaded = load_image_from_path(path)
    except ImageValidationError as error:
        print(f" REJECTED: {error}")
        return 1
    working = limit_working_size(loaded.image).image

    print(" Loading face models and restoring (the first run can take a minute)...")
    started = time.time()
    try:
        result = FaceRestorer().restore(working, strength=strength)
    except FaceRestorationError as error:
        print(f" ERROR: {error}")
        return 1

    config.ensure_directories()
    result_path = config.OUTPUTS_DIR / "faces_result.png"
    comparison_path = config.OUTPUTS_DIR / "faces_comparison.png"
    result.image.save(result_path)
    side_by_side = Image.new("RGB", (working.width * 2, working.height))
    side_by_side.paste(working, (0, 0))
    side_by_side.paste(result.image, (working.width, 0))
    side_by_side.save(comparison_path)

    print(f" Faces found: {result.faces_found}, restored: {result.faces_restored}")
    for note in result.notes:
        print(f" - {note}")
    print(f" Done in {time.time() - started:.1f} s on {get_device_info().kind.upper()}")
    print(f" Saved: {result_path}")
    print(f" Saved: {comparison_path}  (before | after)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFPGAN face restoration")
    parser.add_argument("image", nargs="?", help="image to process (omit to run the self-test)")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="0.0 = original faces, 1.0 = fully restored (default)")
    arguments = parser.parse_args()
    if arguments.image:
        sys.exit(_restore_file(arguments.image, arguments.strength))
    sys.exit(_run_self_test())