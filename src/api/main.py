"""
main.py

The web API. It lets a browser (or any program) upload a photo, follow the
restoration while it runs, and download the result.

    Start the server:      python -m src.api.main
    Then open:             http://127.0.0.1:8000/docs     (interactive page, try it there)
    Self-test:             python -m src.api.main --self-test      (needs httpx)

Endpoints
    GET    /api/health                  server status, which models are installed
    GET    /api/options                 every option with its allowed values (for building a form)
    POST   /api/jobs                    upload a photo (+ options) -> returns a job id
    GET    /api/jobs/{id}               status, progress, step table, notes
    POST   /api/jobs/{id}/cancel        stop a job
    DELETE /api/jobs/{id}               delete a finished job and its pictures
    GET    /api/jobs/{id}/result        the restored picture (?format=png|jpeg, ?download=1)
    GET    /api/jobs/{id}/original      the "before" picture at working size

Safety
  * Uploads go through src/image_io.py (type, size, corruption, pixel-count
    checks). The uploaded file NAME is never used for anything.
  * By default the server listens on 127.0.0.1 only: there is NO login, so do
    not expose it to a network you do not trust.
  * Pictures are temporary (see jobs.py) and responses are marked no-store.
  * Failures return plain messages; internal details only go to the log file.
"""

import argparse
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, fields

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from PIL import Image
from starlette.concurrency import run_in_threadpool

from src import config
from src.api.jobs import (JOB_TTL_SECONDS, MAX_ACTIVE_JOBS, GENERIC_ERROR, JobManager,
                          JobNotFoundError, JobStateError, QueueFullError)
from src.device import get_device_info
from src.image_io import (CorruptImageError, FileTooLargeError, ImageTooLargeError,
                          ImageTooSmallError, ImageValidationError, UnsupportedFormatError,
                          load_image_from_bytes)
from src.logging_setup import get_logger
from src.pipeline import PipelineOptions

logger = get_logger(__name__)

API_VERSION = "0.1.0"
MULTIPART_OVERHEAD_BYTES = 1024 * 1024        # form fields and boundaries around the file
JPEG_QUALITY = 92
DEFAULTS = PipelineOptions()                  # the single source of the default values

# HTTP status for each kind of image problem
VALIDATION_STATUS = [
    (FileTooLargeError, 413), (ImageTooLargeError, 413), (UnsupportedFormatError, 415),
    (ImageTooSmallError, 422), (CorruptImageError, 400),
]

# Human-readable description of every option (used to build the web form).
OPTION_INFO = [
    {"name": "repair", "label": "Scratch repair", "type": "choice",
     "choices": ["off", "light", "strong"],
     "help": "Fills in thin scratches and cracks. 'light' catches only the clearest ones."},
    {"name": "denoise", "label": "Grain removal", "type": "choice", "choices": ["auto", "off"],
     "help": "'auto' removes grain only when the photo is noticeably grainy."},
    {"name": "colorize", "label": "Colorize", "type": "choice",
     "choices": ["auto", "always", "never"],
     "help": "'auto' colorizes black-and-white photos. Toned prints (very little colour) "
             "are not detected as black-and-white: choose 'always' for those."},
    {"name": "saturation", "label": "Colour strength", "type": "number",
     "min": 0.0, "max": 2.0, "step": 0.1, "help": "1.0 = as predicted. Lower is more muted."},
    {"name": "upscale", "label": "Enlarge", "type": "number", "min": 1.0, "max": 4.0,
     "step": 0.5, "help": "1 = keep the size. 2 = twice as wide and tall."},
    {"name": "upscale_model", "label": "Enlargement quality", "type": "choice",
     "choices": ["auto", "general", "x4plus"],
     "help": "'general' is fast. 'x4plus' looks better but is very slow without a GPU."},
    {"name": "faces", "label": "Restore faces", "type": "choice", "choices": ["on", "off"],
     "help": "Sharpens faces. The AI invents detail, so faces can change slightly."},
    {"name": "face_strength", "label": "Face restoration strength", "type": "number",
     "min": 0.0, "max": 1.0, "step": 0.05,
     "help": "1.0 = fully restored. Lower keeps more of the original face."},
    {"name": "contrast", "label": "Contrast", "type": "choice", "choices": ["auto", "on", "off"],
     "help": "'auto' only helps faded photos and leaves good ones untouched."},
    {"name": "grain", "label": "Film grain", "type": "number", "min": 0.0, "max": 1.0,
     "step": 0.05, "help": "Cosmetic. About 0.4 gives a natural old-photo texture."},
    {"name": "max_side", "label": "Working size limit (pixels)", "type": "integer", "min": 128,
     "advanced": True, "help": "Larger photos are shrunk to this before processing. Empty = automatic."},
    {"name": "max_output_side", "label": "Largest result (pixels)", "type": "integer", "min": 256,
     "advanced": True, "help": "The enlargement is reduced if the result would exceed this."},
]


def create_app(manager: JobManager | None = None) -> FastAPI:
    manager = manager or JobManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(
        title="AI Old Photo Restoration",
        version=API_VERSION,
        description="Upload an old photo, follow its restoration, download the result. "
                    "Everything runs on this computer.",
        lifespan=lifespan,
    )
    app.state.manager = manager

    # -- responses are private and never cached ---------------------------------
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(RequestValidationError)
    async def friendly_validation_error(request: Request, error: RequestValidationError):
        """One readable sentence instead of FastAPI's technical list."""
        parts = []
        for item in error.errors():
            name = str(item["loc"][-1]) if item.get("loc") else "request"
            parts.append(f"'{name}': {item.get('msg', 'invalid value')}")
        return JSONResponse(status_code=422, content={"detail": "Invalid input - " + "; ".join(parts)})

    def find_job(job_id: str):
        try:
            return manager.get(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from None

    # -- information -------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/docs")

    @app.get("/api/health", summary="Server status")
    def health():
        from src.colorize import DDCOLOR_WEIGHTS_PATH
        from src.colorize_backup import WEIGHTS_PATH as BACKUP_WEIGHTS_PATH
        from src.face_restore import DETECTION_WEIGHTS_PATH, GFPGAN_WEIGHTS_PATH, PARSING_WEIGHTS_PATH
        from src.super_resolution import MODEL_FILES
        info = get_device_info()
        return {
            "status": "ok",
            "version": API_VERSION,
            "device": {"kind": info.kind, "name": info.name},
            "models_installed": {
                "colorizer": DDCOLOR_WEIGHTS_PATH.is_file(),
                "backup_colorizer": BACKUP_WEIGHTS_PATH.is_file(),
                "faces": all(p.is_file() for p in (GFPGAN_WEIGHTS_PATH, DETECTION_WEIGHTS_PATH,
                                                   PARSING_WEIGHTS_PATH)),
                "upscaler_general": MODEL_FILES["general"].is_file(),
                "upscaler_x4plus": MODEL_FILES["x4plus"].is_file(),
            },
            "limits": {"max_upload_mb": config.MAX_UPLOAD_MB, "max_active_jobs": manager.max_active,
                       "results_kept_seconds": int(manager.ttl)},
            "active_jobs": manager.active_count(),
        }

    @app.get("/api/options", summary="All options with allowed values and defaults")
    def options():
        return {"options": OPTION_INFO, "defaults": asdict(DEFAULTS)}

    # -- jobs --------------------------------------------------------------------
    @app.post("/api/jobs", status_code=202, summary="Upload a photo and start restoring it")
    async def create_job(
        request: Request,
        file: UploadFile = File(..., description="The photo (JPEG, PNG, WebP, BMP or TIFF)"),
        repair: str = Form(DEFAULTS.repair),
        denoise: str = Form(DEFAULTS.denoise),
        colorize: str = Form(DEFAULTS.colorize),
        saturation: float = Form(DEFAULTS.saturation),
        upscale: float = Form(DEFAULTS.upscale),
        upscale_model: str = Form(DEFAULTS.upscale_model),
        faces: str = Form(DEFAULTS.faces),
        face_strength: float = Form(DEFAULTS.face_strength),
        contrast: str = Form(DEFAULTS.contrast),
        grain: float = Form(DEFAULTS.grain),
        max_side: int | None = Form(None),
        max_output_side: int = Form(DEFAULTS.max_output_side),
    ):
        too_large = (f"The file is larger than the {config.MAX_UPLOAD_MB} MB limit. "
                     "Please upload a smaller file.")
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > config.MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES:
            raise HTTPException(status_code=413, detail=too_large)
        data = await file.read(config.MAX_UPLOAD_BYTES + 1)      # never read more than the limit
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=too_large)

        try:        # decoding a big photo takes a moment: keep the server responsive
            loaded = await run_in_threadpool(load_image_from_bytes, data, file.filename or "")
        except ImageValidationError as error:
            status = next((code for kind, code in VALIDATION_STATUS if isinstance(error, kind)), 400)
            raise HTTPException(status_code=status, detail=str(error)) from None
        try:
            chosen = PipelineOptions(
                repair=repair, denoise=denoise, colorize=colorize, saturation=saturation,
                upscale=upscale, upscale_model=upscale_model, faces=faces,
                face_strength=face_strength, contrast=contrast, grain=grain,
                max_side=max_side, max_output_side=max_output_side)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=f"Invalid option - {error}") from None
        try:
            job = manager.submit(loaded.image, chosen)
        except QueueFullError as error:
            raise HTTPException(status_code=503, detail=str(error),
                                headers={"Retry-After": "30"}) from None
        return {"job_id": job.id, "status": job.status.value, "status_url": f"/api/jobs/{job.id}"}

    @app.get("/api/jobs/{job_id}", summary="Status and progress of a job")
    def job_status(job_id: str):
        return manager.public_view(find_job(job_id))

    @app.post("/api/jobs/{job_id}/cancel", summary="Cancel a job")
    def cancel_job(job_id: str):
        find_job(job_id)
        try:
            return manager.public_view(manager.cancel(job_id))
        except JobStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None

    @app.delete("/api/jobs/{job_id}", status_code=204, summary="Delete a job and its pictures")
    def delete_job(job_id: str):
        find_job(job_id)
        try:
            manager.delete(job_id)
        except JobStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/result", summary="Download the restored picture")
    async def job_result(job_id: str, format: str = Query("png", pattern="^(png|jpeg)$"),
                         download: bool = False):
        find_job(job_id)
        try:
            path = manager.result_path(job_id, "result")
        except JobStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        disposition = "attachment" if download else "inline"
        if format == "png":
            return FileResponse(path, media_type="image/png", filename="restored.png",
                                content_disposition_type=disposition)

        def to_jpeg() -> bytes:
            buffer = io.BytesIO()
            with Image.open(path) as picture:
                picture.convert("RGB").save(buffer, "JPEG", quality=JPEG_QUALITY)
            return buffer.getvalue()

        content = await run_in_threadpool(to_jpeg)
        return Response(content, media_type="image/jpeg",
                        headers={"Content-Disposition": f'{disposition}; filename="restored.jpg"'})

    @app.get("/api/jobs/{job_id}/original", summary="The 'before' picture (working size)")
    def job_original(job_id: str):
        find_job(job_id)
        try:
            path = manager.result_path(job_id, "original")
        except JobStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        return FileResponse(path, media_type="image/png", content_disposition_type="inline")

    return app


# ---------------------------------------------------------------------------
# Self-test: the whole API through FastAPI's test client, with fast stand-ins
# for the AI models (so no weights are needed).
# ---------------------------------------------------------------------------
def _run_self_test() -> int:
    try:
        from fastapi.testclient import TestClient
    except ImportError as error:
        print(f" The self-test needs the 'httpx' package: pip install httpx==0.28.1  ({error})")
        return 1
    import logging
    import tempfile
    from pathlib import Path
    from unittest import mock

    import numpy as np

    # Keep the console readable while testing: only warnings and errors are shown here
    # (the log file still gets everything).
    for handler in logging.getLogger("photo_restorer").handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.WARNING)

    from src.colorize import ColorizationError
    from src.pipeline import Components, RestorationPipeline, _StubColorizer, _StubFaces, _StubUpscaler

    failures = 0

    def check(description: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        if condition:
            print(f" [PASS] {description}")
        else:
            failures += 1
            print(f" [FAIL] {description} {detail}")

    class SlowColorizer(_StubColorizer):
        def __init__(self, delay, error=None):
            super().__init__(error)
            self.delay = delay

        def colorize(self, image, saturation=1.0):
            time.sleep(self.delay)
            return super().colorize(image, saturation)

    def png(size=(200, 150), gradient=True) -> bytes:
        array = np.tile(np.linspace(40, 210, size[0]), (size[1], 1)).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array, "L").convert("RGB").save(buffer, "PNG")
        return buffer.getvalue()

    def make_manager(folder: Path, colorizer=None, **kwargs) -> JobManager:
        parts = Components(colorizer or _StubColorizer(), _StubFaces(), _StubUpscaler())
        return JobManager(RestorationPipeline(parts), jobs_dir=folder / "jobs", **kwargs)

    def upload(client, data=None, name="photo.png", **form):
        return client.post("/api/jobs", files={"file": (name, data or png(), "image/png")},
                           data={k: str(v) for k, v in form.items()})

    def wait_for(client, job_id, wanted, timeout=25.0):
        deadline, view = time.time() + timeout, {}
        while time.time() < deadline:
            view = client.get(f"/api/jobs/{job_id}").json()
            if view.get("status") in wanted:
                return view
            time.sleep(0.05)
        return view

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)

        # ============ normal flow ============
        manager = make_manager(temp / "a")
        with TestClient(create_app(manager)) as client:
            health = client.get("/api/health").json()
            check("health: ok, with device, models and limits",
                  health["status"] == "ok" and "kind" in health["device"]
                  and "colorizer" in health["models_installed"] and "max_upload_mb" in health["limits"])
            schema = client.get("/api/options").json()
            names = {o["name"] for o in schema["options"]}
            check("options: the schema covers every pipeline option, defaults match",
                  names == {f.name for f in fields(PipelineOptions)}
                  and schema["defaults"] == asdict(PipelineOptions()))
            check("the home page redirects to the interactive docs",
                  client.get("/", follow_redirects=False).status_code in (302, 307)
                  and client.get("/docs").status_code == 200)

            response = upload(client, name="../../evil name.png")
            body = response.json()
            check("upload: accepted (202) and returns a 32-character job id",
                  response.status_code == 202 and len(body["job_id"]) == 32)
            job_id = body["job_id"]
            view = wait_for(client, job_id, {"done", "failed", "cancelled"})
            check("the job finishes: done, progress 1.0, all 7 steps listed, result ready",
                  view["status"] == "done" and view["progress"] == 1.0 and len(view["steps"]) == 7
                  and view["result_ready"] is True and view["error"] is None, str(view)[:200])
            result = client.get(f"/api/jobs/{job_id}/result")
            picture = Image.open(io.BytesIO(result.content))
            check("result: a PNG, 2x the size of the upload (400x300)",
                  result.status_code == 200 and result.headers["content-type"] == "image/png"
                  and picture.size == (400, 300))
            jpeg = client.get(f"/api/jobs/{job_id}/result?format=jpeg&download=1")
            check("result as JPEG, offered as a download",
                  jpeg.headers["content-type"] == "image/jpeg"
                  and "attachment" in jpeg.headers["content-disposition"]
                  and Image.open(io.BytesIO(jpeg.content)).format == "JPEG")
            original = client.get(f"/api/jobs/{job_id}/original")
            check("original (before) picture is available at the working size",
                  Image.open(io.BytesIO(original.content)).size == (200, 150))
            check("responses are no-store and nosniff",
                  result.headers["cache-control"] == "no-store"
                  and result.headers["x-content-type-options"] == "nosniff")
            folders = [p.name for p in (temp / "a" / "jobs").iterdir()]
            check("the uploaded file name was ignored: only the random job id is on disk",
                  folders == [job_id])
            check("a bad result format is rejected", client.get(f"/api/jobs/{job_id}/result?format=gif").status_code == 422)

            # ---- bad input ----
            r = upload(client, data=b"this is not an image" * 40)
            check("a text file named .png is rejected with 400 and a clear message",
                  r.status_code == 400 and "could not be read" in r.json()["detail"])
            r = upload(client, data=b"MZ", name="virus.exe")
            check("a .exe is rejected with 415", r.status_code == 415)
            with mock.patch.object(config, "MAX_UPLOAD_BYTES", 200):     # smaller than the test picture
                r = upload(client)
            check("a file over the size limit is rejected with 413", r.status_code == 413
                  and "limit" in r.json()["detail"])
            check("a tiny image is rejected with 422", upload(client, data=png((10, 10))).status_code == 422)
            r = upload(client, upscale=9)
            check("an invalid option value is rejected with 422 naming the option",
                  r.status_code == 422 and "upscale" in r.json()["detail"])
            r = upload(client, upscale="abc")
            check("a non-number gives ONE readable sentence (not a technical list)",
                  r.status_code == 422 and isinstance(r.json()["detail"], str)
                  and "upscale" in r.json()["detail"])
            check("'medium' repair is not offered", upload(client, repair="medium").status_code == 422)
            check("a request without a file is rejected with 422",
                  client.post("/api/jobs", data={"upscale": "2"}).status_code == 422)

            # ---- unknown / malformed ids ----
            check("unknown and malformed job ids give 404",
                  client.get("/api/jobs/" + "0" * 32).status_code == 404
                  and client.get("/api/jobs/not-a-job").status_code == 404
                  and client.get("/api/jobs/" + job_id.upper()).status_code == 404
                  and client.get("/api/jobs/" + job_id + "/result/../../x").status_code == 404)

            # ---- delete ----
            check("delete: 204, then the job and its files are gone",
                  client.delete(f"/api/jobs/{job_id}").status_code == 204
                  and client.get(f"/api/jobs/{job_id}").status_code == 404
                  and not (temp / "a" / "jobs" / job_id).exists())

        # ============ failures inside a job ============
        with TestClient(create_app(make_manager(temp / "b", colorizer=_StubColorizer(RuntimeError(
                "secret internal detail C:\\Users\\x\\model.pt"))))) as client:
            logging.disable(logging.CRITICAL)      # this job crashes ON PURPOSE; keep the log quiet
            job_id = upload(client).json()["job_id"]
            view = wait_for(client, job_id, {"failed", "done"})
            logging.disable(logging.NOTSET)
            text = str(view)
            check("an unexpected crash: job failed, with a generic message and NO internal details",
                  view["status"] == "failed" and view["error"] == GENERIC_ERROR
                  and "secret" not in text and "C:\\" not in text and "RuntimeError" not in text)
            check("a failed job has no result (409)", client.get(f"/api/jobs/{job_id}/result").status_code == 409)

        with TestClient(create_app(make_manager(temp / "c", colorizer=_StubColorizer(
                ColorizationError("model file missing"))))) as client:
            job_id = upload(client).json()["job_id"]
            view = wait_for(client, job_id, {"failed", "done"})
            check("a step failure still gives a result, with the step marked failed and a note",
                  view["status"] == "done" and "some steps" in view["message"]
                  and any(s["status"] == "failed" for s in view["steps"])
                  and any("model file missing" in n for n in view["notes"]))

        # ============ cancel, queue, busy ============
        with TestClient(create_app(make_manager(temp / "d", colorizer=SlowColorizer(1.0)))) as client:
            first = upload(client).json()["job_id"]
            wait_for(client, first, {"running"})
            check("a running job cannot be deleted (409) and has no result yet (409)",
                  client.delete(f"/api/jobs/{first}").status_code == 409
                  and client.get(f"/api/jobs/{first}/result").status_code == 409)
            second = upload(client).json()["job_id"]
            queued = client.get(f"/api/jobs/{second}").json()
            check("a second job waits in the queue at position 0",
                  queued["status"] == "queued" and queued["queue_position"] == 0)
            check("cancelling a queued job takes effect immediately",
                  client.post(f"/api/jobs/{second}/cancel").json()["status"] == "cancelled")
            client.post(f"/api/jobs/{first}/cancel")
            view = wait_for(client, first, {"cancelled", "done"})
            check("cancelling a running job stops it between steps; nothing is left on disk",
                  view["status"] == "cancelled" and not (temp / "d" / "jobs" / first).exists())
            check("cancelling a finished job gives 409",
                  client.post(f"/api/jobs/{first}/cancel").status_code == 409)

        with TestClient(create_app(make_manager(temp / "e", colorizer=SlowColorizer(1.0),
                                                max_active=2))) as client:
            ids = [upload(client).json()["job_id"] for _ in range(2)]
            busy = upload(client)
            check("when too many jobs are active: 503 with Retry-After and a friendly message",
                  busy.status_code == 503 and "Retry-After" in busy.headers and "busy" in busy.json()["detail"])
            for job_id in ids:
                client.post(f"/api/jobs/{job_id}/cancel")
            wait_for(client, ids[0], {"cancelled", "done"})

        # ============ expiry and clean-up ============
        now = [1000.0]
        manager = make_manager(temp / "f", clock=lambda: now[0], ttl=100)
        with TestClient(create_app(manager)) as client:
            job_id = upload(client).json()["job_id"]
            wait_for(client, job_id, {"done"})
            now[0] += 101
            upload(client, data=png((100, 80)))          # any new upload triggers the purge
            check("a finished job is deleted after its time limit, with its files",
                  client.get(f"/api/jobs/{job_id}").status_code == 404
                  and not (temp / "f" / "jobs" / job_id).exists())

        orphan = temp / "g" / "jobs" / ("ab" * 16)
        orphan.mkdir(parents=True)
        (orphan / "result.png").write_bytes(b"left over from an earlier run")
        manager = make_manager(temp / "g")
        with TestClient(create_app(manager)) as client:
            check("leftovers from an earlier run are removed at start-up", not orphan.exists())
            job_id = upload(client).json()["job_id"]
            wait_for(client, job_id, {"done"})
            has_files = (temp / "g" / "jobs" / job_id).exists()
        check("results exist while running, and are all deleted when the server stops",
              has_files and not any((temp / "g" / "jobs").iterdir()))

    print("-" * 62)
    print(" RESULT:", "all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Photo restoration web API")
    parser.add_argument("--host", default=os.environ.get("PHOTO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PHOTO_PORT", "8000")))
    parser.add_argument("--self-test", action="store_true", help="test the API with stand-in models")
    arguments = parser.parse_args()
    if arguments.self_test:
        return _run_self_test()

    import uvicorn
    if arguments.host not in ("127.0.0.1", "localhost", "::1"):
        print(" WARNING: this server has NO login. Anyone who can reach it can use it.\n"
              "          Only use a host other than 127.0.0.1 on a network you trust.")
    config.ensure_directories()
    print(f"\n Photo restoration server starting.\n"
          f" Open  http://{arguments.host}:{arguments.port}/docs  in your browser.\n"
          f" Stop it with Ctrl+C. Results are deleted when it stops.\n")
    uvicorn.run(create_app(), host=arguments.host, port=arguments.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
