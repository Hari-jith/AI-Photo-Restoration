"""
jobs.py

Runs restoration jobs in the background, one at a time, and remembers their
state. The web API (main.py) is a thin layer on top of this.

Design (kept deliberately simple; no database, no Redis):
  * ONE worker thread takes jobs from a queue. The models are heavy and lock
    themselves anyway, so running two at once would only fight for memory.
  * Jobs live in memory. Result pictures are written to outputs/jobs/<id>/.
  * Everything is TEMPORARY: results are deleted after JOB_TTL_SECONDS, when a
    job is deleted, when the server stops, and any leftovers from a previous
    run are removed at start-up. (Your photos are not kept around.)
  * Job ids are random 128-bit values and must match a strict pattern, so an id
    can never be used to reach any other file.

    manager = JobManager()
    manager.start()
    job = manager.submit(pil_image, PipelineOptions())
    manager.public_view(job)        # dict for the API: status, progress, notes...
"""

import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from PIL import Image

from src import config
from src.logging_setup import get_logger
from src.pipeline import PipelineCancelled, PipelineOptions, RestorationPipeline

logger = get_logger(__name__)

JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
JOBS_DIR = config.OUTPUTS_DIR / "jobs"
MAX_ACTIVE_JOBS = 5            # queued + running; more are refused ("server busy")
MAX_STORED_JOBS = 50           # finished jobs kept; the oldest are removed first
JOB_TTL_SECONDS = 3600         # finished jobs are deleted after one hour
PURGE_INTERVAL_SECONDS = 30
GENERIC_ERROR = ("Something went wrong while restoring this photo. Please try again, "
                 "or try different options. (The details are in the server log.)")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


FINISHED_STATES = {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}


class QueueFullError(Exception):
    """Too many jobs are waiting or running."""


class JobNotFoundError(Exception):
    """No such job (the same error is used for malformed ids)."""


class JobStateError(Exception):
    """The requested action does not fit the job's current state."""


@dataclass
class Job:
    id: str
    options: PipelineOptions
    created: float
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = "Waiting in the queue"
    started: float | None = None
    finished: float | None = None
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    analysis: dict = field(default_factory=dict)
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    image: Image.Image | None = None      # the uploaded photo, dropped once processed


class JobManager:
    def __init__(self, pipeline: RestorationPipeline | None = None, jobs_dir: Path | None = None,
                 clock=time.time, ttl: float = JOB_TTL_SECONDS,
                 max_active: int = MAX_ACTIVE_JOBS, max_stored: int = MAX_STORED_JOBS):
        self.pipeline = pipeline or RestorationPipeline()
        self.jobs_dir = Path(jobs_dir) if jobs_dir else JOBS_DIR
        self.clock = clock                 # replaceable, so tests can fast-forward time
        self.ttl, self.max_active, self.max_stored = ttl, max_active, max_stored
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []        # job ids, oldest first
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_purge = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the worker thread and remove leftovers of an earlier run."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._remove_orphan_directories()
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="restoration-worker",
                                        daemon=True)
        self._thread.start()
        logger.info("Job worker started (jobs folder: %s)", self.jobs_dir)

    def stop(self) -> None:
        """Stop the worker and delete every stored result."""
        self._stop.set()
        with self._lock:
            for job in self._jobs.values():
                if job.status == JobStatus.RUNNING:
                    job.cancel_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        with self._lock:
            for job_id in list(self._jobs):
                self._remove_files(job_id)
            self._jobs.clear()
            self._order.clear()
        logger.info("Job worker stopped; stored results deleted")

    # ------------------------------------------------------------------ public API
    def submit(self, image: Image.Image, options: PipelineOptions) -> Job:
        with self._lock:
            self._purge_expired(force=True)
            active = sum(1 for j in self._jobs.values() if j.status not in FINISHED_STATES)
            if active >= self.max_active:
                raise QueueFullError(
                    f"The server is busy ({active} photos are waiting or being restored). "
                    "Please try again in a few minutes.")
            job = Job(id=uuid.uuid4().hex, options=options, created=self.clock(), image=image)
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._queue.put(job.id)
        logger.info("Job %s queued (%dx%d)", job.id[:8], image.width, image.height)
        return job

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id) if JOB_ID_PATTERN.match(job_id or "") else None
        if job is None:
            raise JobNotFoundError("Job not found.")
        return job

    def queue_position(self, job: Job) -> int | None:
        """0 = next in line; None if the job is not waiting."""
        with self._lock:
            if job.status != JobStatus.QUEUED:
                return None
            waiting = [self._jobs[i] for i in self._order
                       if i in self._jobs and self._jobs[i].status == JobStatus.QUEUED]
            return waiting.index(job)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status not in FINISHED_STATES)

    def public_view(self, job: Job) -> dict:
        """Everything the client may see about a job (never file paths)."""
        with self._lock:
            end = job.finished or self.clock()
            return {
                "job_id": job.id,
                "status": job.status.value,
                "progress": round(job.progress, 3),
                "message": job.message,
                "queue_position": self.queue_position(job),
                "created_at": job.created,
                "elapsed_seconds": round(end - job.started, 1) if job.started else 0.0,
                "options": asdict(job.options),
                "steps": list(job.steps),
                "notes": list(job.notes),
                "analysis": dict(job.analysis),
                "error": job.error,
                "result_ready": job.status == JobStatus.DONE,
            }

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self.get(job_id)
            if job.status in FINISHED_STATES:
                raise JobStateError("This job has already finished.")
            if job.status == JobStatus.QUEUED:
                self._finish(job, JobStatus.CANCELLED, "Cancelled before it started")
            else:                                       # running: stops between two steps
                job.cancel_event.set()
                job.message = "Cancelling (the current step must finish first)"
            return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self.get(job_id)
            if job.status == JobStatus.RUNNING:
                raise JobStateError("This job is running. Cancel it first.")
            if job.status == JobStatus.QUEUED:
                self._finish(job, JobStatus.CANCELLED, "Deleted")
            self._remove_job(job_id)

    def result_path(self, job_id: str, kind: str) -> Path:
        """Path of 'result' or 'original' picture of a finished job."""
        if kind not in ("result", "original"):
            raise ValueError("kind must be 'result' or 'original'.")
        with self._lock:
            job = self.get(job_id)
            if job.status != JobStatus.DONE:
                raise JobStateError("The result is not ready yet." if job.status in
                                    (JobStatus.QUEUED, JobStatus.RUNNING)
                                    else "This job has no result.")
            path = self.jobs_dir / job.id / f"{kind}.png"
            if not path.is_file():
                raise JobStateError("The result is no longer available.")
            return path

    # ------------------------------------------------------------------ internals
    def _finish(self, job: Job, status: JobStatus, message: str, error: str | None = None):
        job.status, job.message, job.error = status, message, error
        job.finished = self.clock()
        job.image = None                    # never keep the uploaded picture longer than needed
        if status != JobStatus.DONE:
            job.progress = min(job.progress, 0.999)

    def _remove_files(self, job_id: str) -> None:
        if JOB_ID_PATTERN.match(job_id):
            shutil.rmtree(self.jobs_dir / job_id, ignore_errors=True)

    def _remove_job(self, job_id: str) -> None:
        self._remove_files(job_id)
        self._jobs.pop(job_id, None)
        if job_id in self._order:
            self._order.remove(job_id)

    def _remove_orphan_directories(self) -> None:
        if not self.jobs_dir.is_dir():
            return
        for entry in self.jobs_dir.iterdir():
            if entry.is_dir() and JOB_ID_PATTERN.match(entry.name):
                shutil.rmtree(entry, ignore_errors=True)

    def _purge_expired(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._last_purge < PURGE_INTERVAL_SECONDS:
            return
        self._last_purge = now
        with self._lock:
            for job_id in [j.id for j in self._jobs.values()
                           if j.status in FINISHED_STATES and j.finished is not None
                           and now - j.finished > self.ttl]:
                self._remove_job(job_id)
            finished = [i for i in self._order
                        if i in self._jobs and self._jobs[i].status in FINISHED_STATES]
            for job_id in finished[:max(0, len(finished) - self.max_stored)]:
                self._remove_job(job_id)

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.25)
            except queue.Empty:
                self._purge_expired()
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status != JobStatus.QUEUED:
                    continue                # cancelled or deleted while waiting
                job.status, job.started = JobStatus.RUNNING, self.clock()
                job.message = "Starting"
            self._run(job)

    def _run(self, job: Job) -> None:
        def progress(fraction: float, message: str) -> None:
            with self._lock:
                job.progress = max(job.progress, min(fraction, 0.999))
                if not job.cancel_event.is_set():
                    job.message = message

        try:
            result = self.pipeline.run(job.image, job.options, progress=progress,
                                       cancel=job.cancel_event)
            directory = self.jobs_dir / job.id
            directory.mkdir(parents=True, exist_ok=True)
            result.original.save(directory / "original.png", "PNG")
            result.image.save(directory / "result.png", "PNG")
            with self._lock:
                job.steps = [asdict(step) for step in result.steps]
                job.notes = list(result.notes)
                job.analysis = dict(result.analysis)
                job.progress = 1.0
                skipped = bool(result.failed_steps)
                self._finish(job, JobStatus.DONE,
                             "Done (some steps could not be completed)" if skipped else "Done")
                job.progress = 1.0
            logger.info("Job %s done in %.1f s", job.id[:8], job.finished - job.started)
        except PipelineCancelled:
            with self._lock:
                self._finish(job, JobStatus.CANCELLED, "Cancelled")
            self._remove_files(job.id)
            logger.info("Job %s cancelled", job.id[:8])
        except Exception:                   # a bug or an unexpected failure
            logger.exception("Job %s failed", job.id[:8])    # details go to the log only
            with self._lock:
                self._finish(job, JobStatus.FAILED, "Failed", GENERIC_ERROR)
            self._remove_files(job.id)
        finally:
            with self._lock:
                job.image = None
