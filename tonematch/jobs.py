"""Background job queue for match runs.

One worker thread executes jobs serially (NAM inference saturates the CPU/GPU,
so parallel jobs would just fight each other). The GUI polls `snapshot()`.

Inspired by community PR #2 (job queue concept), re-implemented to integrate
with the current pipeline (device selection, preview length, tone stack).
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field


@dataclass
class Job:
    id: int
    label: str
    fn: object
    kwargs: dict
    status: str = "queued"  # queued | running | done | error | canceled
    progress: float = 0.0
    message: str = "waiting in queue"
    result: object = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    @property
    def runtime_s(self) -> float | None:
        if self.started is None:
            return None
        return (self.finished or time.time()) - self.started


class JobManager:
    def __init__(self):
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._next_id = 1
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # -- public API -----------------------------------------------------------

    def submit(self, label: str, fn, **kwargs) -> int:
        """Queue `fn(progress_cb=..., **kwargs)`. Returns the job id."""
        with self._lock:
            job = Job(id=self._next_id, label=label, fn=fn, kwargs=kwargs)
            self._next_id += 1
            self._jobs.append(job)
        self._wake.set()
        return job.id

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            for j in self._jobs:
                if j.id == job_id:
                    return j
        return None

    def cancel(self, job_id: int) -> bool:
        """Cancel a job that hasn't started yet."""
        with self._lock:
            for j in self._jobs:
                if j.id == job_id and j.status == "queued":
                    j.status = "canceled"
                    j.message = "canceled"
                    j.finished = time.time()
                    return True
        return False

    def running(self) -> bool:
        with self._lock:
            return any(j.status == "running" for j in self._jobs)

    def snapshot(self) -> list[dict]:
        """Job list for the UI, newest first."""
        with self._lock:
            jobs = list(self._jobs)
        out = []
        for j in reversed(jobs):
            rt = j.runtime_s
            out.append(
                {
                    "id": j.id,
                    "label": j.label,
                    "status": j.status,
                    "progress": f"{int(100 * j.progress)}%" if j.status == "running" else "",
                    "message": j.message,
                    "runtime": f"{rt:.0f}s" if rt is not None else "",
                }
            )
        return out

    def completed(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs if j.status == "done"]

    # -- worker ----------------------------------------------------------------

    def _next_queued(self) -> Job | None:
        with self._lock:
            for j in self._jobs:
                if j.status == "queued":
                    return j
        return None

    def _loop(self):
        while True:
            job = self._next_queued()
            if job is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            job.status = "running"
            job.started = time.time()
            job.message = "starting"

            def progress_cb(frac, msg):
                job.progress = float(frac)
                job.message = str(msg)

            try:
                job.result = job.fn(progress_cb=progress_cb, **job.kwargs)
                job.status = "done"
                job.progress = 1.0
                job.message = "finished"
            except Exception as e:  # noqa: BLE001 - report in UI
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.message = job.error
                traceback.print_exc()
            finally:
                job.finished = time.time()
