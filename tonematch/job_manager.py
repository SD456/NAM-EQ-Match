"""Shared job manager for background tone-match runs.

Provides:
- ``JobManager``: thread-pooled queue with ``Job`` dataclass tracking status.
- ``_run_match_core``: Gradio-free core matching logic extracted from app.do_match.

Both batch mode and multi-brand tabs submit to the same JobManager instance.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .nam_backend import load_captures
from .pipeline import MatchOutput, run_match


# ---------------------------------------------------------------------------
# Job data model
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    label: str
    target: str
    di: str
    models_dir: str
    params: dict
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    result: MatchOutput | None = None
    error: str | None = None
    stem_path: str | None = None
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Core matching logic (Gradio-free)
# ---------------------------------------------------------------------------

def _run_match_core(
    target_path: str,
    di_path: str,
    models_dir: str,
    params: dict,
    progress_cb=None,
) -> tuple[MatchOutput, str | None]:
    """Run the full tone-match pipeline.

    Returns ``(MatchOutput, stem_path_or_None)``.
    Raises ``ValueError`` / ``RuntimeError`` on failure (callers translate to
    ``gr.Error`` for the Gradio path).
    """
    if target_path is None or di_path is None:
        raise ValueError("Please provide both a target recording and a DI track.")

    stem_path = None
    target = target_path
    demix = params.get("demix", False)
    stem = params.get("stem", "guitar")

    if demix:
        from .stems import extract_stem
        stem_dir = tempfile.mkdtemp(prefix="tonematch_stems_")
        target = extract_stem(
            target_path,
            stem_dir,
            stem=stem,
            progress_cb=progress_cb,
        )
        stem_path = target

    limit_val = int(params.get("limit") or 0) or None

    device = params.get("device", "auto")
    library_files = params.get("library_files")
    if library_files:
        captures, load_errors = [], []
        captures.extend(load_captures(library_files, limit=limit_val, errors_out=load_errors, device=device))
    else:
        sources = []
        if models_dir and os.path.isdir(models_dir.strip()):
            sources.append(models_dir.strip())
        if not sources:
            raise ValueError("Point me at a folder of .nam files.")
        captures, load_errors = [], []
        for s in sources:
            captures.extend(load_captures(s, limit=limit_val, errors_out=load_errors, device=device))
    seen, uniq = set(), []
    for c in captures:
        if c.path not in seen:
            seen.add(c.path)
            uniq.append(c)
    cap_limit = int(params.get("limit") or 0)
    captures = uniq[:cap_limit] if cap_limit else uniq
    if not captures:
        detail = "\n".join(
            f"  {os.path.basename(p)} — {msg}" for p, msg in load_errors[:5]
        ) or "No .nam files were found."
        raise ValueError(
            f"Could not load any NAM captures:\n{detail}\n\n"
            "Run `python -m tonematch.doctor your_model.nam` for a full diagnosis."
        )

    out_dir = tempfile.mkdtemp(prefix="tonematch_")
    result = run_match(
        target,
        di_path,
        captures,
        out_dir,
        gain_range_db=(float(params.get("gain_lo", -12)), float(params.get("gain_hi", 12))),
        refine_top=max(int(params.get("refine_top", 5)), int(params.get("render_top", 1))),
        render_top=int(params.get("render_top", 1)),
        preview_s=float(params.get("preview_s", 30)),
        progress_cb=progress_cb,
    )
    return result, stem_path


# ---------------------------------------------------------------------------
# JobManager
# ---------------------------------------------------------------------------

class JobManager:
    """Thread-pooled job queue.  Queue-by-default (max_workers=1)."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.max_workers = 1

    def set_concurrency(self, max_workers: int) -> None:
        self.max_workers = max(1, max_workers)
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=False)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            import torch
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // self.max_workers))
        except ImportError:
            pass

    def submit(self, target, di, models_dir, params, label=None) -> str:
        job = Job(
            id=str(uuid.uuid4()),
            label=label or os.path.basename(models_dir.rstrip("/\\")) or "Match",
            target=target,
            di=di,
            models_dir=models_dir,
            params=params,
        )
        with self.lock:
            self.jobs[job.id] = job
        self.executor.submit(self._run, job.id)
        return job.id

    def _run(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.status = "running"
        try:
            def cb(frac, msg):
                job.progress = min(frac, 1.0)
                job.message = msg
            result, stem_path = _run_match_core(
                job.target, job.di, job.models_dir, job.params,
                progress_cb=cb,
            )
            job.result = result
            job.stem_path = stem_path
            job.status = "done"
        except Exception as e:
            job.status = "error"
            job.error = str(e)

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [
                {"id": j.id, "label": j.label, "status": j.status,
                 "progress": j.progress, "message": j.message}
                for j in self.jobs.values()
            ]
