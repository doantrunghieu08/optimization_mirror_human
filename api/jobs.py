"""
api/jobs.py — Quản lý job (upload -> inference -> render) trong bộ nhớ.

Mỗi job có thư mục riêng dưới runs/{job_id}/ (inputs/ + output/) để nhiều
request không ghi đè lẫn nhau. inference.py/render_video.py vốn là script
CLI thao tác trên 1 bộ config/đường dẫn cố định — JobManager chạy chúng qua
run_inference()/render_video() (hàm Python thuần, không qua subprocess) trên
1 ThreadPoolExecutor DUY NHẤT một worker, vì pipeline dùng chung state
torch/CUDA global (SMPLForwardPass, device mặc định) nên không an toàn khi
chạy song song nhiều job cùng lúc.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class JobStatus(str, Enum):
    CREATED = "created"
    INFERENCE_RUNNING = "inference_running"
    INFERENCE_DONE = "inference_done"
    RENDER_RUNNING = "render_running"
    RENDER_DONE = "render_done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    root_dir: Path
    status: JobStatus = JobStatus.CREATED
    error: Optional[str] = None
    inference_pkl: Optional[str] = None
    output_video: Optional[str] = None
    uploaded_files: dict = field(default_factory=dict)

    @property
    def inputs_dir(self) -> Path:
        return self.root_dir / "inputs"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "output"


class JobManager:
    def __init__(self, base_dir: str = "runs"):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def create_job(self) -> Job:
        job_id = uuid.uuid4().hex[:12]
        root = self._base_dir / job_id
        (root / "inputs" / "real" / "preprocess").mkdir(parents=True, exist_ok=True)
        (root / "inputs" / "mirror" / "preprocess").mkdir(parents=True, exist_ok=True)
        (root / "output").mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, root_dir=root)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def wait_for_job(self, job_id: str, timeout: float = 300.0, poll_interval: float = 0.1) -> Job:
        import time
        start_time = time.time()
        job = self.get_job(job_id)
        while job.status in (JobStatus.INFERENCE_RUNNING, JobStatus.RENDER_RUNNING):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
            time.sleep(poll_interval)
        return job

    # ── Inference ────────────────────────────────────────────────────────
    def submit_inference(self, job: Job, config: dict, bypass_refit: bool) -> None:
        job.status = JobStatus.INFERENCE_RUNNING
        job.error = None
        self._executor.submit(self._run_inference, job, config, bypass_refit)

    def _run_inference(self, job: Job, config: dict, bypass_refit: bool) -> None:
        from inference import run_inference  # import trễ — tránh load torch/SMPL lúc khởi động API

        try:
            job.inference_pkl = run_inference(config, bypass_refit=bypass_refit)
            job.status = JobStatus.INFERENCE_DONE
        except Exception:
            job.status = JobStatus.FAILED
            job.error = traceback.format_exc()

    # ── Render ───────────────────────────────────────────────────────────
    def submit_render(
        self, job: Job, config: dict, pkl_path: str, input_video: str, output_video: str
    ) -> None:
        job.status = JobStatus.RENDER_RUNNING
        job.error = None
        self._executor.submit(self._run_render, job, config, pkl_path, input_video, output_video)

    def _run_render(
        self, job: Job, config: dict, pkl_path: str, input_video: str, output_video: str
    ) -> None:
        from render_video import render_video  # import trễ, cùng lý do như trên

        try:
            max_frames = config.get("visualization", {}).get("max_frames", -1)
            render_video(config, pkl_path, input_video, output_video, max_frames=max_frames)
            job.output_video = output_video
            job.status = JobStatus.RENDER_DONE
        except Exception:
            job.status = JobStatus.FAILED
            job.error = traceback.format_exc()


# Instance dùng chung cho toàn bộ app (in-memory — mất khi restart process).
job_manager = JobManager()
