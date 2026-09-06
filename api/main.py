"""
api/main.py — Lớp API phủ lên pipeline upload -> inference -> render -> download.

Chạy:
    uvicorn api.main:app --reload --port 8000

Luồng sử dụng:
    1. POST /jobs                                    -> {job_id}
    2. POST /jobs/{job_id}/upload/{file_type}         -> upload từng file
       file_type ∈ {video, real_pt, mirror_pt, vitpose_real, vitpose_mirror}
    3. POST /jobs/{job_id}/infer                      -> chạy belief-fusion refit (chạy nền)
    4. POST /jobs/{job_id}/render                      -> render video overlay (chạy nền)
    5. GET  /jobs/{job_id}                             -> theo dõi trạng thái
    6. GET  /jobs/{job_id}/download/{artifact}         -> tải kết quả
       artifact ∈ {inference_pkl, diagnostics_pkl, video}

Job chạy inference/render trên 1 worker thread duy nhất (xem api/jobs.py) —
gọi /infer hoặc /render nhiều lần trước khi job trước hoàn tất sẽ bị xếp
hàng tuần tự, không chạy song song.
"""

from __future__ import annotations

import shutil
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.config_utils import build_job_config
from api.jobs import Job, JobStatus, job_manager

app = FastAPI(
    title="Optimization Mirror Human API",
    version="1.0.0",
    description="""
API điều phối pipeline fusion pose 3D từ video view thật và view gương.

Tạo job, upload dữ liệu (2 file .pt HMR4D, 2 file ViTPose, 1 video gốc), chạy inference/render, sau đó poll trạng thái và tải artifact.
Inference và render chạy nền trên một worker duy nhất, nên các job được xử lý tuần tự.
""",
    openapi_tags=[
        {"name": "System", "description": "Kiểm tra trạng thái dịch vụ."},
        {"name": "Jobs", "description": "Tạo và theo dõi job xử lý."},
        {"name": "Files", "description": "Upload input và download kết quả của job."},
        {"name": "Processing", "description": "Chạy inference và render video nền."},
    ],
)


class UploadFileType(str, Enum):
    VIDEO = "video"
    REAL_PT = "real_pt"
    MIRROR_PT = "mirror_pt"
    VITPOSE_REAL = "vitpose_real"
    VITPOSE_MIRROR = "vitpose_mirror"


class DownloadArtifact(str, Enum):
    INFERENCE_PKL = "inference_pkl"
    DIAGNOSTICS_PKL = "diagnostics_pkl"
    VIDEO = "video"


class InferRequest(BaseModel):
    bypass_refit: bool = Field(False, description="Bỏ qua pose refit, dùng pose real gốc để debug.")


class RenderRequest(BaseModel):
    run_inference_if_missing: bool = Field(
        False, description="Tự chạy inference nếu chưa có inference_results.pkl."
    )
    bypass_refit: bool = Field(False, description="Bỏ qua pose refit khi inference tự động chạy.")


class JobCreatedResponse(BaseModel):
    job_id: str = Field(description="ID của job mới tạo.", examples=["a1b2c3d4e5f6"])


class JobStatusResponse(BaseModel):
    job_id: str = Field(description="ID của job.")
    status: JobStatus = Field(description="Trạng thái xử lý hiện tại.")
    error: str | None = Field(None, description="Chi tiết lỗi khi trạng thái là failed.")
    uploaded_files: dict = Field(description="Các input đã upload, theo file_type.")
    has_inference_pkl: bool = Field(description="Đã có file inference_results.pkl.")
    has_output_video: bool = Field(description="Đã có video output.")


def _job_or_404(job_id: str) -> Job:
    try:
        return job_manager.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


def _job_status_response(job: Job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        error=job.error,
        uploaded_files=job.uploaded_files,
        has_inference_pkl=job.inference_pkl is not None,
        has_output_video=job.output_video is not None,
    )


@app.get("/health", tags=["System"], summary="Kiểm tra sức khỏe dịch vụ")
def health() -> dict:
    return {"status": "ok"}


_UPLOAD_DEST = {
    UploadFileType.REAL_PT: lambda job: job.inputs_dir / "real" / "hmr4d_results.pt",
    UploadFileType.MIRROR_PT: lambda job: job.inputs_dir / "mirror" / "hmr4d_results.pt",
    UploadFileType.VITPOSE_REAL: lambda job: job.inputs_dir / "real" / "preprocess" / "vitpose_real.pt",
    UploadFileType.VITPOSE_MIRROR: lambda job: job.inputs_dir / "mirror" / "preprocess" / "vitpose_mirror.pt",
}
_PT_TYPES = {UploadFileType.REAL_PT, UploadFileType.MIRROR_PT, UploadFileType.VITPOSE_REAL, UploadFileType.VITPOSE_MIRROR}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _process_and_save_upload(job: Job, file_type: UploadFileType, file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix.lower()

    if file_type in _PT_TYPES:
        if suffix != ".pt":
            raise HTTPException(status_code=400, detail=f"{file_type.value} phải là file .pt, nhận được: {file.filename}")
        dest = _UPLOAD_DEST[file_type](job)
    else:  # VIDEO
        if suffix not in _VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"video phải có đuôi {sorted(_VIDEO_EXTENSIONS)}, nhận được: {file.filename}",
            )
        dest = job.inputs_dir / f"input_video{suffix}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    file.file.close()

    job.uploaded_files[file_type.value] = str(dest)
    return dest


@app.post(
    "/jobs",
    response_model=JobStatusResponse,
    status_code=201,
    tags=["Jobs"],
    summary="Tạo job mới và upload các file input (2 file .pt, 2 file ViTPose, 1 video gốc)",
    description=(
        "Tạo job mới. Có thể upload đồng thời 5 file: `real_pt`, `mirror_pt`, "
        "`vitpose_real`, `vitpose_mirror`, và `video`. Nếu `auto_infer=True`, job sẽ "
        "tự động kích hoạt inference ngay sau khi tạo và upload xong."
    ),
)
def create_job(
    real_pt: UploadFile | None = File(None, description="File .pt pose real (HMR4D)"),
    mirror_pt: UploadFile | None = File(None, description="File .pt pose mirror (HMR4D)"),
    vitpose_real: UploadFile | None = File(None, description="File .pt ViTPose real"),
    vitpose_mirror: UploadFile | None = File(None, description="File .pt ViTPose mirror"),
    video: UploadFile | None = File(None, description="File video gốc (.mp4, .avi, .mov, .mkv)"),
    auto_infer: bool = Form(False, description="Tự động chạy inference ngay sau khi upload"),
    bypass_refit: bool = Form(False, description="Bỏ qua refit khi tự động chạy inference"),
) -> JobStatusResponse:
    job = job_manager.create_job()

    if real_pt is not None:
        _process_and_save_upload(job, UploadFileType.REAL_PT, real_pt)
    if mirror_pt is not None:
        _process_and_save_upload(job, UploadFileType.MIRROR_PT, mirror_pt)
    if vitpose_real is not None:
        _process_and_save_upload(job, UploadFileType.VITPOSE_REAL, vitpose_real)
    if vitpose_mirror is not None:
        _process_and_save_upload(job, UploadFileType.VITPOSE_MIRROR, vitpose_mirror)
    if video is not None:
        _process_and_save_upload(job, UploadFileType.VIDEO, video)

    if auto_infer:
        config = build_job_config(job)
        real_pt_path = Path(config["data"]["real_dir"])
        mirror_pt_path = Path(config["data"]["mirror_dir"])
        if not real_pt_path.exists() or not mirror_pt_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Thiếu file real_pt/mirror_pt — không thể tự động chạy inference",
            )
        job_manager.submit_inference(job, config, bypass_refit=bypass_refit)

    return _job_status_response(job)


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
    summary="Lấy trạng thái job",
    responses={404: {"description": "Không tìm thấy job."}},
)
def get_job_status(job_id: str) -> JobStatusResponse:
    return _job_status_response(_job_or_404(job_id))


# ── 1. Upload video / .pt (HMR4D) / ViTPose ─────────────────────────────────
@app.post(
    "/jobs/{job_id}/upload/{file_type}",
    tags=["Files"],
    summary="Upload video hoặc dữ liệu pose đơn lẻ",
    description=(
        "`file_type` chấp nhận: `video`, `real_pt`, `mirror_pt`, `vitpose_real`, "
        "`vitpose_mirror`. Video phải là .mp4, .avi, .mov, .mkv; các loại còn lại phải là .pt."
    ),
    responses={
        400: {"description": "Định dạng file không hợp lệ."},
        404: {"description": "Không tìm thấy job."},
    },
)
def upload_file(job_id: str, file_type: UploadFileType, file: UploadFile = File(...)) -> dict:
    job = _job_or_404(job_id)
    dest = _process_and_save_upload(job, file_type, file)
    return {"job_id": job.job_id, "file_type": file_type.value, "saved_to": str(dest)}


# ── 2. Chạy inference (belief-theory refit) ─────────────────────────────────
@app.post(
    "/jobs/{job_id}/infer",
    response_model=JobStatusResponse,
    tags=["Processing"],
    summary="Chạy inference belief-fusion theo job_id",
    description="Khởi chạy inference cho job. Đặt `wait=True` để chờ kết quả hoàn tất trước khi response.",
    responses={
        400: {"description": "Thiếu input bắt buộc."},
        404: {"description": "Không tìm thấy job."},
        409: {"description": "Job đang chạy."},
        500: {"description": "Inference thất bại."},
        504: {"description": "Quá thời gian chờ khi wait=True."},
    },
)
def run_infer(
    job_id: str,
    req: InferRequest = InferRequest(),
    wait: bool = Query(False, description="Đợi inference chạy xong trước khi trả response"),
    timeout: float = Query(300.0, description="Thời gian chờ tối đa khi wait=True (giây)"),
) -> JobStatusResponse:
    job = _job_or_404(job_id)
    if job.status in (JobStatus.INFERENCE_RUNNING, JobStatus.RENDER_RUNNING):
        raise HTTPException(status_code=409, detail=f"Job đang chạy ({job.status.value}), vui lòng đợi")

    config = build_job_config(job)
    real_pt = Path(config["data"]["real_dir"])
    mirror_pt = Path(config["data"]["mirror_dir"])
    if not real_pt.exists() or not mirror_pt.exists():
        raise HTTPException(
            status_code=400,
            detail="Thiếu file real_pt/mirror_pt — upload qua /jobs/{job_id}/upload/{real_pt,mirror_pt} hoặc lúc POST /jobs trước",
        )

    job_manager.submit_inference(job, config, bypass_refit=req.bypass_refit)

    if wait:
        try:
            job_manager.wait_for_job(job_id, timeout=timeout)
        except TimeoutError:
            raise HTTPException(status_code=504, detail=f"Quá thời gian chờ ({timeout}s) cho job {job_id}")

        if job.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=f"Inference thất bại: {job.error}")

    return _job_status_response(job)


# ── 3. Render video ──────────────────────────────────────────────────────────
@app.post(
    "/jobs/{job_id}/render",
    response_model=JobStatusResponse,
    tags=["Processing"],
    summary="Render video overlay",
    description=(
        "Cần upload `video` và có kết quả inference. Đặt `run_inference_if_missing=true` "
        "để tự chạy inference khi chưa có kết quả."
    ),
    responses={
        400: {"description": "Thiếu video hoặc kết quả inference."},
        404: {"description": "Không tìm thấy job."},
        409: {"description": "Job đang chạy."},
        500: {"description": "Inference tự động thất bại."},
    },
)
def run_render(job_id: str, req: RenderRequest = RenderRequest()) -> JobStatusResponse:
    job = _job_or_404(job_id)
    if job.status in (JobStatus.INFERENCE_RUNNING, JobStatus.RENDER_RUNNING):
        raise HTTPException(status_code=409, detail=f"Job đang chạy ({job.status.value}), vui lòng đợi")

    config = build_job_config(job)
    input_video = Path(job.uploaded_files.get("video", config["visualization"]["input_video"]))
    if not input_video.exists():
        raise HTTPException(status_code=400, detail="Thiếu video gốc — upload trước khi render")
    config["visualization"]["input_video"] = str(input_video)

    pkl_path = job.inference_pkl or config["visualization"]["output_dir"] + "/inference_results.pkl"
    if not Path(pkl_path).exists():
        if not req.run_inference_if_missing:
            raise HTTPException(
                status_code=400,
                detail="Chưa có inference_results.pkl — gọi /jobs/{job_id}/infer trước, "
                "hoặc đặt run_inference_if_missing=true để tự chạy inference luôn.",
            )
        from inference import run_inference

        job.status = JobStatus.INFERENCE_RUNNING
        try:
            pkl_path = run_inference(config, bypass_refit=req.bypass_refit)
            job.inference_pkl = pkl_path
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            raise HTTPException(status_code=500, detail=f"Inference thất bại: {exc}")

    job_manager.submit_render(
        job, config, pkl_path, str(input_video), config["visualization"]["output_video"]
    )
    return _job_status_response(job)


# ── 4. Download output ───────────────────────────────────────────────────────
_ARTIFACT_FILENAME = {
    DownloadArtifact.INFERENCE_PKL: "inference_results.pkl",
    DownloadArtifact.DIAGNOSTICS_PKL: "belief_diagnostics.pkl",
    DownloadArtifact.VIDEO: "fused_output.mp4",
}


@app.get(
    "/jobs/{job_id}/download/{artifact}",
    response_class=FileResponse,
    tags=["Files"],
    summary="Tải artifact kết quả của job",
    description="`artifact` chấp nhận: `inference_pkl`, `diagnostics_pkl`, hoặc `video`. Đặt `wait=True` để chờ job nếu đang chạy.",
    responses={
        404: {"description": "Không tìm thấy job hoặc artifact chưa sẵn sàng."},
        500: {"description": "Job bị lỗi."},
        504: {"description": "Quá thời gian chờ khi wait=True."},
    },
)
def download_artifact(
    job_id: str,
    artifact: DownloadArtifact,
    wait: bool = Query(False, description="Đợi job xử lý hoàn tất nếu đang chạy"),
    timeout: float = Query(300.0, description="Thời gian chờ tối đa khi wait=True (giây)"),
) -> FileResponse:
    job = _job_or_404(job_id)
    if wait and job.status in (JobStatus.INFERENCE_RUNNING, JobStatus.RENDER_RUNNING):
        try:
            job_manager.wait_for_job(job_id, timeout=timeout)
        except TimeoutError:
            raise HTTPException(status_code=504, detail=f"Quá thời gian chờ ({timeout}s) cho job {job_id}")

    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Job bị lỗi: {job.error}")

    path = job.output_dir / _ARTIFACT_FILENAME[artifact]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Chưa có file này: {path.name} (job chưa chạy xong?)")
    return FileResponse(path, filename=path.name)


# ── 5. Trọn gói Pipeline 1-Click ─────────────────────────────────────────────
@app.post(
    "/jobs/pipeline",
    response_class=FileResponse,
    tags=["Processing"],
    summary="Tạo job, upload 5 file, chạy inference và tải kết quả trực tiếp",
    description="Upload 5 file -> Khởi tạo job -> Chạy inference -> Trả về file result (.pkl) trực tiếp.",
    responses={
        400: {"description": "File không hợp lệ hoặc thiếu input."},
        500: {"description": "Inference thất bại."},
        504: {"description": "Quá thời gian chờ."},
    },
)
def run_pipeline_and_download(
    real_pt: UploadFile = File(..., description="File .pt pose real (HMR4D)"),
    mirror_pt: UploadFile = File(..., description="File .pt pose mirror (HMR4D)"),
    vitpose_real: UploadFile = File(..., description="File .pt ViTPose real"),
    vitpose_mirror: UploadFile = File(..., description="File .pt ViTPose mirror"),
    video: UploadFile = File(..., description="File video gốc"),
    bypass_refit: bool = Form(False, description="Bỏ qua pose refit"),
    timeout: float = Form(300.0, description="Thời gian chờ tối đa (giây)"),
) -> FileResponse:
    job = job_manager.create_job()

    _process_and_save_upload(job, UploadFileType.REAL_PT, real_pt)
    _process_and_save_upload(job, UploadFileType.MIRROR_PT, mirror_pt)
    _process_and_save_upload(job, UploadFileType.VITPOSE_REAL, vitpose_real)
    _process_and_save_upload(job, UploadFileType.VITPOSE_MIRROR, vitpose_mirror)
    _process_and_save_upload(job, UploadFileType.VIDEO, video)

    config = build_job_config(job)
    job_manager.submit_inference(job, config, bypass_refit=bypass_refit)

    try:
        job_manager.wait_for_job(job.job_id, timeout=timeout)
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f"Quá thời gian chờ ({timeout}s) khi xử lý inference")

    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Inference thất bại: {job.error}")

    pkl_path = job.output_dir / "inference_results.pkl"
    if not pkl_path.exists():
        raise HTTPException(status_code=500, detail="Không tìm thấy file kết quả inference_results.pkl")

    return FileResponse(pkl_path, filename=f"inference_results_{job.job_id}.pkl")

