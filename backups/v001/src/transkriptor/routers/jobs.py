import asyncio
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from sse_starlette.sse import EventSourceResponse

from transkriptor.models import JobResponse

router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".webm", ".mp4"}


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    request: Request,
    file: UploadFile,
    language: str = Form(default="auto"),
    diarization_enabled: str = Form(default="true"),
    summarization_enabled: str = Form(default="true"),
    min_speakers: str = Form(default=""),
    max_speakers: str = Form(default=""),
):
    settings = request.app.state.settings
    db = request.app.state.db

    diarization_on = diarization_enabled.lower() in ("true", "on", "1", "yes")
    summarization_on = summarization_enabled.lower() in ("true", "on", "1", "yes")
    min_spk = int(min_speakers) if min_speakers.strip() else None
    max_spk = int(max_speakers) if max_speakers.strip() else None

    ext = Path(file.filename or "unknown").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    content = await file.read()
    size = len(content)
    if size > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {settings.max_upload_size_mb}MB limit")

    job_id = uuid.uuid4().hex[:12]
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / (file.filename or f"audio{ext}")

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    job = await db.create_job(
        job_id=job_id,
        filename=file.filename or "unknown",
        file_path=str(file_path),
        file_size_bytes=size,
        language=language,
        whisper_model=settings.whisper_model,
        diarization_on=diarization_on,
        summarization_on=summarization_on,
        min_speakers=min_spk,
        max_speakers=max_spk,
    )

    pipeline = request.app.state.pipeline
    asyncio.create_task(pipeline.process_job(job_id))

    return _job_to_response(job)


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(request: Request, limit: int = 50, offset: int = 0):
    db = request.app.state.db
    jobs = await db.list_jobs(limit=limit, offset=offset)
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(request: Request, job_id: str):
    db = request.app.state.db
    job = await db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _job_to_response(job)


@router.delete("/jobs/{job_id}")
async def delete_job(request: Request, job_id: str):
    db = request.app.state.db
    settings = request.app.state.settings

    job = await db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    await db.delete_job(job_id)

    upload_dir = settings.upload_dir / job_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    output_dir = settings.output_dir / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    return {"ok": True}


@router.get("/jobs/{job_id}/progress")
async def job_progress_sse(request: Request, job_id: str):
    db = request.app.state.db

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            job = await db.get_job(job_id)
            if job is None:
                yield {"event": "error", "data": json.dumps({"error": "Job not found"})}
                break
            yield {
                "event": "progress",
                "data": json.dumps({
                    "status": job["status"],
                    "progress_pct": job["progress_pct"],
                    "status_message": job["status_message"],
                }),
            }
            if job["status"] in ("completed", "failed"):
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "status": job["status"],
                        "error_message": job.get("error_message"),
                    }),
                }
                break
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@router.get("/health")
async def health_check(request: Request):
    settings = request.app.state.settings
    checks = {"whisper_model": settings.whisper_model, "ollama_model": settings.ollama_model}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            checks["ollama_reachable"] = True
            checks["ollama_model_available"] = any(settings.ollama_model in m for m in models)
    except Exception:
        checks["ollama_reachable"] = False
        checks["ollama_model_available"] = False
    checks["hf_token_set"] = bool(settings.hf_token)
    return checks


def _job_to_response(job: dict) -> JobResponse:
    return JobResponse(
        id=job["id"],
        filename=job["filename"],
        status=job["status"],
        progress_pct=job["progress_pct"],
        status_message=job.get("status_message"),
        error_message=job.get("error_message"),
        duration_secs=job.get("duration_secs"),
        detected_language=job.get("detected_language"),
        speaker_count=job.get("speaker_count"),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        processing_secs=job.get("processing_secs"),
    )
