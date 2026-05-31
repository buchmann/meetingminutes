import asyncio
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from sse_starlette.sse import EventSourceResponse

from transkriptor.auth import require_user
from transkriptor.models import JobResponse

router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".webm", ".mp4"}


async def _get_owned_job(db, job_id: str, user: dict) -> dict:
    """Fetch a job the user owns, else 404."""
    job = await db.get_job(job_id)
    if job is None or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    return job


async def _get_visible_job(db, job_id: str, user: dict) -> dict:
    """Fetch a job the user owns or that is shared, else 404."""
    job = await db.get_job(job_id)
    if job is None or (job["user_id"] != user["id"] and job["visibility"] != "shared"):
        raise HTTPException(404, "Job not found")
    return job


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    request: Request,
    file: UploadFile,
    language: str = Form(default="auto"),
    diarization_enabled: str = Form(default="true"),
    summarization_enabled: str = Form(default="true"),
    min_speakers: str = Form(default=""),
    max_speakers: str = Form(default=""),
    user: dict = Depends(require_user),
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

    # Stream upload to disk in chunks to avoid loading the whole file into RAM
    job_id = uuid.uuid4().hex[:12]
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / (file.filename or f"audio{ext}")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    size = 0
    async with aiofiles.open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            size += len(chunk)
            if size > max_bytes:
                await f.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(400, f"File exceeds {settings.max_upload_size_mb}MB limit")
            await f.write(chunk)

    job = await db.create_job(
        job_id=job_id,
        user_id=user["id"],
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
async def list_jobs(
    request: Request, limit: int = 50, offset: int = 0, user: dict = Depends(require_user)
):
    db = request.app.state.db
    jobs = await db.list_user_jobs(user["id"], limit=limit, offset=offset)
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/uploads")
async def list_uploads(request: Request, user: dict = Depends(require_user)):
    """List the current user's audio files already on the server (for reprocessing)."""
    db = request.app.state.db

    jobs = await db.list_user_jobs(user["id"], limit=200)
    uploads = []
    seen_files = set()
    for job in jobs:
        fp = Path(job["file_path"])
        if fp.exists() and job["filename"] not in seen_files:
            seen_files.add(job["filename"])
            uploads.append({
                "filename": job["filename"],
                "file_path": str(fp),
                "file_size_bytes": job["file_size_bytes"],
                "file_size_mb": round(job["file_size_bytes"] / (1024 * 1024), 1),
                "source_job_id": job["id"],
                "source_status": job["status"],
                "has_transcript": job.get("transcript_json") is not None,
            })
    return {"uploads": uploads}


@router.post("/jobs/reprocess")
async def reprocess_upload(
    request: Request,
    source_job_id: str = Form(...),
    language: str = Form(default="auto"),
    diarization_enabled: str = Form(default="true"),
    summarization_enabled: str = Form(default="true"),
    min_speakers: str = Form(default=""),
    max_speakers: str = Form(default=""),
    retranscribe: str = Form(default="false"),
    user: dict = Depends(require_user),
):
    """Create a new job (owned by the current user) from an already-uploaded file."""
    db = request.app.state.db
    settings = request.app.state.settings

    # Source must be the user's own job or a shared one.
    source_job = await _get_visible_job(db, source_job_id, user)

    source_path = Path(source_job["file_path"])
    if not source_path.exists():
        raise HTTPException(410, "Original audio file no longer exists on server")

    diarization_on = diarization_enabled.lower() in ("true", "on", "1", "yes")
    summarization_on = summarization_enabled.lower() in ("true", "on", "1", "yes")
    retranscribe_on = retranscribe.lower() in ("true", "on", "1", "yes")
    min_spk = int(min_speakers) if min_speakers.strip() else None
    max_spk = int(max_speakers) if max_speakers.strip() else None

    # Copy file to new job dir
    job_id = uuid.uuid4().hex[:12]
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / source_job["filename"]
    shutil.copy2(str(source_path), str(dest))

    job = await db.create_job(
        job_id=job_id,
        user_id=user["id"],
        filename=source_job["filename"],
        file_path=str(dest),
        file_size_bytes=source_job["file_size_bytes"],
        language=language,
        whisper_model=settings.whisper_model,
        diarization_on=diarization_on,
        summarization_on=summarization_on,
        min_speakers=min_spk,
        max_speakers=max_spk,
    )

    use_cached = not retranscribe_on and source_job.get("transcript_json") is not None
    if use_cached:
        await db.update_job(
            job_id,
            transcript_json=source_job["transcript_json"],
            detected_language=source_job.get("detected_language"),
            speaker_count=source_job.get("speaker_count"),
            duration_secs=source_job.get("duration_secs"),
        )

    pipeline = request.app.state.pipeline
    asyncio.create_task(pipeline.process_job(job_id))

    return _job_to_response(job)


@router.delete("/jobs/cleanup")
async def cleanup_jobs(request: Request, user: dict = Depends(require_user)):
    """Delete the current user's failed jobs and their files."""
    db = request.app.state.db
    settings = request.app.state.settings

    jobs = await db.list_user_jobs(user["id"], limit=1000)
    deleted = 0
    freed_bytes = 0
    for job in jobs:
        if job["status"] == "failed":
            upload_dir = settings.upload_dir / job["id"]
            if upload_dir.exists():
                for f in upload_dir.iterdir():
                    freed_bytes += f.stat().st_size
                shutil.rmtree(upload_dir)
            output_dir = settings.output_dir / job["id"]
            if output_dir.exists():
                shutil.rmtree(output_dir)
            await db.delete_job(job["id"])
            deleted += 1

    return {"ok": True, "deleted_jobs": deleted, "freed_mb": round(freed_bytes / (1024 * 1024), 1)}


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(request: Request, job_id: str, user: dict = Depends(require_user)):
    db = request.app.state.db
    job = await _get_visible_job(db, job_id, user)
    return _job_to_response(job)


@router.post("/jobs/{job_id}/share")
async def set_share(
    request: Request,
    job_id: str,
    user: dict = Depends(require_user),
):
    """Toggle a job between private and shared. Owner only."""
    db = request.app.state.db
    job = await _get_owned_job(db, job_id, user)
    body = await request.json()
    shared = bool(body.get("shared"))
    await db.set_job_visibility(job_id, "shared" if shared else "private")
    return {"ok": True, "job_id": job_id, "visibility": "shared" if shared else "private"}


@router.delete("/jobs/{job_id}")
async def delete_job(request: Request, job_id: str, user: dict = Depends(require_user)):
    db = request.app.state.db
    settings = request.app.state.settings

    await _get_owned_job(db, job_id, user)
    await db.delete_job(job_id)

    upload_dir = settings.upload_dir / job_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    output_dir = settings.output_dir / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    return {"ok": True}


@router.post("/jobs/{job_id}/retry")
async def retry_job(request: Request, job_id: str, user: dict = Depends(require_user)):
    """Re-run the full pipeline on a failed job (owner only, file must still exist)."""
    db = request.app.state.db

    job = await _get_owned_job(db, job_id, user)

    if job["status"] not in ("failed", "completed"):
        raise HTTPException(409, f"Job is currently {job['status']}, cannot retry")

    file_path = Path(job["file_path"])
    if not file_path.exists():
        raise HTTPException(410, "Original audio file no longer exists")

    await db.update_job(
        job_id, status="pending", progress_pct=0,
        status_message="Retrying...", error_message=None,
        transcript_json=None, summary_json=None,
        detected_language=None, speaker_count=None,
        completed_at=None, processing_secs=None,
    )

    pipeline = request.app.state.pipeline
    asyncio.create_task(pipeline.process_job(job_id))

    return {"ok": True, "job_id": job_id}


@router.post("/jobs/{job_id}/resummarize")
async def resummarize_job(request: Request, job_id: str, user: dict = Depends(require_user)):
    """Re-run summarization on a completed job (owner only)."""
    db = request.app.state.db

    job = await _get_owned_job(db, job_id, user)

    if not job.get("transcript_json"):
        raise HTTPException(400, "Job has no transcript to summarize")

    await db.update_job(job_id, status="summarizing", progress_pct=85,
                        status_message="Re-running summarization...")

    pipeline = request.app.state.pipeline
    asyncio.create_task(pipeline.resummarize_job(job_id))

    return {"ok": True, "job_id": job_id}


@router.get("/jobs/{job_id}/progress")
async def job_progress_sse(request: Request, job_id: str, user: dict = Depends(require_user)):
    db = request.app.state.db
    # Authorize once up front (owner or shared).
    await _get_visible_job(db, job_id, user)

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


# ── Recording endpoints ──────────────────────────────────────────────

@router.get("/recording/devices")
async def recording_devices(user: dict = Depends(require_user)):
    """List available audio input devices."""
    from transkriptor.services.recorder import list_audio_devices
    return {"devices": list_audio_devices()}


@router.get("/recording/status")
async def recording_status(request: Request, user: dict = Depends(require_user)):
    """Get current recording status."""
    recorder = request.app.state.recorder
    return recorder.status


@router.post("/recording/start")
async def recording_start(
    request: Request, device_index: int = 0, device_name: str = "", user: dict = Depends(require_user)
):
    """Start recording from the specified audio device."""
    recorder = request.app.state.recorder
    try:
        return recorder.start(device_index=device_index, device_name=device_name)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.post("/recording/stop")
async def recording_stop(
    request: Request,
    language: str = "auto",
    diarization_enabled: str = "true",
    summarization_enabled: str = "true",
    user: dict = Depends(require_user),
):
    """Stop recording and auto-submit to transcription pipeline (owned by current user)."""
    recorder = request.app.state.recorder
    settings = request.app.state.settings
    db = request.app.state.db

    try:
        result = recorder.stop()
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    file_path = Path(result["file_path"])
    filename = result["filename"]
    file_size = result["file_size_bytes"]

    diarization_on = diarization_enabled.lower() in ("true", "on", "1", "yes")
    summarization_on = summarization_enabled.lower() in ("true", "on", "1", "yes")

    job_id = uuid.uuid4().hex[:12]

    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / filename
    shutil.move(str(file_path), str(dest))

    job = await db.create_job(
        job_id=job_id,
        user_id=user["id"],
        filename=filename,
        file_path=str(dest),
        file_size_bytes=file_size,
        language=language,
        whisper_model=settings.whisper_model,
        diarization_on=diarization_on,
        summarization_on=summarization_on,
        min_speakers=None,
        max_speakers=None,
    )

    pipeline = request.app.state.pipeline
    asyncio.create_task(pipeline.process_job(job_id))

    return {**result, "job_id": job_id, "job_url": f"/jobs/{job_id}"}


# ── Style profile endpoints (per-user) ───────────────────────────────

@router.get("/style/profile")
async def get_style_profile(request: Request, user: dict = Depends(require_user)):
    """Get the current user's writing style profile."""
    db = request.app.state.db
    profile = await db.get_user_style_profile(user["id"])
    return {"profile": profile, "has_profile": profile is not None}


@router.post("/style/analyze")
async def analyze_style_from_text(request: Request, user: dict = Depends(require_user)):
    """Analyze pasted writing samples and generate the user's style profile."""
    settings = request.app.state.settings
    db = request.app.state.db
    body = await request.json()
    samples = body.get("samples", [])

    if not samples or not any(s.strip() for s in samples):
        raise HTTPException(400, "Provide at least one non-empty writing sample")

    samples = [s.strip() for s in samples if s.strip()]

    from transkriptor.services.style_analyzer import analyze_style
    profile = await analyze_style(
        samples,
        backend=settings.summary_backend,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )

    await db.set_user_style_profile(user["id"], profile)
    return {"profile": profile}


@router.post("/style/analyze-emails")
async def analyze_style_from_emails(request: Request, user: dict = Depends(require_user)):
    """Analyze sent emails to build the user's style profile (macOS Apple Mail)."""
    settings = request.app.state.settings
    db = request.app.state.db

    import subprocess
    script = '''
    tell application "Mail"
        set sentBox to mailbox "Sent" of account 1
        set msgs to (messages 1 through 10 of sentBox)
        set output to ""
        repeat with m in msgs
            set output to output & "---EMAIL_SEP---" & (content of m)
        end repeat
        return output
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Could not read emails: {result.stderr[:200]}")

        raw = result.stdout
        emails = [e.strip() for e in raw.split("---EMAIL_SEP---") if e.strip()]

        if len(emails) < 3:
            raise HTTPException(400, f"Only found {len(emails)} sent emails. Need at least 3.")

        samples = [e[:2000] for e in emails[:10]]

    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Timeout reading emails from Apple Mail")
    except FileNotFoundError:
        raise HTTPException(500, "osascript not found — this feature requires macOS")

    from transkriptor.services.style_analyzer import analyze_style
    profile = await analyze_style(
        samples,
        backend=settings.summary_backend,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )

    await db.set_user_style_profile(user["id"], profile)
    return {"profile": profile, "emails_analyzed": len(samples)}


@router.post("/style/save")
async def save_style(request: Request, user: dict = Depends(require_user)):
    """Manually save/edit the user's style profile."""
    db = request.app.state.db
    body = await request.json()
    profile = body.get("profile", "").strip()

    if not profile:
        raise HTTPException(400, "Profile text cannot be empty")

    await db.set_user_style_profile(user["id"], profile)
    return {"profile": profile}


@router.delete("/style/profile")
async def delete_style_profile(request: Request, user: dict = Depends(require_user)):
    """Remove the user's style profile (summaries will use default style)."""
    db = request.app.state.db
    await db.set_user_style_profile(user["id"], None)
    return {"ok": True}


@router.get("/gpu/metrics")
async def gpu_metrics(request: Request, user: dict = Depends(require_user)):
    """Fetch GPU metrics from DCGM exporter + gpu-manager on DGX Spark."""
    import httpx

    settings = request.app.state.settings
    spark_host = "192.168.178.190"
    if settings.gpu_manager_url:
        from urllib.parse import urlparse
        spark_host = urlparse(settings.gpu_manager_url).hostname or spark_host

    result = {
        "gpu_util": None,
        "mem_util": None,
        "temperature": None,
        "power_watts": None,
        "sm_clock_mhz": None,
        "memory_available_gb": None,
        "active_vllm_profile": None,
        "whisperx_running": None,
        "vllm_running": None,
        "coexistence": None,
        "error": None,
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"http://{spark_host}:9400/metrics")
            resp.raise_for_status()
            for line in resp.text.splitlines():
                if line.startswith("#") or "{" not in line:
                    continue
                name = line.split("{")[0]
                value_str = line.rsplit("}", 1)[-1].strip()
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                if name == "DCGM_FI_DEV_GPU_UTIL":
                    result["gpu_util"] = value
                elif name == "DCGM_FI_DEV_MEM_COPY_UTIL":
                    result["mem_util"] = value
                elif name == "DCGM_FI_DEV_GPU_TEMP":
                    result["temperature"] = value
                elif name == "DCGM_FI_DEV_POWER_USAGE":
                    result["power_watts"] = round(value, 1)
                elif name == "DCGM_FI_DEV_SM_CLOCK":
                    result["sm_clock_mhz"] = value
        except Exception as exc:
            result["error"] = f"DCGM: {exc}"

        if settings.gpu_manager_url:
            try:
                resp = await client.get(f"{settings.gpu_manager_url.rstrip('/')}/status")
                resp.raise_for_status()
                status = resp.json()
                result["memory_available_gb"] = status.get("memory_available_gb")
                result["active_vllm_profile"] = status.get("active_vllm_profile")
                result["whisperx_running"] = status.get("whisperx")
                result["vllm_running"] = status.get("vllm")
                result["coexistence"] = status.get("coexistence")
            except Exception as exc:
                if result["error"]:
                    result["error"] += f"; GPU manager: {exc}"
                else:
                    result["error"] = f"GPU manager: {exc}"

    return result


@router.get("/livez")
async def liveness():
    """Lightweight liveness probe — no auth, no external calls."""
    return {"status": "ok"}


@router.get("/readyz")
async def readiness(request: Request):
    """Readiness probe — no auth; checks DB is accessible."""
    db = request.app.state.db
    try:
        await db.list_jobs(limit=1)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/health")
async def health_check(request: Request):
    settings = request.app.state.settings
    checks = {
        "transcription_backend": settings.transcription_backend,
        "summary_backend": settings.summary_backend,
    }

    import httpx

    if settings.transcription_backend == "remote":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.whisperx_url.rstrip('/')}/health")
                data = resp.json()
                checks["whisperx_reachable"] = True
                checks["whisperx_gpu"] = data.get("gpu")
                checks["whisperx_device"] = data.get("device")
        except Exception:
            checks["whisperx_reachable"] = False
    else:
        checks["whisper_model"] = settings.whisper_model
        checks["hf_token_set"] = bool(settings.hf_token)

    if settings.summary_backend == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.ollama_base_url}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                checks["ollama_reachable"] = True
                checks["ollama_model_available"] = any(settings.ollama_model in m for m in models)
        except Exception:
            checks["ollama_reachable"] = False
            checks["ollama_model_available"] = False
    elif settings.summary_backend == "openai":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{settings.openai_base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                )
                checks["openai_reachable"] = True
                checks["openai_model"] = settings.openai_model
        except Exception:
            checks["openai_reachable"] = False

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
