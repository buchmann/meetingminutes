from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from local_ai.auth import require_user

router = APIRouter(prefix="/api")


@router.get("/jobs/{job_id}/export/{fmt}")
async def export_job(request: Request, job_id: str, fmt: str, user: dict = Depends(require_user)):
    db = request.app.state.db
    job = await db.get_job(job_id)
    # Exportable if the user owns the job or it is shared.
    if job is None or (job["user_id"] != user["id"] and job["visibility"] != "shared"):
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, "Job not yet completed")

    transcript = db.parse_transcript(job)
    summary = db.parse_summary(job)

    if fmt == "txt":
        from local_ai.exporters.txt import export_txt
        content = export_txt(job, transcript, summary)
        return PlainTextResponse(
            content,
            headers={"Content-Disposition": f'attachment; filename="{job["filename"]}.txt"'},
        )
    elif fmt == "srt":
        from local_ai.exporters.srt import export_srt
        content = export_srt(transcript)
        return PlainTextResponse(
            content,
            media_type="text/srt",
            headers={"Content-Disposition": f'attachment; filename="{job["filename"]}.srt"'},
        )
    elif fmt == "json":
        from local_ai.exporters.json_export import export_json
        data = export_json(job, transcript, summary)
        return JSONResponse(
            data,
            headers={"Content-Disposition": f'attachment; filename="{job["filename"]}.json"'},
        )
    else:
        raise HTTPException(400, f"Unknown export format: {fmt}. Use txt, srt, or json.")
