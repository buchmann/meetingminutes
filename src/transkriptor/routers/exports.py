from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse

router = APIRouter(prefix="/api")


@router.get("/jobs/{job_id}/export/{fmt}")
async def export_job(request: Request, job_id: str, fmt: str):
    db = request.app.state.db
    job = await db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, "Job not yet completed")

    transcript = db.parse_transcript(job)
    summary = db.parse_summary(job)

    if fmt == "txt":
        from transkriptor.exporters.txt import export_txt
        content = export_txt(job, transcript, summary)
        return PlainTextResponse(
            content,
            headers={"Content-Disposition": f'attachment; filename="{job["filename"]}.txt"'},
        )
    elif fmt == "srt":
        from transkriptor.exporters.srt import export_srt
        content = export_srt(transcript)
        return PlainTextResponse(
            content,
            media_type="text/srt",
            headers={"Content-Disposition": f'attachment; filename="{job["filename"]}.srt"'},
        )
    elif fmt == "json":
        from transkriptor.exporters.json_export import export_json
        data = export_json(job, transcript, summary)
        return JSONResponse(
            data,
            headers={"Content-Disposition": f'attachment; filename="{job["filename"]}.json"'},
        )
    else:
        raise HTTPException(400, f"Unknown export format: {fmt}. Use txt, srt, or json.")
