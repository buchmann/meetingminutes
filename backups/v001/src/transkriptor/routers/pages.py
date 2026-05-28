from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/")
async def index(request: Request):
    db = request.app.state.db
    jobs = await db.list_jobs()
    return request.app.state.templates.TemplateResponse(
        "index.html", {"request": request, "jobs": jobs}
    )


@router.get("/partials/jobs", response_class=HTMLResponse)
async def partial_job_list(request: Request):
    db = request.app.state.db
    jobs = await db.list_jobs()
    return request.app.state.templates.TemplateResponse(
        "partials/job_list.html", {"request": request, "jobs": jobs}
    )


@router.get("/jobs/{job_id}")
async def job_detail(request: Request, job_id: str):
    db = request.app.state.db
    job = await db.get_job(job_id)
    if job is None:
        return request.app.state.templates.TemplateResponse(
            "index.html", {"request": request, "jobs": [], "error": "Job not found"}, status_code=404
        )
    transcript = db.parse_transcript(job)
    summary = db.parse_summary(job)
    return request.app.state.templates.TemplateResponse(
        "job_detail.html",
        {"request": request, "job": job, "transcript": transcript, "summary": summary},
    )
