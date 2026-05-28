from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/")
async def index(request: Request):
    db = request.app.state.db
    jobs = await db.list_jobs()
    settings = request.app.state.settings
    from transkriptor.services.style_analyzer import load_style_profile
    has_profile = load_style_profile(settings.style_profile_path) is not None
    return request.app.state.templates.TemplateResponse(
        request, "index.html", {"jobs": jobs, "has_style_profile": has_profile}
    )


@router.get("/partials/jobs", response_class=HTMLResponse)
async def partial_job_list(request: Request):
    db = request.app.state.db
    jobs = await db.list_jobs()
    return request.app.state.templates.TemplateResponse(
        request, "partials/job_list.html", {"jobs": jobs}
    )


@router.get("/settings")
async def settings_page(request: Request):
    settings = request.app.state.settings
    from transkriptor.services.style_analyzer import load_style_profile
    profile = load_style_profile(settings.style_profile_path)
    return request.app.state.templates.TemplateResponse(
        request, "settings.html", {"profile": profile}
    )


@router.get("/jobs/{job_id}")
async def job_detail(request: Request, job_id: str):
    db = request.app.state.db
    job = await db.get_job(job_id)
    if job is None:
        return request.app.state.templates.TemplateResponse(
            request, "index.html", {"jobs": [], "error": "Job not found"}, status_code=404
        )
    transcript = db.parse_transcript(job)
    summary = db.parse_summary(job)
    return request.app.state.templates.TemplateResponse(
        request, "job_detail.html",
        {"job": job, "transcript": transcript, "summary": summary},
    )
