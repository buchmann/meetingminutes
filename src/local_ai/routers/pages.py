from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from local_ai.auth import (
    SESSION_COOKIE,
    get_current_user,
    hash_password,
    require_admin,
    require_user,
    verify_password,
)

router = APIRouter()


def _safe_next(next_url: str | None) -> str:
    """Only allow local redirects (must start with a single '/')."""
    if next_url and next_url.startswith("/") and not next_url.startswith("//"):
        return next_url
    return "/"


def _set_session_cookie(response, token: str, request: Request) -> None:
    settings = request.app.state.settings
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        max_age=settings.session_ttl_hours * 3600,
        path="/",
    )


# ── Authentication ───────────────────────────────────────────────────

@router.get("/login")
async def login_page(request: Request, next: str = "/"):
    user = await get_current_user(request)
    if user is not None:
        return RedirectResponse(url=_safe_next(next), status_code=303)
    return request.app.state.templates.TemplateResponse(
        request, "login.html", {"next": _safe_next(next), "error": None, "user": None}
    )


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form(default="/"),
):
    db = request.app.state.db
    settings = request.app.state.settings
    user = await db.get_user_by_username(username.strip())
    if user is None or not verify_password(password, user["password_hash"]):
        return request.app.state.templates.TemplateResponse(
            request,
            "login.html",
            {"next": _safe_next(next), "error": "Invalid username or password.", "user": None},
            status_code=401,
        )
    token = await db.create_session(user["id"], settings.session_ttl_hours)
    response = RedirectResponse(url=_safe_next(next), status_code=303)
    _set_session_cookie(response, token, request)
    return response


@router.post("/logout")
async def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        await request.app.state.db.delete_session(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


# ── Main pages ───────────────────────────────────────────────────────

@router.get("/")
async def index(request: Request, user: dict = Depends(require_user)):
    # Startpage is now Web Search; the meetings app lives at /meetings.
    return RedirectResponse(url="/search", status_code=307)


@router.get("/meetings")
async def meetings(request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    jobs = await db.list_user_jobs(user["id"])
    shared_jobs = await db.list_shared_jobs(exclude_user_id=user["id"])
    has_profile = bool(user.get("style_profile"))
    return request.app.state.templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": user,
            "jobs": jobs,
            "shared_jobs": shared_jobs,
            "has_style_profile": has_profile,
        },
    )


@router.get("/partials/jobs", response_class=HTMLResponse)
async def partial_job_list(request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    jobs = await db.list_user_jobs(user["id"])
    return request.app.state.templates.TemplateResponse(
        request,
        "partials/job_list.html",
        {"jobs": jobs, "user": user, "current_user_id": user["id"], "shared_view": False},
    )


@router.get("/partials/shared", response_class=HTMLResponse)
async def partial_shared_list(request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    jobs = await db.list_shared_jobs(exclude_user_id=user["id"])
    return request.app.state.templates.TemplateResponse(
        request,
        "partials/job_list.html",
        {"jobs": jobs, "user": user, "current_user_id": user["id"], "shared_view": True},
    )


@router.get("/system")
async def system_page(request: Request, user: dict = Depends(require_user)):
    """DGX Spark GPU / system utilization."""
    return request.app.state.templates.TemplateResponse(
        request, "system.html", {"user": user}
    )


@router.get("/settings")
async def settings_page(request: Request, user: dict = Depends(require_user)):
    from local_ai.config import LLM_MODELS
    profile = user.get("style_profile")
    settings = request.app.state.settings
    return request.app.state.templates.TemplateResponse(
        request, "settings.html",
        {
            "profile": profile, "user": user,
            "llm_models": LLM_MODELS,
            "active_llm": getattr(settings, "active_llm", "gptoss"),
        },
    )


@router.post("/api/settings/model")
async def switch_model(
    request: Request,
    model_key: str = Form(...),
    user: dict = Depends(require_admin),
):
    """Switch the active LLM (admin only). Persists the choice, repoints the
    app at the new model, and triggers the GPU-manager swap (≈3-5 min reload)."""
    from local_ai.config import LLM_MODELS, apply_llm
    import asyncio
    from local_ai.services.pipeline import _activate_gpu_service

    if model_key not in LLM_MODELS:
        raise HTTPException(400, f"Unknown model '{model_key}'.")

    db = request.app.state.db
    settings = request.app.state.settings
    await db.set_app_config("active_llm", model_key)
    model = apply_llm(settings, model_key)
    request.app.state.active_llm = model_key

    # Kick off the GPU swap in the background (the model takes minutes to load).
    if settings.gpu_manager_url:
        asyncio.create_task(_activate_gpu_service(settings.gpu_manager_url, model["gpu_endpoint"]))

    return {
        "ok": True,
        "model_key": model_key,
        "label": model["label"],
        "model": model["model"],
        "loading": bool(settings.gpu_manager_url),
    }


@router.post("/api/profile/role")
async def set_role(request: Request, user: dict = Depends(require_user)):
    """Set the user's role profile (gates role-specific PA modules)."""
    from fastapi.responses import JSONResponse
    db = request.app.state.db
    body = await request.json()
    roles = body.get("roles") or []
    allowed = {"vermieter"}
    value = ",".join(sorted(r for r in roles if r in allowed))
    await db.set_user_profile(user["id"], value)
    return JSONResponse({"ok": True, "profile": value})


@router.get("/jobs/{job_id}")
async def job_detail(request: Request, job_id: str, user: dict = Depends(require_user)):
    db = request.app.state.db
    job = await db.get_job(job_id)
    # Visible if the user owns it or it is shared.
    if job is None or (job["user_id"] != user["id"] and job["visibility"] != "shared"):
        return request.app.state.templates.TemplateResponse(
            request,
            "index.html",
            {"user": user, "jobs": [], "shared_jobs": [], "error": "Job not found"},
            status_code=404,
        )
    transcript = db.parse_transcript(job)
    summary = db.parse_summary(job)
    projects = await db.list_projects(user["id"])
    return request.app.state.templates.TemplateResponse(
        request,
        "job_detail.html",
        {
            "user": user,
            "job": job,
            "transcript": transcript,
            "summary": summary,
            "projects": projects,
            "is_owner": job["user_id"] == user["id"],
        },
    )


# ── Admin: user management ───────────────────────────────────────────

@router.get("/admin/users")
async def admin_users_page(request: Request, user: dict = Depends(require_admin)):
    users = await request.app.state.db.list_users()
    return request.app.state.templates.TemplateResponse(
        request, "admin_users.html", {"user": user, "users": users, "error": None, "notice": None}
    )


@router.post("/admin/users")
async def admin_create_user(
    request: Request,
    new_username: str = Form(...),
    new_password: str = Form(...),
    is_admin: str = Form(default=""),
    admin: dict = Depends(require_admin),
):
    db = request.app.state.db
    uname = new_username.strip()
    error = notice = None
    if not uname or not new_password:
        error = "Username and password are required."
    elif await db.get_user_by_username(uname) is not None:
        error = f"User '{uname}' already exists."
    else:
        await db.create_user(
            username=uname,
            password_hash=hash_password(new_password),
            is_admin=is_admin.lower() in ("true", "on", "1", "yes"),
        )
        notice = f"Created user '{uname}'."
    users = await db.list_users()
    return request.app.state.templates.TemplateResponse(
        request, "admin_users.html", {"user": admin, "users": users, "error": error, "notice": notice}
    )


@router.post("/admin/users/{user_id}/password")
async def admin_reset_password(
    request: Request,
    user_id: str,
    password: str = Form(...),
    admin: dict = Depends(require_admin),
):
    db = request.app.state.db
    target = await db.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(404, "User not found")
    await db.update_user_password(user_id, hash_password(password))
    users = await db.list_users()
    return request.app.state.templates.TemplateResponse(
        request,
        "admin_users.html",
        {"user": admin, "users": users, "error": None, "notice": f"Password reset for '{target['username']}'."},
    )


@router.post("/admin/users/{user_id}/delete")
async def admin_delete_user(
    request: Request, user_id: str, admin: dict = Depends(require_admin)
):
    db = request.app.state.db
    error = notice = None
    if user_id == admin["id"]:
        error = "You cannot delete your own account."
    else:
        target = await db.get_user_by_id(user_id)
        if target is None:
            raise HTTPException(404, "User not found")
        # Reassign / clean up: delete the user's jobs and files.
        settings = request.app.state.settings
        import shutil

        for job in await db.list_user_jobs(user_id, limit=10000):
            for base in (settings.upload_dir, settings.output_dir):
                d = base / job["id"]
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            await db.delete_job(job["id"])
        await db.delete_user(user_id)
        notice = f"Deleted user '{target['username']}' and their transcriptions."
    users = await db.list_users()
    return request.app.state.templates.TemplateResponse(
        request, "admin_users.html", {"user": admin, "users": users, "error": error, "notice": notice}
    )
