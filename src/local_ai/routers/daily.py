"""Daily companion — Phase 1.

Pages / API:
    GET  /daily                 — the daily check-in page
    POST /api/daily/chat        — one conversational turn   {history:[{role,content}]}
    POST /api/daily/finalize    — propose structured updates {history:[...]}
    POST /api/daily/apply       — write the (confirmed) proposal {proposal:{...}}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from local_ai.auth import require_user
from local_ai.services import companion

logger = logging.getLogger(__name__)
router = APIRouter()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@router.get("/daily")
async def daily_page(request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    projects = await db.list_projects(user["id"])
    recent = await db.list_daily_logs(user["id"], limit=10)
    return request.app.state.templates.TemplateResponse(
        request, "daily.html",
        {"user": user, "projects": projects, "recent_logs": recent},
    )


@router.post("/api/daily/chat")
async def api_daily_chat(request: Request, user: dict = Depends(require_user)):
    body = await request.json()
    history = body.get("history") or []
    if not isinstance(history, list):
        raise HTTPException(400, "history must be a list")
    settings = request.app.state.settings
    db = request.app.state.db
    projects = await db.list_projects(user["id"])
    try:
        reply = await companion.chat(history, projects, settings)
    except Exception as exc:
        logger.exception("daily chat failed")
        raise HTTPException(502, f"LLM error: {exc}") from exc
    return {"reply": reply}


@router.post("/api/daily/finalize")
async def api_daily_finalize(request: Request, user: dict = Depends(require_user)):
    body = await request.json()
    history = body.get("history") or []
    if not [m for m in history if m.get("role") == "user"]:
        raise HTTPException(400, "Nothing to finalize yet — say something first.")
    settings = request.app.state.settings
    db = request.app.state.db
    projects = await db.list_projects(user["id"])
    try:
        proposal = await companion.finalize(history, projects, settings)
    except Exception as exc:
        logger.exception("daily finalize failed")
        raise HTTPException(502, f"LLM error: {exc}") from exc
    return {"proposal": proposal, "projects": [{"id": p["id"], "name": p["name"]} for p in projects]}


@router.post("/api/daily/apply")
async def api_daily_apply(request: Request, user: dict = Depends(require_user)):
    """Write the confirmed proposal: per-project logbook + todos, new projects,
    and the global day journal."""
    body = await request.json()
    proposal = body.get("proposal") or {}
    db = request.app.state.db
    today = _today()

    projects = await db.list_projects(user["id"])
    by_name = {p["name"].strip().lower(): p for p in projects}

    written = {"new_projects": [], "logbook": 0, "todos": 0, "journal": False, "projects_touched": []}

    for u in (proposal.get("updates") or []):
        if not isinstance(u, dict):
            continue
        target = None
        new_name = (u.get("new_project") or "").strip()
        proj_name = (u.get("project") or "").strip()
        if new_name and new_name.lower() not in by_name:
            target = await db.create_project(
                user_id=user["id"], name=new_name,
                description=(u.get("new_project_description") or "").strip(),
            )
            by_name[new_name.lower()] = target
            written["new_projects"].append(new_name)
        elif proj_name and proj_name.lower() in by_name:
            target = by_name[proj_name.lower()]
        elif new_name and new_name.lower() in by_name:
            target = by_name[new_name.lower()]
        if not target:
            continue

        touched = False
        logbook = (u.get("logbook") or "").strip()
        if logbook:
            await db.add_project_doc(
                project_id=target["id"], user_id=user["id"],
                title=f"Logbuch {today}", content=logbook,
                doc_type="logbook", section="logbook", source="daily", fmt="md",
            )
            written["logbook"] += 1
            touched = True
        for t in (u.get("todos") or []):
            t = str(t).strip()
            if not t:
                continue
            await db.add_project_doc(
                project_id=target["id"], user_id=user["id"],
                title=t[:80], content=t,
                doc_type="todo", section="todos", source="daily", fmt="md",
            )
            written["todos"] += 1
            touched = True
        if touched and target["name"] not in written["projects_touched"]:
            written["projects_touched"].append(target["name"])

    summary = (proposal.get("daily_summary") or "").strip()
    if summary:
        await db.add_daily_log(user_id=user["id"], day=today, content=summary)
        written["journal"] = True

    return {"ok": True, "written": written}
