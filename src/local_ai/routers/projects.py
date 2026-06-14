"""Router for Projects: a workspace that groups a description plus generated
or uploaded documents. The Consolidator can deposit its output here.

Pages:
    GET  /projects                       — list projects + create form
    GET  /projects/{id}                  — project detail (description + docs)

API:
    POST /api/projects                   — create project
    POST /api/projects/{id}              — update name/description
    POST /api/projects/{id}/delete       — delete project (+ its docs)
    POST /api/projects/{id}/docs         — add a doc (pasted text or file upload)
    GET  /api/projects/docs/{doc_id}/download?format=md|docx|pdf|txt
    POST /api/projects/docs/{doc_id}/delete
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response

from local_ai.auth import require_user
from local_ai.services.document_checker import (
    extract_text,
    generate_docx,
    generate_markdown,
    generate_pdf,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_DOC_EXTS = {".docx", ".pdf", ".txt", ".md"}
_MAX_DOC_BYTES = 25 * 1024 * 1024

DOC_TYPE_LABELS = {
    "summary": "Summary",
    "product_spec": "Product Spec",
    "project_spec": "Project Spec",
    "minutes": "Meeting Minutes",
    "note": "Note",
    "upload": "Upload",
    "document": "Document",
}


async def _owned_project(db, project_id: str, user: dict) -> dict:
    project = await db.get_project(project_id)
    if not project or project.get("user_id") != user["id"]:
        raise HTTPException(404, "Project not found.")
    return project


# ── Pages ─────────────────────────────────────────────────────────────


@router.get("/projects")
async def projects_page(request: Request, user: dict = Depends(require_user), q: str = ""):
    db = request.app.state.db
    q = (q or "").strip()
    if q:
        projects = await db.search_projects(user["id"], q)
    else:
        projects = await db.list_projects(user["id"])
    return request.app.state.templates.TemplateResponse(
        request, "projects.html",
        {"user": user, "projects": projects, "q": q},
    )


@router.get("/projects/{project_id}")
async def project_detail_page(project_id: str, request: Request, q: str = "",
                              user: dict = Depends(require_user)):
    db = request.app.state.db
    project = await _owned_project(db, project_id, user)
    # Chronological history: oldest first.
    docs = await db.list_project_docs(project_id, user["id"], order="asc")
    return request.app.state.templates.TemplateResponse(
        request, "project_detail.html",
        {"user": user, "project": project, "docs": docs,
         "doc_type_labels": DOC_TYPE_LABELS, "q": (q or "").strip()},
    )


@router.get("/api/projects/{project_id}/search")
async def api_search_project(project_id: str, request: Request, q: str = "",
                             user: dict = Depends(require_user)):
    """Search within one project's documents (title + content). Returns JSON."""
    db = request.app.state.db
    await _owned_project(db, project_id, user)
    results = await db.search_project_docs(project_id, user["id"], q)
    return {
        "query": (q or "").strip(),
        "count": len(results),
        "results": [
            {
                "id": r["id"], "title": r["title"], "doc_type": r["doc_type"],
                "source": r["source"], "created_at": r["created_at"], "snippet": r["snippet"],
            }
            for r in results
        ],
    }


# ── Project CRUD ──────────────────────────────────────────────────────


@router.post("/api/projects")
async def api_create_project(
    request: Request,
    name: str = Form(...),
    description: str = Form(default=""),
    user: dict = Depends(require_user),
):
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "A project name is required.")
    db = request.app.state.db
    project = await db.create_project(user_id=user["id"], name=name, description=description.strip())
    return RedirectResponse(url=f"/projects/{project['id']}", status_code=303)


@router.post("/api/projects/{project_id}")
async def api_update_project(
    project_id: str,
    request: Request,
    name: str = Form(default=None),
    description: str = Form(default=None),
    user: dict = Depends(require_user),
):
    db = request.app.state.db
    await _owned_project(db, project_id, user)
    await db.update_project(
        project_id, user["id"],
        name=(name.strip() if name is not None and name.strip() else None),
        description=(description if description is not None else None),
    )
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.post("/api/projects/{project_id}/delete")
async def api_delete_project(project_id: str, request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    await _owned_project(db, project_id, user)
    await db.delete_project(project_id, user["id"])
    return RedirectResponse(url="/projects", status_code=303)


# ── Documents ─────────────────────────────────────────────────────────


@router.post("/api/projects/{project_id}/docs")
async def api_add_doc(
    project_id: str,
    request: Request,
    title: str = Form(default=""),
    content: str = Form(default=""),
    files: list[UploadFile] = Form(default=[]),  # noqa: B008
    user: dict = Depends(require_user),
):
    """Add a document to a project — either pasted text or one/more uploaded files."""
    db = request.app.state.db
    await _owned_project(db, project_id, user)

    files = [f for f in (files or []) if f and getattr(f, "filename", None)]
    content = (content or "").strip()
    added = 0

    # Pasted text → one doc
    if content:
        await db.add_project_doc(
            project_id=project_id, user_id=user["id"],
            title=(title.strip() or "Note"), content=content,
            doc_type="note", source="manual", fmt="md",
        )
        added += 1

    # Uploaded files → one doc each (extracted to text)
    for upload in files:
        ext = Path(upload.filename or "upload").suffix.lower()
        if ext not in _ALLOWED_DOC_EXTS:
            raise HTTPException(400, f"Unsupported document type {ext} in {upload.filename}")
        raw = await upload.read()
        if len(raw) > _MAX_DOC_BYTES:
            raise HTTPException(400, f"{upload.filename} exceeds {_MAX_DOC_BYTES // (1024*1024)}MB limit")
        tmp = request.app.state.settings.upload_dir / f"_proj_{user['id']}_{upload.filename}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_bytes(raw)
            text = await asyncio.to_thread(extract_text, tmp, ext)
        except Exception as exc:
            logger.exception("Project doc extraction failed for %s", upload.filename)
            raise HTTPException(400, f"Could not extract {upload.filename}: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)
        if text.strip():
            await db.add_project_doc(
                project_id=project_id, user_id=user["id"],
                title=(upload.filename or "Upload"), content=text,
                doc_type="upload", source="upload", fmt="md",
            )
            added += 1

    if added == 0:
        raise HTTPException(400, "Nothing to add — paste text or choose a file.")
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@router.get("/api/projects/docs/{doc_id}/download")
async def api_download_doc(
    doc_id: str, request: Request,
    format: str = "md",
    user: dict = Depends(require_user),
):
    db = request.app.state.db
    doc = await db.get_project_doc(doc_id)
    if not doc or doc.get("user_id") != user["id"]:
        raise HTTPException(404, "Document not found.")

    fmt = (format or "md").lower()
    if fmt not in {"md", "docx", "pdf", "txt"}:
        raise HTTPException(400, "Invalid format.")

    title = doc["title"]
    body = doc["content"] or ""
    if fmt == "docx":
        data = generate_docx(body, title=title)
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ext = "docx"
    elif fmt == "pdf":
        data = generate_pdf(body, title=title)
        media = "application/pdf"
        ext = "pdf"
    elif fmt == "md":
        data = generate_markdown(body, title=title)
        media = "text/markdown; charset=utf-8"
        ext = "md"
    else:
        data = body.encode("utf-8")
        media = "text/plain; charset=utf-8"
        ext = "txt"

    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title).strip().replace(" ", "_") or "document"
    headers = {"Content-Disposition": f'attachment; filename="{safe}.{ext}"'}
    return Response(content=data, media_type=media, headers=headers)


@router.post("/api/projects/docs/{doc_id}/delete")
async def api_delete_doc(doc_id: str, request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    doc = await db.get_project_doc(doc_id)
    if not doc or doc.get("user_id") != user["id"]:
        raise HTTPException(404, "Document not found.")
    project_id = doc["project_id"]
    await db.delete_project_doc(doc_id, user["id"])
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)
