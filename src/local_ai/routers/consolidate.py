"""Router for the Multi-Source Consolidator: combine meetings + documents
into a single Summary / Product Spec / Project Spec deliverable.

Endpoints:
    GET  /consolidate                 — render the consolidation page
    POST /api/consolidate             — run the consolidator + return the file
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from local_ai.auth import require_user
from local_ai.services.consolidator import (
    LABELS,
    OUTPUT_TYPES,
    Source,
    consolidate,
)
from local_ai.services.document_checker import (
    extract_text,
    generate_docx,
    generate_markdown,
    generate_pdf,
)
from local_ai.services.summarizer import clamp_temperature

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_DOC_EXTS = {".docx", ".pdf", ".txt", ".md"}
_ALLOWED_OUTPUTS = {"docx", "pdf", "txt", "md"}
_MAX_DOC_BYTES = 25 * 1024 * 1024


@router.get("/consolidate")
async def consolidate_page(request: Request, user: dict = Depends(require_user)):
    """Render the consolidator page with the user's available jobs as picker options."""
    db = request.app.state.db
    profile = await db.get_user_style_profile(user["id"])
    projects = await db.list_projects(user["id"])

    # Show only completed jobs that actually have a transcript (consolidation needs text)
    jobs = await db.list_user_jobs(user["id"], limit=200)
    pickable = [
        {
            "id": j["id"],
            "filename": j["filename"],
            "duration_secs": j.get("duration_secs"),
            "created_at": j.get("created_at"),
            "has_summary": bool(j.get("summary_json")),
        }
        for j in jobs
        if j.get("status") == "completed" and j.get("transcript_json")
    ]

    return request.app.state.templates.TemplateResponse(
        request,
        "consolidate.html",
        {
            "user": user,
            "has_style_profile": profile is not None,
            "jobs": pickable,
            "projects": projects,
            "output_types": OUTPUT_TYPES,
            "labels": LABELS,
        },
    )


@router.post("/api/consolidate")
async def api_consolidate(
    request: Request,
    output_type: str = Form(default="summary"),
    output_format: str = Form(default="docx"),
    language: str = Form(default="en"),
    apply_style: str = Form(default="true"),
    temperature: str = Form(default=""),
    job_ids: list[str] = Form(default=[]),  # noqa: B008
    use_summaries: str = Form(default="true"),
    project_id: str = Form(default=""),
    files: list[UploadFile] = Form(default=[]),  # noqa: B008
    user: dict = Depends(require_user),
):
    """Build a consolidated document from the selected jobs + uploaded files.

    Form fields:
        output_type:     "summary" | "product_spec" | "project_spec"
        output_format:   "docx" | "pdf" | "md" | "txt"
        language:        "en" | "de"
        apply_style:     "true" / "false"
        temperature:     optional "hallucination" dial (0..1.5); blank = default
        job_ids:         repeating field — IDs of jobs to include
        use_summaries:   "true" => use the job's summary JSON if available
                         (cheaper); "false" => always use the full transcript
        files:           repeating field — additional uploaded documents
    """
    settings = request.app.state.settings
    db = request.app.state.db

    output_type = (output_type or "summary").lower()
    if output_type not in OUTPUT_TYPES:
        raise HTTPException(400, f"Invalid output_type: {output_type}. Allowed: {sorted(OUTPUT_TYPES)}")
    out_fmt = (output_format or "docx").lower()
    if out_fmt not in _ALLOWED_OUTPUTS:
        raise HTTPException(400, f"Invalid output_format: {out_fmt}. Allowed: {sorted(_ALLOWED_OUTPUTS)}")
    language = "de" if language == "de" else "en"
    apply_style_flag = apply_style.lower() in ("true", "on", "1", "yes")
    use_summaries_flag = use_summaries.lower() in ("true", "on", "1", "yes")
    temp = clamp_temperature(temperature)

    # Optional target project: validate ownership up front
    project_id = (project_id or "").strip()
    target_project = None
    if project_id:
        target_project = await db.get_project(project_id)
        if not target_project or target_project.get("user_id") != user["id"]:
            raise HTTPException(404, "Selected project not found.")

    # Filter and clean optional fields
    job_ids = [j for j in (job_ids or []) if j and j.strip()]
    files = [f for f in (files or []) if f and getattr(f, "filename", None)]

    if not job_ids and not files:
        raise HTTPException(400, "Select at least one job or upload at least one document.")

    # Pull the selected jobs (only those owned by the user)
    sources: list[Source] = []
    for job_id in job_ids:
        job = await db.get_job(job_id)
        if not job or job.get("user_id") != user["id"]:
            logger.warning("User %s tried to consolidate inaccessible job %s", user["id"], job_id)
            continue
        if not job.get("transcript_json"):
            continue

        meta = []
        if job.get("duration_secs"):
            d = int(job["duration_secs"])
            meta.append(f"Duration: {d // 60}m {d % 60}s")
        if job.get("detected_language"):
            meta.append(f"Language: {job['detected_language']}")
        if job.get("speaker_count"):
            meta.append(f"Speakers: {job['speaker_count']}")
        if job.get("created_at"):
            meta.append(f"Created: {job['created_at']}")

        summary = db.parse_summary(job) if use_summaries_flag else None
        if summary:
            # Render the summary structure as readable prose instead of raw JSON
            body = _render_summary(summary)
            sources.append(Source(kind="summary", title=job["filename"], body=body, meta=meta))
        else:
            transcript = db.parse_transcript(job)
            body = _render_transcript(transcript) if transcript else ""
            sources.append(Source(kind="transcript", title=job["filename"], body=body, meta=meta))

    # Pull the uploaded files
    scratch_id = uuid.uuid4().hex[:12]
    scratch_dir = settings.upload_dir / "consolidate" / scratch_id
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_paths: list[Path] = []
    try:
        for upload in files:
            ext = Path(upload.filename or "upload").suffix.lower()
            if ext not in _ALLOWED_DOC_EXTS:
                raise HTTPException(400, f"Unsupported document type {ext} in {upload.filename}")
            scratch_path = scratch_dir / (upload.filename or f"input{ext}")
            size = 0
            async with aiofiles.open(scratch_path, "wb") as fh:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > _MAX_DOC_BYTES:
                        raise HTTPException(400, f"{upload.filename} exceeds {_MAX_DOC_BYTES // (1024*1024)}MB limit")
                    await fh.write(chunk)
            scratch_paths.append(scratch_path)
            try:
                text = await asyncio.to_thread(extract_text, scratch_path, ext)
            except Exception as exc:
                logger.exception("Document extraction failed for %s", upload.filename)
                raise HTTPException(400, f"Could not extract {upload.filename}: {exc}") from exc
            if text.strip():
                sources.append(Source(
                    kind="document",
                    title=upload.filename or scratch_path.name,
                    body=text,
                    meta=[f"Type: {ext.lstrip('.').upper()}", f"Size: {size} bytes"],
                ))

        if not sources:
            raise HTTPException(400, "All selected sources were empty.")

        # Style profile (optional)
        style_profile = await db.get_user_style_profile(user["id"]) if apply_style_flag else None

        # Run the consolidation
        result = await consolidate(
            sources,
            output_type=output_type,
            language=language,
            style_profile=style_profile,
            backend=settings.summary_backend,
            openai_base_url=settings.openai_base_url,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            detail_level="detailed",
            temperature=temp,
        )

        if result.get("error"):
            raise HTTPException(502, result["error"])
        markdown = result.get("markdown") or ""
        if not markdown:
            raise HTTPException(502, "LLM returned an empty document.")

        # Title for the generated file: e.g. "Project_Spec_2026-06-02"
        title = LABELS["de" if language == "de" else "en"][output_type]

        # Optionally file the result into a project (markdown body kept verbatim)
        saved_project_name = ""
        if target_project:
            await db.add_project_doc(
                project_id=target_project["id"], user_id=user["id"],
                title=title, content=markdown,
                doc_type=output_type, source="consolidator", fmt="md",
            )
            saved_project_name = target_project["name"]

        if out_fmt == "docx":
            data = generate_docx(markdown, title=title)
            media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ext = "docx"
        elif out_fmt == "pdf":
            data = generate_pdf(markdown, title=title)
            media = "application/pdf"
            ext = "pdf"
        elif out_fmt == "md":
            data = generate_markdown(markdown, title=title)
            media = "text/markdown; charset=utf-8"
            ext = "md"
        else:  # txt
            data = markdown.encode("utf-8")
            media = "text/plain; charset=utf-8"
            ext = "txt"

        safe_title = title.replace(" ", "_")
        out_name = f"{safe_title}.{ext}"
        headers = {
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Sources-Count": str(result["sources_count"]),
            "X-Truncated": "1" if result.get("truncated") else "0",
            "X-Style-Applied": "1" if apply_style_flag and style_profile else "0",
            "X-Output-Type": output_type,
            "X-Language": language,
            "X-Project-Saved": "1" if saved_project_name else "0",
        }
        return Response(content=data, media_type=media, headers=headers)

    finally:
        for p in scratch_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            scratch_dir.rmdir()
        except OSError:
            pass


# ── Helpers for turning stored JSON back into readable prose ──────────────


def _render_summary(summary: dict) -> str:
    """Flatten a summary JSON into a Markdown-ish prose block."""
    out: list[str] = []
    if summary.get("overall_summary"):
        out.append("### Overall summary\n" + summary["overall_summary"])
    if summary.get("participants"):
        out.append("### Participants\n- " + "\n- ".join(summary["participants"]))
    if summary.get("key_topics"):
        out.append("### Key topics")
        for t in summary["key_topics"]:
            if isinstance(t, dict):
                name = t.get("name", "Topic")
                body = t.get("summary", "")
                out.append(f"**{name}**\n{body}")
                if t.get("sub_points"):
                    for sp in t["sub_points"]:
                        if isinstance(sp, dict):
                            out.append(f"  - {sp.get('text','')} — {sp.get('detail','')}")
            else:
                out.append(f"- {t}")
    if summary.get("key_decisions"):
        out.append("### Key decisions\n- " + "\n- ".join(map(str, summary["key_decisions"])))
    if summary.get("action_items"):
        out.append("### Action items")
        for a in summary["action_items"]:
            if isinstance(a, dict):
                out.append(
                    f"- {a.get('description','')} "
                    f"(assignee: {a.get('assignee','—')}, deadline: {a.get('deadline','—')})"
                )
            else:
                out.append(f"- {a}")
    if summary.get("next_steps"):
        out.append("### Next steps\n- " + "\n- ".join(map(str, summary["next_steps"])))
    if summary.get("open_questions"):
        out.append("### Open questions\n- " + "\n- ".join(map(str, summary["open_questions"])))
    return "\n\n".join(out)


def _render_transcript(transcript: dict) -> str:
    """Flatten a transcript JSON into a speaker-labeled prose block."""
    if not transcript:
        return ""
    segments = transcript.get("segments") or []
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)
