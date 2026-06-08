"""Router for the Document Checker: upload Word/PDF, get an improved doc back.

Endpoints:
    GET  /documents                — render the upload page
    POST /api/documents/check      — extract → improve → return downloadable file
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
from local_ai.services.document_checker import (
    extract_text,
    generate_docx,
    generate_markdown,
    generate_pdf,
    improve_document_text,
)
from local_ai.services.summarizer import clamp_temperature

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_INPUT_EXTS = {".docx", ".pdf", ".txt", ".md"}
_ALLOWED_OUTPUTS = {"docx", "pdf", "txt", "md"}
_MAX_DOC_BYTES = 25 * 1024 * 1024  # 25 MB upload cap


@router.get("/documents")
async def documents_page(request: Request, user: dict = Depends(require_user)):
    """Render the Document Checker page."""
    db = request.app.state.db
    profile = await db.get_user_style_profile(user["id"])
    return request.app.state.templates.TemplateResponse(
        request,
        "documents.html",
        {"has_style_profile": profile is not None, "user": user},
    )


@router.post("/api/documents/check")
async def api_check_document(
    request: Request,
    file: UploadFile,
    output_format: str = Form(default="docx"),
    apply_style: str = Form(default="true"),
    temperature: str = Form(default=""),
    user: dict = Depends(require_user),
):
    """Upload a document, improve it through the LLM, return the result file.

    Form fields:
        file:           the input document (.docx / .pdf / .txt / .md)
        output_format:  "docx" | "pdf" | "txt"  (default: docx)
        apply_style:    "true" / "false" — apply user's writing-style profile
        temperature:    optional "hallucination" dial (0..1.5); blank = default
    """
    settings = request.app.state.settings
    db = request.app.state.db

    in_ext = Path(file.filename or "upload").suffix.lower()
    if in_ext not in _ALLOWED_INPUT_EXTS:
        raise HTTPException(400, f"Unsupported input format: {in_ext}. "
                                 f"Allowed: {sorted(_ALLOWED_INPUT_EXTS)}")

    out_fmt = output_format.lower().strip()
    if out_fmt not in _ALLOWED_OUTPUTS:
        raise HTTPException(400, f"Unsupported output format: {out_fmt}. "
                                 f"Allowed: {sorted(_ALLOWED_OUTPUTS)}")

    apply_style_flag = apply_style.lower() in ("true", "on", "1", "yes")
    temp = clamp_temperature(temperature)

    # Stream upload to a scratch path under the user's upload dir.
    scratch_id = uuid.uuid4().hex[:12]
    scratch_dir = settings.upload_dir / "documents" / scratch_id
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_path = scratch_dir / (file.filename or f"input{in_ext}")

    try:
        size = 0
        async with aiofiles.open(scratch_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_DOC_BYTES:
                    raise HTTPException(400, f"File exceeds {_MAX_DOC_BYTES // (1024*1024)}MB limit")
                await f.write(chunk)

        # Extract
        try:
            text = await asyncio.to_thread(extract_text, scratch_path, in_ext)
        except Exception as exc:
            logger.exception("Document extraction failed")
            raise HTTPException(400, f"Could not extract text from document: {exc}") from exc
        if not text.strip():
            raise HTTPException(400, "Document contains no readable text.")

        # Improve (optionally with the user's style profile)
        style_profile = await db.get_user_style_profile(user["id"]) if apply_style_flag else None
        result = await improve_document_text(
            text,
            style_profile=style_profile,
            backend=settings.summary_backend,
            openai_base_url=settings.openai_base_url,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            temperature=temp,
        )
        improved = result.get("improved") or ""
        if not improved:
            raise HTTPException(502, result.get("error") or "LLM returned no output.")

        # Generate output file
        base_name = Path(file.filename or "document").stem
        improved_title = f"{base_name} — improved"
        if out_fmt == "docx":
            data = generate_docx(improved, title=improved_title)
            media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            out_name = f"{base_name}-improved.docx"
        elif out_fmt == "pdf":
            data = generate_pdf(improved, title=improved_title)
            media = "application/pdf"
            out_name = f"{base_name}-improved.pdf"
        elif out_fmt == "md":
            data = generate_markdown(improved, title=improved_title)
            media = "text/markdown; charset=utf-8"
            out_name = f"{base_name}-improved.md"
        else:  # txt
            data = improved.encode("utf-8")
            media = "text/plain; charset=utf-8"
            out_name = f"{base_name}-improved.txt"

        headers = {
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Doc-Language": result.get("language", "en"),
            "X-Doc-Chunks": str(result.get("chunks", 1)),
            "X-Doc-Original-Chars": str(len(text)),
            "X-Doc-Improved-Chars": str(len(improved)),
            "X-Doc-Style-Applied": "1" if apply_style_flag and style_profile else "0",
        }
        if result.get("error"):
            headers["X-Doc-Warning"] = result["error"][:200]
        return Response(content=data, media_type=media, headers=headers)

    finally:
        # Clean up the scratch upload
        try:
            scratch_path.unlink(missing_ok=True)
            scratch_dir.rmdir()
        except OSError:
            pass
