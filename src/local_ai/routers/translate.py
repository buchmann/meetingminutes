"""Router for the Translator: EN <-> DE for both pasted text and documents.

Endpoints:
    GET  /translate                  — render the translator page
    POST /api/translate/text         — JSON in/out, inline translation
    POST /api/translate/document     — file upload, returns downloadable file
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from local_ai.auth import require_user
from local_ai.services.document_checker import (
    extract_text,
    generate_docx,
    generate_markdown,
    generate_pdf,
)
from local_ai.services.summarizer import clamp_temperature
from local_ai.services.translator import LANG_NAMES, translate_text

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_DOC_EXTS = {".docx", ".pdf", ".txt", ".md"}
_ALLOWED_OUTPUTS = {"docx", "pdf", "txt", "md"}
_MAX_DOC_BYTES = 25 * 1024 * 1024


@router.get("/translate")
async def translate_page(request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    profile = await db.get_user_style_profile(user["id"])
    return request.app.state.templates.TemplateResponse(
        request,
        "translate.html",
        {
            "user": user,
            "has_style_profile": profile is not None,
            "lang_names": LANG_NAMES,
        },
    )


@router.post("/api/translate/text")
async def api_translate_text(request: Request, user: dict = Depends(require_user)):
    """Translate pasted text. Returns JSON with the result inline."""
    settings = request.app.state.settings
    db = request.app.state.db

    body = await request.json()
    text = (body.get("text") or "").strip()
    direction = (body.get("direction") or "auto").lower()
    apply_style = bool(body.get("apply_style", False))
    temperature = clamp_temperature(body.get("temperature"))

    if not text:
        return JSONResponse({"error": "No text provided."}, status_code=400)

    style_profile = await db.get_user_style_profile(user["id"]) if apply_style else None

    result = await translate_text(
        text,
        direction=direction,
        style_profile=style_profile,
        backend=settings.summary_backend,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        temperature=temperature,
    )
    return JSONResponse(result)


@router.post("/api/translate/document")
async def api_translate_document(
    request: Request,
    file: UploadFile,
    direction: str = Form(default="auto"),
    output_format: str = Form(default="docx"),
    apply_style: str = Form(default="false"),
    temperature: str = Form(default=""),
    user: dict = Depends(require_user),
):
    """Translate an uploaded document and return the translated file."""
    settings = request.app.state.settings
    db = request.app.state.db

    in_ext = Path(file.filename or "upload").suffix.lower()
    if in_ext not in _ALLOWED_DOC_EXTS:
        raise HTTPException(400, f"Unsupported input format: {in_ext}. "
                                 f"Allowed: {sorted(_ALLOWED_DOC_EXTS)}")
    out_fmt = (output_format or "docx").lower()
    if out_fmt not in _ALLOWED_OUTPUTS:
        raise HTTPException(400, f"Unsupported output format: {out_fmt}. "
                                 f"Allowed: {sorted(_ALLOWED_OUTPUTS)}")
    apply_style_flag = apply_style.lower() in ("true", "on", "1", "yes")
    temp = clamp_temperature(temperature)

    scratch_id = uuid.uuid4().hex[:12]
    scratch_dir = settings.upload_dir / "translate" / scratch_id
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

        try:
            text = await asyncio.to_thread(extract_text, scratch_path, in_ext)
        except Exception as exc:
            logger.exception("Document extraction failed")
            raise HTTPException(400, f"Could not extract text: {exc}") from exc
        if not text.strip():
            raise HTTPException(400, "Document contains no readable text.")

        style_profile = await db.get_user_style_profile(user["id"]) if apply_style_flag else None
        result = await translate_text(
            text,
            direction=direction,
            style_profile=style_profile,
            backend=settings.summary_backend,
            openai_base_url=settings.openai_base_url,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            temperature=temp,
        )
        translated = result.get("translated") or ""
        if not translated:
            raise HTTPException(502, result.get("error") or "LLM returned an empty translation.")

        base_name = Path(file.filename or "document").stem
        tgt = result.get("target_lang", "")
        title = f"{base_name} ({tgt})"

        if out_fmt == "docx":
            data = generate_docx(translated, title=title)
            media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ext = "docx"
        elif out_fmt == "pdf":
            data = generate_pdf(translated, title=title)
            media = "application/pdf"
            ext = "pdf"
        elif out_fmt == "md":
            data = generate_markdown(translated, title=title)
            media = "text/markdown; charset=utf-8"
            ext = "md"
        else:
            data = translated.encode("utf-8")
            media = "text/plain; charset=utf-8"
            ext = "txt"

        out_name = f"{base_name}-{tgt}.{ext}"
        headers = {
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Source-Lang": result.get("source_lang", ""),
            "X-Target-Lang": result.get("target_lang", ""),
            "X-Chunks": str(result.get("chunks", 1)),
            "X-Original-Chars": str(len(text)),
            "X-Translated-Chars": str(len(translated)),
            "X-Style-Applied": "1" if apply_style_flag and style_profile else "0",
        }
        if result.get("error"):
            headers["X-Warning"] = result["error"][:200]
        return Response(content=data, media_type=media, headers=headers)

    finally:
        try:
            scratch_path.unlink(missing_ok=True)
            scratch_dir.rmdir()
        except OSError:
            pass
