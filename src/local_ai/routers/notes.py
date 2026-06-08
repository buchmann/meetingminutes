"""Router for Notes & Manuals: upload documents into a personal knowledge base,
then search them with an LLM-summarised, cited answer."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from local_ai.auth import require_user
from local_ai.services.document_checker import extract_text
from local_ai.services.notes_kb import (
    chunk_text,
    embed_texts,
    embedding_available,
    pack_vector,
    search_notes,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}
_MAX_BYTES = 25 * 1024 * 1024
_MAX_TEXT = 1_000_000  # cap stored text per doc (~1 MB)


def _llm_cfg(settings) -> dict:
    return {
        "backend": settings.summary_backend,
        "openai_base_url": settings.openai_base_url,
        "openai_api_key": settings.openai_api_key,
        "openai_model": settings.openai_model,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
    }


@router.get("/notes")
async def notes_page(request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    docs = await db.list_kb_documents(user["id"])
    return request.app.state.templates.TemplateResponse(
        request, "notes.html", {"user": user, "documents": docs},
    )


@router.get("/api/notes/list")
async def list_notes(request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    return JSONResponse({"documents": await db.list_kb_documents(user["id"])})


@router.post("/api/notes/upload")
async def upload_note(request: Request, file: UploadFile,
                      user: dict = Depends(require_user)):
    """Extract text from an uploaded doc and store it in the user's KB."""
    settings = request.app.state.settings
    db = request.app.state.db

    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported type {ext}. Allowed: {sorted(_ALLOWED_EXTS)}")

    scratch_dir = settings.upload_dir / "notes" / uuid.uuid4().hex[:12]
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_path = scratch_dir / (file.filename or f"doc{ext}")
    try:
        size = 0
        async with aiofiles.open(scratch_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_BYTES:
                    raise HTTPException(400, f"File exceeds {_MAX_BYTES // (1024*1024)}MB limit")
                await f.write(chunk)
        try:
            text = await asyncio.to_thread(extract_text, scratch_path, ext)
        except Exception as exc:
            logger.exception("Notes extract failed")
            raise HTTPException(400, f"Textauszug fehlgeschlagen: {exc}") from exc
        text = (text or "").strip()
        if not text:
            raise HTTPException(400, "Kein lesbarer Text im Dokument.")
        if len(text) > _MAX_TEXT:
            text = text[:_MAX_TEXT]

        title = Path(file.filename or "Dokument").stem
        saved = await db.add_kb_document(
            user_id=user["id"], title=title, filename=file.filename or f"dokument{ext}",
            content=text,
        )
        # Chunk + embed (if the embedding server is up) and store for semantic search.
        await _index_document(db, user["id"], saved["id"], title, text, settings)
        return JSONResponse({"ok": True, "document": saved})
    finally:
        try:
            scratch_path.unlink(missing_ok=True)
            scratch_dir.rmdir()
        except OSError:
            pass


@router.delete("/api/notes/{doc_id}")
async def delete_note(request: Request, doc_id: str, user: dict = Depends(require_user)):
    db = request.app.state.db
    ok = await db.delete_kb_document(doc_id, user["id"])
    return JSONResponse({"ok": ok})


async def _index_document(db, user_id: str, doc_id: str, title: str, text: str, settings) -> None:
    """Chunk a document, embed the chunks (if server up), and store them."""
    chunks = chunk_text(text)
    embs = await embed_texts(chunks, settings.embedding_base_url, settings.embedding_model)
    rows = []
    for i, ch in enumerate(chunks):
        emb = pack_vector(embs[i]) if embs and i < len(embs) else None
        rows.append({"doc_id": doc_id, "user_id": user_id, "chunk_index": i,
                     "title": title, "text": ch, "embedding": emb})
    await db.add_kb_chunks(rows)


async def _ensure_indexed(db, user_id: str, settings) -> None:
    """Auto-index docs that have no semantic embeddings yet (e.g. uploaded
    before the embedding server existed). No-op once embeddings are present or
    if the server is unreachable (keyword fallback stays in effect)."""
    if await db.count_user_embedded_chunks(user_id) > 0:
        return
    if not settings.embedding_base_url or not await embedding_available(settings.embedding_base_url):
        return
    docs = await db.get_kb_contents(user_id)
    if not docs:
        return
    await db.delete_user_chunks(user_id)   # rebuild cleanly
    for d in docs:
        await _index_document(db, user_id, d["id"], d["title"], d["content"], settings)


@router.post("/api/notes/search")
async def search_kb(request: Request, user: dict = Depends(require_user)):
    """Search the user's notes/manuals → cited, summarised answer (semantic if available)."""
    settings = request.app.state.settings
    db = request.app.state.db

    body = await request.json()
    query = (body.get("query") or "").strip()
    language = (body.get("language") or "auto").lower()
    if not query:
        return JSONResponse({"error": "Keine Frage angegeben."}, status_code=400)

    await _ensure_indexed(db, user["id"], settings)
    docs = await db.get_kb_contents(user["id"])            # keyword fallback source
    chunks = await db.get_user_chunks(user["id"], with_embedding=True)

    result = await search_notes(
        query, docs, llm=_llm_cfg(settings), language=language,
        embedded_chunks=chunks,
        embedding_base_url=settings.embedding_base_url,
        embedding_model=settings.embedding_model,
    )
    return JSONResponse(result)
