"""Router for the Immobilien / Mietverwaltung module (profile: 'vermieter').

Stufe a: upload an HV statement → LLM extraction → editable positions → code
checks (BetrKV apportionability, formal, arithmetic) → Befund report.
"""

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
from local_ai.services.immo import extract_statement, run_checks

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED = {".pdf", ".docx", ".txt", ".md"}
_MAX_BYTES = 25 * 1024 * 1024


def has_vermieter(user: dict) -> bool:
    return "vermieter" in (user.get("profile") or "").lower()


def _require_vermieter(user: dict) -> None:
    if not has_vermieter(user):
        raise HTTPException(403, "Dieses Modul ist deinem Profil nicht zugeordnet.")


def _llm_cfg(settings) -> dict:
    return {
        "backend": settings.summary_backend,
        "openai_base_url": settings.openai_base_url,
        "openai_api_key": settings.openai_api_key,
        "openai_model": settings.openai_model,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
    }


@router.get("/immobilien")
async def immo_page(request: Request, user: dict = Depends(require_user)):
    _require_vermieter(user)
    db = request.app.state.db
    objects = await db.list_re_objects(user["id"])
    units = await db.list_re_units(user["id"])
    return request.app.state.templates.TemplateResponse(
        request, "immobilien.html",
        {"user": user, "objects": objects, "units": units},
    )


# ── Stammdaten ────────────────────────────────────────────────────────────

@router.post("/api/immo/objects")
async def add_object(request: Request, user: dict = Depends(require_user)):
    _require_vermieter(user)
    db = request.app.state.db
    b = await request.json()
    name = (b.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "Name erforderlich."}, status_code=400)
    obj = await db.add_re_object(
        user_id=user["id"], name=name,
        address=(b.get("address") or "").strip(),
        hausverwaltung=(b.get("hausverwaltung") or "").strip(),
        total_area=_f(b.get("total_area")),
    )
    return JSONResponse({"ok": True, "object": obj})


@router.delete("/api/immo/objects/{object_id}")
async def del_object(request: Request, object_id: str, user: dict = Depends(require_user)):
    _require_vermieter(user)
    ok = await request.app.state.db.delete_re_object(object_id, user["id"])
    return JSONResponse({"ok": ok})


@router.post("/api/immo/units")
async def add_unit(request: Request, user: dict = Depends(require_user)):
    _require_vermieter(user)
    db = request.app.state.db
    b = await request.json()
    object_id = b.get("object_id") or ""
    label = (b.get("label") or "").strip()
    obj = await db.get_re_object(object_id)
    if not obj or obj.get("user_id") != user["id"]:
        return JSONResponse({"error": "Objekt nicht gefunden."}, status_code=404)
    if not label:
        return JSONResponse({"error": "Bezeichnung erforderlich."}, status_code=400)
    unit = await db.add_re_unit(
        user_id=user["id"], object_id=object_id, label=label,
        area=_f(b.get("area")), mea=(b.get("mea") or "").strip(),
        persons=_i(b.get("persons")),
        tenant_name=(b.get("tenant_name") or "").strip(),
        tenant_prepayment=_f(b.get("tenant_prepayment")),
        umlage_key=(b.get("umlage_key") or "flaeche").strip(),
    )
    return JSONResponse({"ok": True, "unit": unit})


@router.delete("/api/immo/units/{unit_id}")
async def del_unit(request: Request, unit_id: str, user: dict = Depends(require_user)):
    _require_vermieter(user)
    ok = await request.app.state.db.delete_re_unit(unit_id, user["id"])
    return JSONResponse({"ok": ok})


# ── Stufe a: Abrechnungs-Check ────────────────────────────────────────────

@router.post("/api/immo/extract")
async def extract(request: Request, file: UploadFile, user: dict = Depends(require_user)):
    """Upload HV statement → LLM extracts positions (for the editable table)."""
    _require_vermieter(user)
    settings = request.app.state.settings

    ext = Path(file.filename or "x").suffix.lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"Format {ext} nicht unterstützt.")
    scratch = settings.upload_dir / "immo" / uuid.uuid4().hex[:12]
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / (file.filename or f"a{ext}")
    try:
        size = 0
        async with aiofiles.open(path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_BYTES:
                    raise HTTPException(400, "Datei zu groß (max 25 MB).")
                await f.write(chunk)
        try:
            text = await asyncio.to_thread(extract_text, path, ext)
        except Exception as exc:
            raise HTTPException(400, f"Textauszug fehlgeschlagen: {exc}") from exc
        if not text.strip():
            raise HTTPException(400, "Kein lesbarer Text (evtl. Scan ohne OCR).")
        try:
            extracted = await extract_statement(text, _llm_cfg(settings))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Immo extraction failed")
            raise HTTPException(502, f"Extraktion fehlgeschlagen: {type(exc).__name__}") from exc
        return JSONResponse({"ok": True, "extracted": extracted})
    finally:
        try:
            path.unlink(missing_ok=True)
            scratch.rmdir()
        except OSError:
            pass


@router.post("/api/immo/check")
async def check(request: Request, user: dict = Depends(require_user)):
    """Run code-based checks on the (possibly user-corrected) extracted data."""
    _require_vermieter(user)
    db = request.app.state.db
    b = await request.json()
    extracted = b.get("extracted") or {}
    if not extracted.get("positionen"):
        return JSONResponse({"error": "Keine Positionen zum Prüfen."}, status_code=400)

    unit = None
    if b.get("unit_id"):
        unit = await db.get_re_unit(b["unit_id"])
        if unit and unit.get("user_id") != user["id"]:
            unit = None
    report = run_checks(extracted, unit)
    return JSONResponse(report)


def _f(v):
    try:
        return float(str(v).replace(",", ".")) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
