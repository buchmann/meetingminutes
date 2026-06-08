"""Router for the per-user Email Digest: configure IMAP accounts, then get a
weekly "important emails" digest with spam-rescue. Read-only."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from local_ai.auth import require_user
from local_ai.services.email_digest import (
    PROVIDERS,
    build_weekly_digest,
    decrypt_password,
    encrypt_password,
    fetch_attachment,
    fetch_message_body,
    fetch_recent,
    send_reply,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _llm_cfg(settings) -> dict:
    return {
        "backend": settings.summary_backend,
        "openai_base_url": settings.openai_base_url,
        "openai_api_key": settings.openai_api_key,
        "openai_model": settings.openai_model,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
    }


def _public_account(a: dict) -> dict:
    """Account dict without the encrypted password."""
    return {
        "id": a["id"],
        "provider": a["provider"],
        "email_address": a["email_address"],
        "imap_host": a["imap_host"],
        "created_at": a.get("created_at"),
    }


@router.get("/email")
async def email_page(request: Request, user: dict = Depends(require_user)):
    settings = request.app.state.settings
    db = request.app.state.db
    accounts = await db.list_email_accounts(user["id"])
    return request.app.state.templates.TemplateResponse(
        request,
        "email.html",
        {
            "user": user,
            "enabled": bool(settings.email_enc_key),
            "providers": PROVIDERS,
            "accounts": [_public_account(a) for a in accounts],
        },
    )


@router.get("/api/email/accounts")
async def list_accounts(request: Request, user: dict = Depends(require_user)):
    db = request.app.state.db
    accounts = await db.list_email_accounts(user["id"])
    return JSONResponse({"accounts": [_public_account(a) for a in accounts]})


@router.post("/api/email/accounts")
async def add_account(request: Request, user: dict = Depends(require_user)):
    """Add an IMAP account. Verifies the credentials before saving."""
    settings = request.app.state.settings
    db = request.app.state.db

    if not settings.email_enc_key:
        return JSONResponse({"error": "Email feature not configured (no encryption key)."}, status_code=503)

    body = await request.json()
    provider = (body.get("provider") or "").lower()
    email_address = (body.get("email_address") or "").strip()
    password = body.get("password") or ""
    if provider not in PROVIDERS:
        return JSONResponse({"error": f"Unknown provider: {provider}"}, status_code=400)
    if not email_address or not password:
        return JSONResponse({"error": "Email address and app password required."}, status_code=400)

    preset = PROVIDERS[provider]
    account = {
        "email_address": email_address,
        "imap_host": preset["imap_host"],
        "imap_port": preset["imap_port"],
        "username": email_address,
        "spam_folder": preset["spam"],
    }

    # Verify by actually fetching a tiny window (1 day). Fails fast on bad creds.
    try:
        await fetch_recent(account, password, days=1)
    except Exception as exc:  # noqa: BLE001
        logger.info("Email account verify failed for %s: %s", email_address, exc)
        return JSONResponse(
            {"error": f"Verbindung fehlgeschlagen: {type(exc).__name__}: {exc}. "
                      "App-Passwort und IMAP-Freischaltung prüfen."},
            status_code=400,
        )

    enc = encrypt_password(password, settings.email_enc_key)
    saved = await db.add_email_account(
        user_id=user["id"], provider=provider, email_address=email_address,
        imap_host=preset["imap_host"], imap_port=preset["imap_port"],
        username=email_address, password_enc=enc, spam_folder=preset["spam"],
    )
    return JSONResponse({"ok": True, "account": _public_account(saved)})


@router.delete("/api/email/accounts/{account_id}")
async def delete_account(request: Request, account_id: str, user: dict = Depends(require_user)):
    db = request.app.state.db
    ok = await db.delete_email_account(account_id, user["id"])
    return JSONResponse({"ok": ok})


@router.get("/api/email/digest")
async def get_digest(request: Request, days: int = 7, lang: str = "de",
                     user: dict = Depends(require_user)):
    """Build the weekly digest across the user's configured accounts."""
    settings = request.app.state.settings
    db = request.app.state.db

    if not settings.email_enc_key:
        return JSONResponse({"error": "Email feature not configured."}, status_code=503)

    rows = await db.list_email_accounts(user["id"])
    accounts = [a for a in rows if a.get("enabled", 1)]
    if not accounts:
        return JSONResponse({"error": "Keine E-Mail-Konten konfiguriert."}, status_code=400)

    # Decrypt passwords for use this request only.
    decrypted = []
    for a in accounts:
        try:
            pw = decrypt_password(a["password_enc"], settings.email_enc_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not decrypt account %s: %s", a["email_address"], exc)
            continue
        decrypted.append({**a, "password": pw})

    days = max(1, min(30, days))
    language = "de" if lang == "de" else "en"
    digest = await build_weekly_digest(
        decrypted, days=days, language=language, llm=_llm_cfg(settings),
    )
    return JSONResponse(digest)


@router.get("/api/email/message")
async def get_message_body(request: Request, account_id: str, folder: str = "inbox",
                           uid: str = "", user: dict = Depends(require_user)):
    """Fetch the full text body of one message on demand (read-only)."""
    settings = request.app.state.settings
    db = request.app.state.db

    if not settings.email_enc_key:
        return JSONResponse({"error": "Email feature not configured."}, status_code=503)
    if folder not in ("inbox", "spam") or not uid:
        return JSONResponse({"error": "Bad request."}, status_code=400)

    account = await db.get_email_account(account_id)
    if not account or account.get("user_id") != user["id"]:
        return JSONResponse({"error": "Konto nicht gefunden."}, status_code=404)

    try:
        pw = decrypt_password(account["password_enc"], settings.email_enc_key)
        msg = await fetch_message_body(account, pw, folder, uid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Body fetch failed (%s/%s): %s", account_id, uid, exc)
        return JSONResponse({"error": f"Abruf fehlgeschlagen: {type(exc).__name__}"}, status_code=502)

    if not msg:
        return JSONResponse({"error": "Nachricht nicht gefunden."}, status_code=404)
    return JSONResponse(msg)


@router.get("/api/email/attachment")
async def get_attachment(request: Request, account_id: str, folder: str = "inbox",
                         uid: str = "", idx: int = 0, user: dict = Depends(require_user)):
    """Download one attachment by index. Forced download (never inline)."""
    settings = request.app.state.settings
    db = request.app.state.db

    if not settings.email_enc_key:
        return JSONResponse({"error": "Email feature not configured."}, status_code=503)
    if folder not in ("inbox", "spam") or not uid:
        return JSONResponse({"error": "Bad request."}, status_code=400)

    account = await db.get_email_account(account_id)
    if not account or account.get("user_id") != user["id"]:
        return JSONResponse({"error": "Konto nicht gefunden."}, status_code=404)

    try:
        pw = decrypt_password(account["password_enc"], settings.email_enc_key)
        att = await fetch_attachment(account, pw, folder, uid, idx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Attachment fetch failed (%s/%s/%s): %s", account_id, uid, idx, exc)
        return JSONResponse({"error": f"Abruf fehlgeschlagen: {type(exc).__name__}"}, status_code=502)

    if not att:
        return JSONResponse({"error": "Anhang nicht gefunden."}, status_code=404)

    # ASCII-safe filename for the header + RFC5987 for the real (UTF-8) name.
    from urllib.parse import quote
    raw = att["filename"] or "anhang"
    ascii_name = raw.encode("ascii", "replace").decode("ascii").replace('"', "_")
    disp = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(raw)}"
    return Response(
        content=att["payload"],
        media_type=att["content_type"] or "application/octet-stream",
        headers={
            "Content-Disposition": disp,           # force download, never inline
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/api/email/reply")
async def send_email_reply(request: Request, user: dict = Depends(require_user)):
    """Send a plain-text reply from the user's account via SMTP.

    Body: {account_id, to, subject, body, in_reply_to?, references?}.
    Only ever sends on this explicit request — never automatically.
    """
    settings = request.app.state.settings
    db = request.app.state.db
    if not settings.email_enc_key:
        return JSONResponse({"error": "Email feature not configured."}, status_code=503)

    data = await request.json()
    account_id = data.get("account_id") or ""
    to = (data.get("to") or "").strip()
    subject = (data.get("subject") or "").strip() or "(kein Betreff)"
    body = data.get("body") or ""
    in_reply_to = data.get("in_reply_to") or ""
    references = data.get("references") or ""

    if not to or not body.strip():
        return JSONResponse({"error": "Empfänger und Text erforderlich."}, status_code=400)

    account = await db.get_email_account(account_id)
    if not account or account.get("user_id") != user["id"]:
        return JSONResponse({"error": "Konto nicht gefunden."}, status_code=404)

    try:
        pw = decrypt_password(account["password_enc"], settings.email_enc_key)
        await send_reply(account, pw, to=to, subject=subject, body=body,
                         in_reply_to=in_reply_to, references=references)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reply send failed (%s → %s): %s", account_id, to, exc)
        return JSONResponse(
            {"error": f"Senden fehlgeschlagen: {type(exc).__name__}: {exc}"},
            status_code=502,
        )
    return JSONResponse({"ok": True})
