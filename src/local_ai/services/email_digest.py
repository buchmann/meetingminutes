"""Email digest: per-user IMAP fetch of the week's mail across providers,
then LLM classification into "important & for me" vs newsletter/promo/spam,
with explicit rescue of important mail that landed in the Spam folder.

Read-only: never marks messages seen, never moves or deletes anything.

Credentials (provider app-passwords) are stored Fernet-encrypted in the DB;
the key comes from settings.email_enc_key (LOCAL_AI_EMAIL_ENC_KEY secret).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from local_ai.services.summarizer import _call_openai_compatible, _call_ollama

logger = logging.getLogger(__name__)


# ── Provider presets ──────────────────────────────────────────────────────

PROVIDERS = {
    "gmail":   {"label": "Gmail",    "imap_host": "imap.gmail.com",          "imap_port": 993, "spam": "[Gmail]/Spam",
                "smtp_host": "smtp.gmail.com",         "smtp_port": 587},
    "yahoo":   {"label": "Yahoo",    "imap_host": "imap.mail.yahoo.com",     "imap_port": 993, "spam": "Bulk Mail",
                "smtp_host": "smtp.mail.yahoo.com",    "smtp_port": 587},
    "tonline": {"label": "T-Online", "imap_host": "secureimap.t-online.de",  "imap_port": 993, "spam": "Spam",
                "smtp_host": "securesmtp.t-online.de", "smtp_port": 587},
}

# Bound the work so a busy week can't blow up time/cost.
_MAX_PER_FOLDER = 60
_CLASSIFY_BATCH = 15


# ── Encryption ────────────────────────────────────────────────────────────

def _fernet(key: str):
    from cryptography.fernet import Fernet
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_password(plain: str, key: str) -> str:
    return _fernet(key).encrypt(plain.encode()).decode()


def decrypt_password(token: str, key: str) -> str:
    return _fernet(key).decrypt(token.encode()).decode()


# ── IMAP fetch (blocking — run via asyncio.to_thread) ─────────────────────

def _fetch_account_sync(account: dict, password: str, days: int) -> list[dict]:
    """Fetch recent messages from INBOX + the spam folder of one account.

    Returns a list of message dicts. Never marks anything seen.
    """
    from imap_tools import AND, MailBox

    since = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    out: list[dict] = []
    host = account["imap_host"]
    port = int(account.get("imap_port") or 993)
    user = account.get("username") or account["email_address"]
    spam_folder = account.get("spam_folder")
    acct_label = account["email_address"]
    acct_id = account.get("id", "")

    folders = [("INBOX", "inbox")]
    if spam_folder:
        folders.append((spam_folder, "spam"))

    with MailBox(host, port).login(user, password, initial_folder="INBOX") as mb:
        for folder_name, folder_kind in folders:
            try:
                if folder_name != "INBOX":
                    mb.folder.set(folder_name)
                for msg in mb.fetch(
                    AND(date_gte=since),
                    reverse=True,
                    limit=_MAX_PER_FOLDER,
                    mark_seen=False,        # READ-ONLY
                    bulk=True,
                    headers_only=True,      # skip body download → MUCH faster
                ):
                    # RFC bulk-mail marker: newsletters/marketing carry a
                    # List-Unsubscribe header; transactional 1:1 mail usually
                    # does not. Used to skip obvious bulk before the (slow) LLM.
                    try:
                        list_unsub = bool(msg.headers.get("list-unsubscribe"))
                    except Exception:  # noqa: BLE001
                        list_unsub = False
                    out.append({
                        "account_id": account.get("id", ""),
                        "account": acct_label,
                        "folder": folder_kind,            # "inbox" | "spam"
                        "uid": msg.uid or "",
                        "subject": (msg.subject or "(kein Betreff)")[:300],
                        "from_name": (msg.from_values.name if msg.from_values else "") or "",
                        "from_email": msg.from_ or "",
                        "to": ", ".join(msg.to or [])[:300],
                        "date": msg.date.isoformat() if msg.date else "",
                        "snippet": "",       # headers-only: triage on subject + sender
                        "list_unsub": list_unsub,
                    })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Folder %s on %s failed: %s", folder_name, acct_label, exc)
                continue
    return out


async def fetch_recent(account: dict, password: str, days: int = 7) -> list[dict]:
    return await asyncio.to_thread(_fetch_account_sync, account, password, days)


def _html_to_text(html: str) -> str:
    """Crude HTML→text for display (no body parser dependency)."""
    import html as _h
    import re
    if not html:
        return ""
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    t = re.sub(r"(?i)<br\s*/?>", "\n", t)
    t = re.sub(r"(?i)</p>", "\n\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = _h.unescape(t)
    # collapse excessive whitespace but keep paragraph breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    return t.strip()


def _fetch_body_sync(account: dict, password: str, folder_kind: str, uid: str) -> dict | None:
    """Fetch the full text body of ONE message by UID. Read-only."""
    from imap_tools import AND, MailBox

    host = account["imap_host"]
    port = int(account.get("imap_port") or 993)
    user = account.get("username") or account["email_address"]
    folder = "INBOX" if folder_kind == "inbox" else (account.get("spam_folder") or "INBOX")

    with MailBox(host, port).login(user, password, initial_folder=folder) as mb:
        for msg in mb.fetch(AND(uid=str(uid)), mark_seen=False, limit=1, bulk=False):
            body = (msg.text or "").strip() or _html_to_text(msg.html or "")
            attachments = []
            for i, att in enumerate(msg.attachments):
                attachments.append({
                    "index": i,
                    "filename": att.filename or f"anhang-{i}",
                    "content_type": att.content_type or "application/octet-stream",
                    "size": len(att.payload) if att.payload else (getattr(att, "size", 0) or 0),
                })
            try:
                msg_id = (msg.headers.get("message-id") or ("",))[0]
            except Exception:  # noqa: BLE001
                msg_id = ""
            reply_to = (msg.reply_to[0] if getattr(msg, "reply_to", None) else "") or msg.from_ or ""
            return {
                "subject": msg.subject or "(kein Betreff)",
                "from_name": (msg.from_values.name if msg.from_values else "") or "",
                "from_email": msg.from_ or "",
                "reply_to": reply_to,
                "message_id": msg_id,
                "to": ", ".join(msg.to or []),
                "date": msg.date.isoformat() if msg.date else "",
                "body": (body or "(kein Textinhalt)")[:20000],
                "attachments": attachments,
            }
    return None


async def fetch_message_body(account: dict, password: str, folder_kind: str, uid: str) -> dict | None:
    return await asyncio.to_thread(_fetch_body_sync, account, password, folder_kind, uid)


def _fetch_attachment_sync(account: dict, password: str, folder_kind: str, uid: str, idx: int) -> dict | None:
    """Return one attachment's bytes by index. Read-only."""
    from imap_tools import AND, MailBox

    host = account["imap_host"]
    port = int(account.get("imap_port") or 993)
    user = account.get("username") or account["email_address"]
    folder = "INBOX" if folder_kind == "inbox" else (account.get("spam_folder") or "INBOX")

    with MailBox(host, port).login(user, password, initial_folder=folder) as mb:
        for msg in mb.fetch(AND(uid=str(uid)), mark_seen=False, limit=1, bulk=False):
            atts = list(msg.attachments)
            if 0 <= idx < len(atts):
                a = atts[idx]
                return {
                    "filename": a.filename or f"anhang-{idx}",
                    "content_type": a.content_type or "application/octet-stream",
                    "payload": a.payload or b"",
                }
    return None


async def fetch_attachment(account: dict, password: str, folder_kind: str, uid: str, idx: int) -> dict | None:
    return await asyncio.to_thread(_fetch_attachment_sync, account, password, folder_kind, uid, idx)


# ── Sending (SMTP) ────────────────────────────────────────────────────────

def _send_reply_sync(account: dict, password: str, *, to: str, subject: str,
                     body: str, in_reply_to: str = "", references: str = "") -> None:
    """Send a plain-text reply via the provider's SMTP (STARTTLS on 587)."""
    import smtplib
    from email.message import EmailMessage

    preset = PROVIDERS.get(account.get("provider") or "", {})
    smtp_host = account.get("smtp_host") or preset.get("smtp_host")
    smtp_port = int(account.get("smtp_port") or preset.get("smtp_port") or 587)
    if not smtp_host:
        raise RuntimeError("Kein SMTP-Server für diesen Anbieter bekannt.")

    user = account.get("username") or account["email_address"]
    msg = EmailMessage()
    msg["From"] = account["email_address"]
    msg["To"] = to
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = references or in_reply_to
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(user, password)
        s.send_message(msg)


async def send_reply(account: dict, password: str, *, to: str, subject: str,
                     body: str, in_reply_to: str = "", references: str = "") -> None:
    await asyncio.to_thread(
        _send_reply_sync, account, password,
        to=to, subject=subject, body=body,
        in_reply_to=in_reply_to, references=references,
    )


# ── LLM classification ────────────────────────────────────────────────────

_CATEGORIES = ("important_personal", "important_transactional",
               "newsletter", "promotional", "spam", "other")


def _build_classify_prompt(batch: list[dict], my_addresses: list[str], language: str) -> str:
    items = []
    for i, m in enumerate(batch):
        items.append(
            f'#{i} | folder={m["folder"]} | from="{m["from_name"]}" <{m["from_email"]}> '
            f'| to={m["to"][:120]} | subject="{m["subject"]}" | snippet="{m["snippet"][:240]}"'
        )
    listing = "\n".join(items)
    mine = ", ".join(my_addresses) if my_addresses else "(unknown)"

    if language == "de":
        return (
            "Du bist ein E-Mail-Triage-Assistent. Stufe jede E-Mail unten ein. "
            f"Die Postfächer des Nutzers: {mine}.\n\n"
            "Gib NUR ein JSON-Objekt zurück: {\"results\":[{...}]} mit einem Eintrag pro E-Mail.\n"
            "Felder pro Eintrag:\n"
            '  "i": die #-Nummer der E-Mail\n'
            '  "category": eine von ["important_personal","important_transactional","newsletter","promotional","spam","other"]\n'
            '  "for_me": true wenn persönlich/direkt an den Nutzer gerichtet und relevant, false bei Massen-/Werbemail\n'
            '  "importance": 0-100 (wie dringend sollte der Nutzer das diese Woche sehen)\n'
            '  "misfiled": true NUR wenn folder=spam UND es klar eine persönliche Nachricht '
            'einer echten Person ODER eine echte Bank-/Rechnungs-/Behörden-/Sicherheitsmail an '
            'den Nutzer ist. NIEMALS true bei Werbung, Gewinnspiel, Krypto, Paket-Scam, '
            'unbekannten Absendern oder generischen Angeboten — im Zweifel false.\n'
            '  "reason": max. 12 Wörter, Deutsch\n'
            "Regeln: Newsletter/Werbung = niedrige importance, for_me=false. "
            "Rechnungen, Termine, persönliche Nachrichten, Behörden, Sicherheit = hoch. "
            "Erfinde nichts; nutze nur Absender/Betreff.\n\n"
            f"E-MAILS:\n{listing}\n\nNur das JSON-Objekt:"
        )
    return (
        "You are an email-triage assistant. Classify each email below. "
        f"The user's own mailboxes: {mine}.\n\n"
        "Return ONLY a JSON object: {\"results\":[{...}]} with one entry per email.\n"
        "Per-entry fields:\n"
        '  "i": the email #number\n'
        '  "category": one of ["important_personal","important_transactional","newsletter","promotional","spam","other"]\n'
        '  "for_me": true if personally/directly addressed and relevant, false for bulk/marketing\n'
        '  "importance": 0-100 (how urgently the user should see it this week)\n'
        '  "misfiled": true ONLY when folder=spam AND it is clearly a personal message from a '
        'real person OR a genuine bank/invoice/government/security mail addressed to the user. '
        'NEVER true for ads, lotteries, crypto, parcel-scams, unknown senders or generic offers '
        '— when in doubt, false.\n'
        '  "reason": max 12 words\n'
        "Rules: newsletters/ads = low importance, for_me=false. "
        "Invoices, appointments, personal messages, authorities, security = high. "
        "Invent nothing; use only sender/subject.\n\n"
        f"EMAILS:\n{listing}\n\nJSON object only:"
    )


async def _classify_batch(batch, my_addresses, language, llm) -> list[dict]:
    prompt = _build_classify_prompt(batch, my_addresses, language)
    try:
        if llm["backend"] == "openai":
            raw = await _call_openai_compatible(
                prompt, llm["openai_base_url"], llm["openai_api_key"], llm["openai_model"],
                response_format="json_object", temperature=0.0,
            )
        else:
            raw = await _call_ollama(prompt, llm["ollama_base_url"], llm["ollama_model"], 0.0)
        data = json.loads(raw)
        rows = data.get("results", data if isinstance(data, list) else [])
        by_i = {int(r["i"]): r for r in rows if "i" in r}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Email classification batch failed: %s", exc)
        by_i = {}

    enriched = []
    for i, m in enumerate(batch):
        r = by_i.get(i, {})
        enriched.append({
            **m,
            "category": r.get("category", "other"),
            "for_me": bool(r.get("for_me", False)),
            "importance": int(r.get("importance", 0) or 0),
            "misfiled": bool(r.get("misfiled", False)),
            "reason": (r.get("reason") or "")[:120],
        })
    return enriched


async def classify(messages: list[dict], my_addresses: list[str], language: str, llm: dict) -> list[dict]:
    # Run batches CONCURRENTLY (vLLM batches the requests) but cap parallelism
    # so a busy week with several accounts doesn't flood the model.
    batches = [messages[i:i + _CLASSIFY_BATCH] for i in range(0, len(messages), _CLASSIFY_BATCH)]
    sem = asyncio.Semaphore(6)

    async def _run(b):
        async with sem:
            return await _classify_batch(b, my_addresses, language, llm)

    results = await asyncio.gather(*[_run(b) for b in batches])
    out: list[dict] = []
    for r in results:
        out.extend(r)
    return out


# ── Orchestration ─────────────────────────────────────────────────────────

async def build_weekly_digest(
    accounts: list[dict],      # decrypted: each {account dict + "password"}
    *,
    days: int = 7,
    language: str = "de",
    llm: dict,
) -> dict:
    """Fetch + classify across all accounts; return a grouped digest.

    Returns ``{"important":[...], "rescued_from_spam":[...], "counts":{...},
    "errors":[...]}`` — each list sorted by importance desc then date desc.
    """
    my_addresses = [a["email_address"] for a in accounts]
    all_msgs: list[dict] = []
    errors: list[str] = []

    # Fetch all accounts concurrently.
    results = await asyncio.gather(
        *[fetch_recent(a, a["password"], days) for a in accounts],
        return_exceptions=True,
    )
    for a, res in zip(accounts, results):
        if isinstance(res, Exception):
            errors.append(f'{a["email_address"]}: {type(res).__name__}: {res}')
        else:
            all_msgs.extend(res)

    total_fetched = len(all_msgs)

    # Pre-filter (cheap, no LLM): mail carrying a List-Unsubscribe header is
    # bulk (newsletter/marketing) — in BOTH inbox and spam. It is never the
    # "important personal/transactional" mail we want, and marketing spam with
    # List-Unsubscribe is real spam, not a misfiled-important rescue candidate.
    # So skip all of it from the slow LLM pass. The remaining spam (no
    # List-Unsubscribe) are the genuine rescue candidates.
    SPAM_LLM_CAP = 25
    INBOX_LLM_CAP = 40
    to_classify, skipped_bulk, spam_seen, inbox_seen = [], 0, 0, 0
    for m in all_msgs:
        if m.get("list_unsub"):
            skipped_bulk += 1
            continue
        if m["folder"] == "spam":
            if spam_seen < SPAM_LLM_CAP:
                to_classify.append(m)
                spam_seen += 1
            continue
        if inbox_seen < INBOX_LLM_CAP:  # inbox, non-bulk → personal/transactional candidate
            to_classify.append(m)
            inbox_seen += 1

    classified = await classify(to_classify, my_addresses, language, llm) if to_classify else []

    def _sortkey(m):
        return (m.get("importance", 0), m.get("date", ""))

    important = sorted(
        [m for m in classified
         if m["folder"] == "inbox" and m["for_me"]
         and m["category"] in ("important_personal", "important_transactional")
         and m["importance"] >= 45],
        key=_sortkey, reverse=True,
    )
    rescued = sorted(
        [m for m in classified
         if m["folder"] == "spam" and m["misfiled"] and m["importance"] >= 55],
        key=_sortkey, reverse=True,
    )

    return {
        "important": important,
        "rescued_from_spam": rescued,
        "counts": {
            "fetched": total_fetched,
            "analysed": len(to_classify),
            "filtered_bulk": skipped_bulk,
            "important": len(important),
            "rescued": len(rescued),
            "accounts": len(accounts),
        },
        "errors": errors,
    }
