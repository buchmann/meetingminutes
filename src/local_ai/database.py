import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    user_id         TEXT,
    visibility      TEXT NOT NULL DEFAULT 'private',
    filename        TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    duration_secs   REAL,

    language        TEXT NOT NULL DEFAULT 'auto',
    whisper_model   TEXT NOT NULL,
    diarization_on  INTEGER NOT NULL DEFAULT 1,
    summarization_on INTEGER NOT NULL DEFAULT 1,
    summary_detail  TEXT NOT NULL DEFAULT 'standard',
    min_speakers    INTEGER,
    max_speakers    INTEGER,

    status          TEXT NOT NULL DEFAULT 'pending',
    progress_pct    INTEGER NOT NULL DEFAULT 0,
    status_message  TEXT,
    error_message   TEXT,

    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    processing_secs REAL,

    transcript_json TEXT,
    summary_json    TEXT,
    detected_language TEXT,
    speaker_count   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    style_profile TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS email_accounts (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    provider      TEXT NOT NULL,          -- gmail | yahoo | tonline | custom
    email_address TEXT NOT NULL,
    imap_host     TEXT NOT NULL,
    imap_port     INTEGER NOT NULL DEFAULT 993,
    username      TEXT NOT NULL,          -- usually = email_address
    password_enc  TEXT NOT NULL,          -- Fernet-encrypted app password
    spam_folder   TEXT,                   -- provider spam/junk folder name
    enabled       INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_email_accounts_user ON email_accounts(user_id);

CREATE TABLE IF NOT EXISTS kb_documents (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    filename    TEXT NOT NULL,
    content     TEXT NOT NULL,          -- extracted plain text
    char_count  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kb_user ON kb_documents(user_id);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    title       TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB,                  -- float32 vector bytes (NULL if not embedded)
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_user ON kb_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks(doc_id);

-- Real estate / landlord module (Immobilien) ---------------------------
CREATE TABLE IF NOT EXISTS re_objects (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    name           TEXT NOT NULL,
    address        TEXT,
    hausverwaltung TEXT,
    total_area     REAL,                 -- Gesamtwohnfläche des Objekts (m²)
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_re_objects_user ON re_objects(user_id);

CREATE TABLE IF NOT EXISTS re_units (
    id                TEXT PRIMARY KEY,
    object_id         TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    label             TEXT NOT NULL,     -- z.B. "Whg 2 OG links"
    area              REAL,              -- Wohnfläche (m²)
    mea               TEXT,              -- Miteigentumsanteil, z.B. "85/1000"
    persons           INTEGER,
    tenant_name       TEXT,
    tenant_prepayment REAL,              -- NK-Vorauszahlung pro Monat (€)
    umlage_key        TEXT DEFAULT 'flaeche',  -- flaeche | personen | einheiten | verbrauch
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_re_units_user ON re_units(user_id);
CREATE INDEX IF NOT EXISTS idx_re_units_object ON re_units(object_id);

-- Projects: a workspace that groups a description + generated/uploaded docs ---
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);

CREATE TABLE IF NOT EXISTS project_docs (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    doc_type    TEXT NOT NULL DEFAULT 'document',  -- summary | product_spec | project_spec | note | upload
    source      TEXT NOT NULL DEFAULT '',          -- e.g. 'consolidator', 'upload', 'manual'
    fmt         TEXT NOT NULL DEFAULT 'md',         -- stored body is markdown/plain text
    content     TEXT NOT NULL DEFAULT '',           -- markdown / plain-text body
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_project_docs_project ON project_docs(project_id);
CREATE INDEX IF NOT EXISTS idx_project_docs_user ON project_docs(user_id);

-- Global app config (key/value), e.g. the active LLM choice -------------
CREATE TABLE IF NOT EXISTS app_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        await self._migrate()

    async def _migrate(self):
        """Add new columns to a pre-existing jobs table and prune legacy data."""
        cursor = await self._db.execute("PRAGMA table_info(jobs)")
        cols = {row["name"] for row in await cursor.fetchall()}
        if "user_id" not in cols:
            await self._db.execute("ALTER TABLE jobs ADD COLUMN user_id TEXT")
        if "visibility" not in cols:
            await self._db.execute(
                "ALTER TABLE jobs ADD COLUMN visibility TEXT NOT NULL DEFAULT 'private'"
            )
        if "summary_detail" not in cols:
            await self._db.execute(
                "ALTER TABLE jobs ADD COLUMN summary_detail TEXT NOT NULL DEFAULT 'standard'"
            )
        # users.profile gates role-specific PA modules (e.g. 'vermieter')
        ucur = await self._db.execute("PRAGMA table_info(users)")
        ucols = {row["name"] for row in await ucur.fetchall()}
        if "profile" not in ucols:
            await self._db.execute("ALTER TABLE users ADD COLUMN profile TEXT NOT NULL DEFAULT ''")
        await self._db.commit()
        # Indexes on the new columns can only be created after the columns exist
        # (so they live here rather than in SCHEMA, which runs before migration).
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_visibility ON jobs(visibility)")
        await self._db.commit()
        # One-time cleanup: legacy (pre-multi-user) jobs have no owner. Per the
        # multi-user migration decision these are removed for a clean start.
        await self._db.execute("DELETE FROM jobs WHERE user_id IS NULL")
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    # ── Jobs ─────────────────────────────────────────────────────────

    async def create_job(
        self,
        job_id: str,
        user_id: str,
        filename: str,
        file_path: str,
        file_size_bytes: int,
        language: str,
        whisper_model: str,
        diarization_on: bool,
        summarization_on: bool,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        summary_detail: str = "standard",
    ) -> dict:
        if summary_detail not in ("standard", "detailed"):
            summary_detail = "standard"
        await self._db.execute(
            """INSERT INTO jobs
               (id, user_id, visibility, filename, file_path, file_size_bytes,
                language, whisper_model, diarization_on, summarization_on,
                summary_detail, min_speakers, max_speakers, status, progress_pct, created_at)
               VALUES (?, ?, 'private', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)""",
            (
                job_id, user_id, filename, file_path, file_size_bytes,
                language, whisper_model, int(diarization_on), int(summarization_on),
                summary_detail, min_speakers, max_speakers, _now(),
            ),
        )
        await self._db.commit()
        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> dict | None:
        cursor = await self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def list_user_jobs(self, user_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def list_shared_jobs(self, exclude_user_id: str | None = None, limit: int = 100) -> list[dict]:
        """Jobs marked shared, with owner username joined for display."""
        sql = (
            "SELECT j.*, u.username AS owner_username FROM jobs j "
            "LEFT JOIN users u ON j.user_id = u.id "
            "WHERE j.visibility = 'shared'"
        )
        params: list = []
        if exclude_user_id is not None:
            sql += " AND j.user_id != ?"
            params.append(exclude_user_id)
        sql += " ORDER BY j.created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(sql, params)
        return [dict(r) for r in await cursor.fetchall()]

    async def update_job(self, job_id: str, **kwargs) -> None:
        if not kwargs:
            return
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(job_id)
        await self._db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", values)
        await self._db.commit()

    async def set_job_visibility(self, job_id: str, visibility: str) -> None:
        await self._db.execute(
            "UPDATE jobs SET visibility = ? WHERE id = ?", (visibility, job_id)
        )
        await self._db.commit()

    async def delete_job(self, job_id: str) -> bool:
        cursor = await self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def recover_stuck_jobs(self):
        terminal = ("completed", "failed", "pending")
        placeholders = ",".join("?" for _ in terminal)
        await self._db.execute(
            f"UPDATE jobs SET status = 'pending', progress_pct = 0, error_message = NULL "
            f"WHERE status NOT IN ({placeholders})",
            terminal,
        )
        await self._db.commit()

    async def all_job_ids(self) -> set[str]:
        cursor = await self._db.execute("SELECT id FROM jobs")
        return {row["id"] for row in await cursor.fetchall()}

    def parse_transcript(self, job: dict) -> dict | None:
        raw = job.get("transcript_json")
        return json.loads(raw) if raw else None

    def parse_summary(self, job: dict) -> dict | None:
        raw = job.get("summary_json")
        return json.loads(raw) if raw else None

    # ── Users ────────────────────────────────────────────────────────

    async def create_user(self, username: str, password_hash: str, is_admin: bool = False) -> dict:
        user_id = secrets.token_hex(8)
        await self._db.execute(
            "INSERT INTO users (id, username, password_hash, is_admin, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, int(is_admin), _now()),
        )
        await self._db.commit()
        return await self.get_user_by_id(user_id)

    async def get_user_by_id(self, user_id: str) -> dict | None:
        cursor = await self._db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_user_by_username(self, username: str) -> dict | None:
        cursor = await self._db.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_users(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT id, username, is_admin, created_at FROM users ORDER BY username"
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def count_users(self) -> int:
        cursor = await self._db.execute("SELECT COUNT(*) AS n FROM users")
        row = await cursor.fetchone()
        return row["n"]

    async def update_user_password(self, user_id: str, password_hash: str) -> None:
        await self._db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
        )
        await self._db.commit()

    async def delete_user(self, user_id: str) -> None:
        await self._db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        await self._db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await self._db.commit()

    async def get_user_style_profile(self, user_id: str | None) -> str | None:
        if not user_id:
            return None
        cursor = await self._db.execute(
            "SELECT style_profile FROM users WHERE id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return row["style_profile"] if row and row["style_profile"] else None

    async def set_user_style_profile(self, user_id: str, profile: str | None) -> None:
        await self._db.execute(
            "UPDATE users SET style_profile = ? WHERE id = ?", (profile, user_id)
        )
        await self._db.commit()

    # ── Email accounts ───────────────────────────────────────────────

    async def add_email_account(
        self, *, user_id: str, provider: str, email_address: str,
        imap_host: str, imap_port: int, username: str, password_enc: str,
        spam_folder: str | None,
    ) -> dict:
        account_id = secrets.token_hex(8)
        await self._db.execute(
            """INSERT INTO email_accounts
               (id, user_id, provider, email_address, imap_host, imap_port,
                username, password_enc, spam_folder, enabled, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (account_id, user_id, provider, email_address, imap_host, imap_port,
             username, password_enc, spam_folder, _now()),
        )
        await self._db.commit()
        return await self.get_email_account(account_id)

    async def get_email_account(self, account_id: str) -> dict | None:
        cur = await self._db.execute(
            "SELECT * FROM email_accounts WHERE id = ?", (account_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_email_accounts(self, user_id: str) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM email_accounts WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def delete_email_account(self, account_id: str, user_id: str) -> bool:
        cur = await self._db.execute(
            "DELETE FROM email_accounts WHERE id = ? AND user_id = ?",
            (account_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Knowledge base (Notes & Manuals) ─────────────────────────────

    async def add_kb_document(self, *, user_id: str, title: str, filename: str,
                              content: str) -> dict:
        doc_id = secrets.token_hex(8)
        await self._db.execute(
            """INSERT INTO kb_documents
               (id, user_id, title, filename, content, char_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, user_id, title, filename, content, len(content), _now()),
        )
        await self._db.commit()
        return {"id": doc_id, "title": title, "filename": filename,
                "char_count": len(content)}

    async def list_kb_documents(self, user_id: str) -> list[dict]:
        """Metadata only (no content) for the UI list."""
        cur = await self._db.execute(
            "SELECT id, title, filename, char_count, created_at "
            "FROM kb_documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_kb_contents(self, user_id: str) -> list[dict]:
        """Full content of all the user's docs (for search)."""
        cur = await self._db.execute(
            "SELECT id, title, content FROM kb_documents WHERE user_id = ?",
            (user_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def delete_kb_document(self, doc_id: str, user_id: str) -> bool:
        cur = await self._db.execute(
            "DELETE FROM kb_documents WHERE id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        await self._db.execute(
            "DELETE FROM kb_chunks WHERE doc_id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def add_kb_chunks(self, chunks: list[dict]) -> None:
        """Bulk-insert chunks. Each: {doc_id,user_id,chunk_index,title,text,embedding(bytes|None)}."""
        if not chunks:
            return
        await self._db.executemany(
            """INSERT INTO kb_chunks
               (id, doc_id, user_id, chunk_index, title, text, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(secrets.token_hex(8), c["doc_id"], c["user_id"], c["chunk_index"],
              c["title"], c["text"], c.get("embedding"), _now()) for c in chunks],
        )
        await self._db.commit()

    async def get_user_chunks(self, user_id: str, *, with_embedding: bool = True) -> list[dict]:
        cols = "id, doc_id, title, text" + (", embedding" if with_embedding else "")
        cur = await self._db.execute(
            f"SELECT {cols} FROM kb_chunks WHERE user_id = ?", (user_id,)
        )
        return [dict(r) for r in await cur.fetchall()]

    async def count_user_embedded_chunks(self, user_id: str) -> int:
        cur = await self._db.execute(
            "SELECT COUNT(*) FROM kb_chunks WHERE user_id = ? AND embedding IS NOT NULL",
            (user_id,),
        )
        row = await cur.fetchone()
        return row[0] if row else 0

    async def delete_user_chunks(self, user_id: str) -> None:
        await self._db.execute("DELETE FROM kb_chunks WHERE user_id = ?", (user_id,))
        await self._db.commit()

    # ── User profile (role-gated PA modules) ─────────────────────────

    async def set_user_profile(self, user_id: str, profile: str) -> None:
        await self._db.execute(
            "UPDATE users SET profile = ? WHERE id = ?", (profile or "", user_id)
        )
        await self._db.commit()

    # ── Real estate: objects + units ─────────────────────────────────

    async def add_re_object(self, *, user_id: str, name: str, address: str = "",
                            hausverwaltung: str = "", total_area: float | None = None) -> dict:
        oid = secrets.token_hex(8)
        await self._db.execute(
            """INSERT INTO re_objects (id, user_id, name, address, hausverwaltung, total_area, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (oid, user_id, name, address, hausverwaltung, total_area, _now()),
        )
        await self._db.commit()
        return await self.get_re_object(oid)

    async def get_re_object(self, object_id: str) -> dict | None:
        cur = await self._db.execute("SELECT * FROM re_objects WHERE id = ?", (object_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_re_objects(self, user_id: str) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM re_objects WHERE user_id = ? ORDER BY created_at", (user_id,)
        )
        return [dict(r) for r in await cur.fetchall()]

    async def delete_re_object(self, object_id: str, user_id: str) -> bool:
        cur = await self._db.execute(
            "DELETE FROM re_objects WHERE id = ? AND user_id = ?", (object_id, user_id)
        )
        await self._db.execute(
            "DELETE FROM re_units WHERE object_id = ? AND user_id = ?", (object_id, user_id)
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def add_re_unit(self, *, user_id: str, object_id: str, label: str,
                          area: float | None = None, mea: str = "", persons: int | None = None,
                          tenant_name: str = "", tenant_prepayment: float | None = None,
                          umlage_key: str = "flaeche") -> dict:
        uid = secrets.token_hex(8)
        await self._db.execute(
            """INSERT INTO re_units
               (id, object_id, user_id, label, area, mea, persons,
                tenant_name, tenant_prepayment, umlage_key, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (uid, object_id, user_id, label, area, mea, persons,
             tenant_name, tenant_prepayment, umlage_key, _now()),
        )
        await self._db.commit()
        return await self.get_re_unit(uid)

    async def get_re_unit(self, unit_id: str) -> dict | None:
        cur = await self._db.execute("SELECT * FROM re_units WHERE id = ?", (unit_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_re_units(self, user_id: str, object_id: str | None = None) -> list[dict]:
        if object_id:
            cur = await self._db.execute(
                "SELECT * FROM re_units WHERE user_id = ? AND object_id = ? ORDER BY created_at",
                (user_id, object_id),
            )
        else:
            cur = await self._db.execute(
                "SELECT * FROM re_units WHERE user_id = ? ORDER BY created_at", (user_id,)
            )
        return [dict(r) for r in await cur.fetchall()]

    async def delete_re_unit(self, unit_id: str, user_id: str) -> bool:
        cur = await self._db.execute(
            "DELETE FROM re_units WHERE id = ? AND user_id = ?", (unit_id, user_id)
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Projects ─────────────────────────────────────────────────────

    async def create_project(self, *, user_id: str, name: str, description: str = "") -> dict:
        pid = secrets.token_hex(8)
        now = _now()
        await self._db.execute(
            """INSERT INTO projects (id, user_id, name, description, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (pid, user_id, name, description, now, now),
        )
        await self._db.commit()
        return await self.get_project(pid)

    async def get_project(self, project_id: str) -> dict | None:
        cur = await self._db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_projects(self, user_id: str) -> list[dict]:
        """Projects for a user, newest first, each with its document count."""
        cur = await self._db.execute(
            """SELECT p.*, (SELECT COUNT(*) FROM project_docs d WHERE d.project_id = p.id) AS doc_count
               FROM projects p WHERE p.user_id = ? ORDER BY p.updated_at DESC""",
            (user_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def update_project(self, project_id: str, user_id: str, *,
                             name: str | None = None, description: str | None = None) -> bool:
        sets, vals = [], []
        if name is not None:
            sets.append("name = ?"); vals.append(name)
        if description is not None:
            sets.append("description = ?"); vals.append(description)
        if not sets:
            return False
        sets.append("updated_at = ?"); vals.append(_now())
        vals.extend([project_id, user_id])
        cur = await self._db.execute(
            f"UPDATE projects SET {', '.join(sets)} WHERE id = ? AND user_id = ?", vals
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_project(self, project_id: str, user_id: str) -> bool:
        cur = await self._db.execute(
            "DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)
        )
        await self._db.execute(
            "DELETE FROM project_docs WHERE project_id = ? AND user_id = ?", (project_id, user_id)
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def touch_project(self, project_id: str) -> None:
        await self._db.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (_now(), project_id)
        )
        await self._db.commit()

    async def add_project_doc(self, *, project_id: str, user_id: str, title: str,
                              content: str, doc_type: str = "document",
                              source: str = "", fmt: str = "md") -> dict:
        did = secrets.token_hex(8)
        await self._db.execute(
            """INSERT INTO project_docs
               (id, project_id, user_id, title, doc_type, source, fmt, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (did, project_id, user_id, title, doc_type, source, fmt, content, _now()),
        )
        await self._db.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (_now(), project_id)
        )
        await self._db.commit()
        return await self.get_project_doc(did)

    async def get_project_doc(self, doc_id: str) -> dict | None:
        cur = await self._db.execute("SELECT * FROM project_docs WHERE id = ?", (doc_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_project_docs(self, project_id: str, user_id: str, *, order: str = "desc") -> list[dict]:
        """Document metadata (no body) for a project's detail view.

        order='asc' → oldest first (chronological history); 'desc' → newest first.
        """
        direction = "ASC" if order == "asc" else "DESC"
        cur = await self._db.execute(
            f"""SELECT id, project_id, user_id, title, doc_type, source, fmt,
                      LENGTH(content) AS char_count, created_at
               FROM project_docs WHERE project_id = ? AND user_id = ? ORDER BY created_at {direction}""",
            (project_id, user_id),
        )
        return [dict(r) for r in await cur.fetchall()]

    @staticmethod
    def _snippet(content: str, query: str, width: int = 140) -> str:
        """Extract a ~width-char snippet of *content* around the first match of *query*."""
        if not content:
            return ""
        lc = content.lower()
        idx = lc.find(query.lower())
        if idx < 0:
            return content[:width].strip()
        start = max(0, idx - width // 3)
        end = min(len(content), idx + len(query) + (2 * width) // 3)
        snip = content[start:end].strip().replace("\n", " ")
        return ("…" if start > 0 else "") + snip + ("…" if end < len(content) else "")

    async def search_project_docs(self, project_id: str, user_id: str, query: str,
                                  limit: int = 30) -> list[dict]:
        """Full-text-ish search within one project's docs (title + content LIKE)."""
        q = (query or "").strip()
        if not q:
            return []
        like = f"%{q}%"
        cur = await self._db.execute(
            """SELECT id, title, doc_type, source, fmt, created_at, content
               FROM project_docs
               WHERE project_id = ? AND user_id = ? AND (title LIKE ? OR content LIKE ?)
               ORDER BY created_at DESC LIMIT ?""",
            (project_id, user_id, like, like, limit),
        )
        out = []
        for r in await cur.fetchall():
            d = dict(r)
            d["snippet"] = self._snippet(d.pop("content", ""), q)
            out.append(d)
        return out

    async def search_projects(self, user_id: str, query: str) -> list[dict]:
        """Search across all of a user's projects: match project name/description
        OR any of its docs (title/content). Returns projects (with doc_count) plus
        ``match_count`` = how many docs matched and ``name_match`` flag."""
        q = (query or "").strip()
        if not q:
            return []
        like = f"%{q}%"
        # Projects whose docs match, with per-project match counts
        cur = await self._db.execute(
            """SELECT project_id, COUNT(*) AS match_count
               FROM project_docs
               WHERE user_id = ? AND (title LIKE ? OR content LIKE ?)
               GROUP BY project_id""",
            (user_id, like, like),
        )
        doc_matches = {r["project_id"]: r["match_count"] for r in await cur.fetchall()}
        results = []
        for p in await self.list_projects(user_id):
            name_match = q.lower() in (p["name"] or "").lower() or q.lower() in (p["description"] or "").lower()
            mc = doc_matches.get(p["id"], 0)
            if name_match or mc:
                p = {**p, "match_count": mc, "name_match": name_match}
                results.append(p)
        return results

    async def delete_project_doc(self, doc_id: str, user_id: str) -> bool:
        cur = await self._db.execute(
            "DELETE FROM project_docs WHERE id = ? AND user_id = ?", (doc_id, user_id)
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── App config (key/value) ───────────────────────────────────────

    async def get_app_config(self, key: str, default: str | None = None) -> str | None:
        cur = await self._db.execute("SELECT value FROM app_config WHERE key = ?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else default

    async def set_app_config(self, key: str, value: str) -> None:
        await self._db.execute(
            "INSERT INTO app_config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await self._db.commit()

    # ── Sessions ─────────────────────────────────────────────────────

    async def create_session(self, user_id: str, ttl_hours: int) -> str:
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=ttl_hours)
        await self._db.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires.isoformat()),
        )
        await self._db.commit()
        return token

    async def get_user_by_session(self, token: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT u.*, s.expires_at AS _session_expires "
            "FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ?",
            (token,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        user = dict(row)
        try:
            if datetime.fromisoformat(user.pop("_session_expires")) < datetime.now(timezone.utc):
                await self.delete_session(token)
                return None
        except (ValueError, TypeError):
            return None
        return user

    async def delete_session(self, token: str) -> None:
        await self._db.execute("DELETE FROM sessions WHERE token = ?", (token,))
        await self._db.commit()
