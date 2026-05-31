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
    ) -> dict:
        await self._db.execute(
            """INSERT INTO jobs
               (id, user_id, visibility, filename, file_path, file_size_bytes,
                language, whisper_model, diarization_on, summarization_on,
                min_speakers, max_speakers, status, progress_pct, created_at)
               VALUES (?, ?, 'private', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)""",
            (
                job_id, user_id, filename, file_path, file_size_bytes,
                language, whisper_model, int(diarization_on), int(summarization_on),
                min_speakers, max_speakers, _now(),
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
