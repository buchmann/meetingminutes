import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
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
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def create_job(
        self,
        job_id: str,
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
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO jobs
               (id, filename, file_path, file_size_bytes, language, whisper_model,
                diarization_on, summarization_on, min_speakers, max_speakers,
                status, progress_pct, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)""",
            (
                job_id, filename, file_path, file_size_bytes, language, whisper_model,
                int(diarization_on), int(summarization_on), min_speakers, max_speakers,
                now,
            ),
        )
        await self._db.commit()
        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> dict | None:
        cursor = await self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_job(self, job_id: str, **kwargs) -> None:
        if not kwargs:
            return
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(job_id)
        await self._db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", values)
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

    def parse_transcript(self, job: dict) -> dict | None:
        raw = job.get("transcript_json")
        if raw:
            return json.loads(raw)
        return None

    def parse_summary(self, job: dict) -> dict | None:
        raw = job.get("summary_json")
        if raw:
            return json.loads(raw)
        return None
