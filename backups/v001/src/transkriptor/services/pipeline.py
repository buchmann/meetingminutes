import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from transkriptor.config import Settings
from transkriptor.database import Database
from transkriptor.models import TranscriptResult

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db

    async def process_job(self, job_id: str) -> None:
        start_time = time.monotonic()
        try:
            job = await self.db.get_job(job_id)
            if job is None:
                return

            now = datetime.now(timezone.utc).isoformat()
            await self.db.update_job(job_id, status="preprocessing", progress_pct=0, started_at=now)

            # Step 1: Preprocess audio
            await self.db.update_job(job_id, status_message="Converting audio...")
            from transkriptor.services.audio import preprocess_audio
            output_dir = self.settings.output_dir / job_id
            audio_info = await preprocess_audio(Path(job["file_path"]), output_dir)
            await self.db.update_job(
                job_id, progress_pct=5, duration_secs=audio_info.duration_secs,
            )

            # Step 2: Transcribe
            await self.db.update_job(
                job_id, status="transcribing", progress_pct=5,
                status_message="Transcribing audio (this may take a while)...",
            )
            from transkriptor.services.transcriber import transcribe

            lang = job["language"]
            loop = asyncio.get_running_loop()

            def on_whisper_progress(pct: int):
                scaled = 5 + int(pct * 0.55)
                loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    self.db.update_job(job_id, progress_pct=scaled),
                )

            raw_transcript = await asyncio.to_thread(
                transcribe,
                audio_path=audio_info.wav_path,
                model_size=self.settings.whisper_model,
                compute_type=self.settings.whisper_compute_type,
                engine=self.settings.whisper_engine,
                language=lang,
                progress_callback=on_whisper_progress,
            )

            # Free whisper model before diarization to reclaim memory
            from transkriptor.services.transcriber import clear_model_cache
            await asyncio.to_thread(clear_model_cache)

            # Step 3: Diarize (if enabled)
            diarization = None
            if job["diarization_on"] and self.settings.hf_token:
                await self.db.update_job(
                    job_id, status="diarizing", progress_pct=60,
                    status_message="Identifying speakers...",
                )
                from transkriptor.services.diarizer import diarize
                diarization = await asyncio.to_thread(
                    diarize,
                    audio_path=audio_info.wav_path,
                    hf_token=self.settings.hf_token,
                    device=self.settings.diarization_device,
                    min_speakers=job["min_speakers"],
                    max_speakers=job["max_speakers"],
                )

            # Step 4: Merge
            await self.db.update_job(
                job_id, status="merging", progress_pct=80,
                status_message="Merging transcript with speaker labels...",
            )
            from transkriptor.services.merger import merge_transcript_diarization, segments_without_diarization

            if diarization:
                merged_segments = merge_transcript_diarization(raw_transcript.segments, diarization)
            else:
                merged_segments = segments_without_diarization(raw_transcript.segments)

            speaker_count = len(set(s.speaker for s in merged_segments if s.speaker))
            transcript_result = TranscriptResult(
                segments=merged_segments,
                detected_language=raw_transcript.detected_language,
                duration_secs=raw_transcript.duration_secs,
                speaker_count=speaker_count,
            )

            await self.db.update_job(
                job_id, progress_pct=85,
                detected_language=raw_transcript.detected_language,
                speaker_count=speaker_count,
                transcript_json=transcript_result.model_dump_json(),
            )

            # Step 5: Summarize (if enabled)
            summary_json = None
            if job["summarization_on"]:
                await self.db.update_job(
                    job_id, status="summarizing", progress_pct=85,
                    status_message="Generating AI summary...",
                )
                from transkriptor.services.summarizer import summarize
                summary_result = await summarize(
                    transcript=transcript_result,
                    ollama_base_url=self.settings.ollama_base_url,
                    model=self.settings.ollama_model,
                    language=self.settings.summary_language,
                )
                summary_json = summary_result.model_dump_json()

            # Step 6: Complete
            elapsed = time.monotonic() - start_time
            now = datetime.now(timezone.utc).isoformat()
            await self.db.update_job(
                job_id,
                status="completed",
                progress_pct=100,
                status_message="Done",
                completed_at=now,
                processing_secs=round(elapsed, 1),
                summary_json=summary_json,
            )
            logger.info("Job %s completed in %.1fs", job_id, elapsed)

        except Exception as e:
            logger.exception("Job %s failed", job_id)
            elapsed = time.monotonic() - start_time
            await self.db.update_job(
                job_id,
                status="failed",
                error_message=f"{type(e).__name__}: {str(e)}"[:1000],
                processing_secs=round(elapsed, 1),
            )
