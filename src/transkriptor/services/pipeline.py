import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from opentelemetry import trace

from transkriptor.config import Settings
from transkriptor.database import Database
from transkriptor.models import TranscriptResult

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("transkriptor.pipeline")


async def _activate_gpu_service(gpu_manager_url: str, service: str) -> None:
    """Call the GPU manager to swap GPU services (whisperx ↔ vLLM).

    The DGX Spark can only run one GPU-heavy service at a time.
    *service* can be "whisperx", "vllm", "vllm-small", or "vllm/{profile}".
    """
    if not gpu_manager_url:
        return
    url = f"{gpu_manager_url.rstrip('/')}/gpu/{service}"
    logger.info("GPU swap: activating %s via %s", service, url)
    # vLLM models can take up to 600s to load; allow 900s total
    async with httpx.AsyncClient(timeout=900.0) as client:
        resp = await client.post(url)
        resp.raise_for_status()
        result = resp.json()
        logger.info("GPU swap result: %s", result)


def _resolve_vllm_profile(settings) -> str:
    """Determine the GPU manager endpoint for vLLM based on config."""
    profile = getattr(settings, "vllm_profile", "auto")
    if profile == "large":
        return "vllm"
    if profile == "small":
        return "vllm-small"
    # auto-detect from model name or port
    model = getattr(settings, "openai_model", "")
    base_url = getattr(settings, "openai_base_url", "")
    if "granite" in model.lower() or ":8001" in base_url:
        return "vllm-small"
    return "vllm"


class Pipeline:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        # Serial queue: only one GPU job at a time (whisperx ↔ vLLM can't coexist)
        self._gpu_lock = asyncio.Lock()

    async def _enqueue(self, coro, job_id: str, label: str):
        """Run a coroutine under the GPU lock so jobs don't fight over GPU services."""
        if self._gpu_lock.locked():
            logger.info("Job %s queued (%s) — waiting for GPU lock", job_id, label)
            await self.db.update_job(job_id, status_message=f"Queued (waiting for GPU)...")
        async with self._gpu_lock:
            await coro

    async def process_job(self, job_id: str) -> None:
        await self._enqueue(self._process_job(job_id), job_id, "full pipeline")

    async def _process_job(self, job_id: str) -> None:
        start_time = time.monotonic()
        with tracer.start_as_current_span(
            "pipeline.process_job",
            attributes={"job.id": job_id},
        ) as root_span:
            try:
                job = await self.db.get_job(job_id)
                if job is None:
                    return

                root_span.set_attribute("job.filename", job["filename"])
                root_span.set_attribute("job.language", job["language"])
                root_span.set_attribute("transcription.backend", self.settings.transcription_backend)
                root_span.set_attribute("summary.backend", self.settings.summary_backend)

                now = datetime.now(timezone.utc).isoformat()

                # Check if transcript was pre-populated (cached from a previous job)
                cached_transcript = job.get("transcript_json")
                if cached_transcript:
                    root_span.set_attribute("pipeline.mode", "summarize-only")
                    logger.info("Job %s: using cached transcript, skipping transcription", job_id)
                    await self.db.update_job(
                        job_id, status="summarizing", progress_pct=80,
                        started_at=now,
                        status_message="Using cached transcript — skipping to summarization...",
                    )
                    transcript_result = TranscriptResult.model_validate_json(cached_transcript)
                else:
                    root_span.set_attribute("pipeline.mode", "full")
                    await self.db.update_job(job_id, status="preprocessing", progress_pct=0, started_at=now)

                    # Step 1: Preprocess audio
                    with tracer.start_as_current_span("pipeline.preprocess") as span:
                        await self.db.update_job(job_id, status_message="Converting audio...")
                        from transkriptor.services.audio import preprocess_audio
                        output_dir = self.settings.output_dir / job_id
                        audio_info = await preprocess_audio(Path(job["file_path"]), output_dir)
                        span.set_attribute("audio.duration_secs", audio_info.duration_secs)
                        span.set_attribute("audio.wav_path", str(audio_info.wav_path))
                        await self.db.update_job(
                            job_id, progress_pct=5, duration_secs=audio_info.duration_secs,
                        )

                    lang = job["language"]
                    root_span.set_attribute("audio.duration_secs", audio_info.duration_secs)

                    # Choose transcription path: remote (DGX Spark) or local
                    if self.settings.transcription_backend == "remote":
                        # GPU swap: activate whisperx (stops vLLM if running)
                        if self.settings.gpu_manager_url:
                            await self.db.update_job(
                                job_id, status_message="Preparing GPU for transcription...",
                            )
                            await _activate_gpu_service(self.settings.gpu_manager_url, "whisperx")

                        with tracer.start_as_current_span(
                            "pipeline.transcribe_remote",
                            attributes={
                                "whisperx.url": self.settings.whisperx_url,
                                "whisperx.language": lang,
                            },
                        ) as span:
                            await self.db.update_job(
                                job_id, status="transcribing", progress_pct=5,
                                status_message="Transcribing on DGX Spark (GPU)...",
                            )
                            from transkriptor.services.remote_transcriber import remote_transcribe
                            transcript_result = await remote_transcribe(
                                audio_path=audio_info.wav_path,
                                whisperx_url=self.settings.whisperx_url,
                                language=lang,
                                min_speakers=job["min_speakers"],
                                max_speakers=job["max_speakers"],
                            )
                            speaker_count = transcript_result.speaker_count
                            span.set_attribute("transcript.segments", len(transcript_result.segments))
                            span.set_attribute("transcript.speakers", speaker_count)
                            span.set_attribute("transcript.language", transcript_result.detected_language or "")
                            await self.db.update_job(job_id, progress_pct=80)

                    else:
                        # Local: separate whisper → diarize → merge steps
                        with tracer.start_as_current_span(
                            "pipeline.transcribe_local",
                            attributes={
                                "whisper.model": self.settings.whisper_model,
                                "whisper.engine": self.settings.whisper_engine,
                            },
                        ) as span:
                            await self.db.update_job(
                                job_id, status="transcribing", progress_pct=5,
                                status_message="Transcribing audio (this may take a while)...",
                            )
                            from transkriptor.services.transcriber import transcribe

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
                            span.set_attribute("transcript.segments", len(raw_transcript.segments))
                            span.set_attribute("transcript.language", raw_transcript.detected_language or "")

                        # Free whisper model before diarization to reclaim memory
                        from transkriptor.services.transcriber import clear_model_cache
                        await asyncio.to_thread(clear_model_cache)

                        # Diarize (if enabled)
                        diarization = None
                        if job["diarization_on"] and self.settings.hf_token:
                            with tracer.start_as_current_span(
                                "pipeline.diarize",
                                attributes={"diarization.device": self.settings.diarization_device},
                            ) as span:
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
                                span.set_attribute("diarization.segments", len(diarization))

                        # Merge transcript + speaker labels
                        with tracer.start_as_current_span("pipeline.merge"):
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

                    root_span.set_attribute("transcript.speakers", transcript_result.speaker_count)
                    root_span.set_attribute("transcript.language", transcript_result.detected_language or "")

                    await self.db.update_job(
                        job_id, progress_pct=85,
                        detected_language=transcript_result.detected_language,
                        speaker_count=transcript_result.speaker_count,
                        transcript_json=transcript_result.model_dump_json(),
                    )

                # Summarize (if enabled) — non-fatal: transcript is saved even if this fails
                summary_json = None
                summary_error = None
                if job["summarization_on"]:
                    try:
                        # GPU swap: activate vLLM (stops whisperx if running)
                        if self.settings.gpu_manager_url and self.settings.summary_backend == "openai":
                            await self.db.update_job(
                                job_id, status_message="Preparing GPU for summarization...",
                            )
                            await _activate_gpu_service(self.settings.gpu_manager_url, _resolve_vllm_profile(self.settings))
                        with tracer.start_as_current_span(
                            "pipeline.summarize",
                            attributes={
                                "summary.backend": self.settings.summary_backend,
                                "summary.model": (
                                    self.settings.openai_model
                                    if self.settings.summary_backend == "openai"
                                    else self.settings.ollama_model
                                ),
                            },
                        ) as span:
                            await self.db.update_job(
                                job_id, status="summarizing", progress_pct=85,
                                status_message="Generating AI summary...",
                            )
                            from transkriptor.services.summarizer import summarize
                            style = await self.db.get_user_style_profile(job.get("user_id"))
                            summary_result = await summarize(
                                transcript=transcript_result,
                                ollama_base_url=self.settings.ollama_base_url,
                                model=self.settings.ollama_model,
                                language=self.settings.summary_language,
                                backend=self.settings.summary_backend,
                                openai_base_url=self.settings.openai_base_url,
                                openai_api_key=self.settings.openai_api_key,
                                openai_model=self.settings.openai_model,
                                style_profile=style,
                            )
                            summary_json = summary_result.model_dump_json()
                            span.set_attribute("summary.language", summary_result.language)
                            span.set_attribute("summary.topics", len(summary_result.key_topics))
                            span.set_attribute("summary.action_items", len(summary_result.action_items))
                    except Exception as sum_err:
                        summary_error = f"Summarization failed: {type(sum_err).__name__}: {sum_err}"
                        logger.warning("Job %s: %s (transcript saved)", job_id, summary_error)

                # Complete — transcript is always saved
                elapsed = time.monotonic() - start_time
                now = datetime.now(timezone.utc).isoformat()
                status_msg = "Done" if not summary_error else summary_error
                await self.db.update_job(
                    job_id,
                    status="completed",
                    progress_pct=100,
                    status_message=status_msg,
                    completed_at=now,
                    processing_secs=round(elapsed, 1),
                    summary_json=summary_json,
                )
                root_span.set_attribute("job.processing_secs", round(elapsed, 1))
                root_span.set_attribute("job.status", "completed")
                if summary_error:
                    root_span.set_attribute("job.summary_error", summary_error)
                logger.info("Job %s completed in %.1fs%s", job_id, elapsed,
                            f" (warning: {summary_error})" if summary_error else "")

            except Exception as e:
                logger.exception("Job %s failed", job_id)
                root_span.set_status(trace.StatusCode.ERROR, str(e))
                root_span.record_exception(e)
                elapsed = time.monotonic() - start_time
                await self.db.update_job(
                    job_id,
                    status="failed",
                    error_message=f"{type(e).__name__}: {str(e)}"[:1000],
                    processing_secs=round(elapsed, 1),
                )

    async def resummarize_job(self, job_id: str) -> None:
        await self._enqueue(self._resummarize_job(job_id), job_id, "resummarize")

    async def _resummarize_job(self, job_id: str) -> None:
        """Re-run only the summarization step on an existing transcript."""
        with tracer.start_as_current_span(
            "pipeline.resummarize", attributes={"job.id": job_id},
        ) as span:
            try:
                job = await self.db.get_job(job_id)
                if job is None or not job.get("transcript_json"):
                    return

                import json as _json
                transcript_result = TranscriptResult.model_validate_json(job["transcript_json"])

                # GPU swap: activate vLLM
                if self.settings.gpu_manager_url and self.settings.summary_backend == "openai":
                    await self.db.update_job(job_id, status_message="Preparing GPU for summarization...")
                    await _activate_gpu_service(self.settings.gpu_manager_url, _resolve_vllm_profile(self.settings))

                await self.db.update_job(
                    job_id, status="summarizing", progress_pct=90,
                    status_message="Generating AI summary...",
                )
                from transkriptor.services.summarizer import summarize
                style = await self.db.get_user_style_profile(job.get("user_id"))
                summary_result = await summarize(
                    transcript=transcript_result,
                    ollama_base_url=self.settings.ollama_base_url,
                    model=self.settings.ollama_model,
                    language=self.settings.summary_language,
                    backend=self.settings.summary_backend,
                    openai_base_url=self.settings.openai_base_url,
                    openai_api_key=self.settings.openai_api_key,
                    openai_model=self.settings.openai_model,
                    style_profile=style,
                )

                now = datetime.now(timezone.utc).isoformat()
                await self.db.update_job(
                    job_id,
                    status="completed",
                    progress_pct=100,
                    status_message="Done",
                    completed_at=now,
                    summary_json=summary_result.model_dump_json(),
                )
                span.set_attribute("summary.language", summary_result.language)
                logger.info("Job %s re-summarized successfully", job_id)

            except Exception as e:
                logger.exception("Job %s re-summarization failed", job_id)
                span.set_status(trace.StatusCode.ERROR, str(e))
                await self.db.update_job(
                    job_id,
                    status="completed",
                    status_message=f"Summarization failed: {type(e).__name__}: {e}",
                )
