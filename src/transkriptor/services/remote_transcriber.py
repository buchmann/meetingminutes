"""Remote transcription via whisperx-blackwell API on DGX Spark.

Sends audio to the Spark's /transcribe endpoint which performs
transcription + alignment + diarization in one GPU-accelerated call.
Returns a TranscriptResult directly — no separate diarize/merge steps needed.
"""

import logging
from pathlib import Path

import httpx

from transkriptor.models import TranscriptResult, TranscriptSegment, WordTiming

logger = logging.getLogger(__name__)

# Very generous timeout: a 4-hour recording can take 15-30 min to upload + transcribe on GPU
_TIMEOUT = httpx.Timeout(connect=30.0, read=3600.0, write=600.0, pool=30.0)


async def remote_transcribe(
    audio_path: Path,
    whisperx_url: str,
    language: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> TranscriptResult:
    """Send audio to whisperx-blackwell and return a fully diarized transcript.

    The remote API does whisper transcription, wav2vec2 alignment, and
    pyannote diarization in a single call on the Spark's Blackwell GPU.
    """
    url = f"{whisperx_url.rstrip('/')}/transcribe"
    logger.info("Remote transcription: sending %s to %s", audio_path.name, url)

    # Build multipart form data
    form_data: dict[str, str] = {"language": language}
    if min_speakers is not None:
        form_data["min_speakers"] = str(min_speakers)
    if max_speakers is not None:
        form_data["max_speakers"] = str(max_speakers)

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    logger.info("File size: %.1f MB", file_size_mb)

    # Determine MIME type from extension
    ext = audio_path.suffix.lower()
    mime_types = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
    }
    mime = mime_types.get(ext, "application/octet-stream")

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, mime)}
            response = await client.post(url, data=form_data, files=files)

    if response.status_code != 200:
        # Try to extract error detail from the response
        try:
            err_data = response.json()
            detail = err_data.get("detail", str(err_data))
        except Exception:
            detail = response.text[:500]
        raise RuntimeError(
            f"whisperx API returned HTTP {response.status_code}: {detail}"
        )

    data = response.json()

    if data.get("status") != "success":
        raise RuntimeError(f"whisperx API returned status={data.get('status')}: {data}")

    detected_language = data.get("language")
    num_speakers = data.get("num_speakers", 0)
    raw_segments = data.get("segments", [])

    logger.info(
        "Remote transcription complete: language=%s, speakers=%d, segments=%d",
        detected_language, num_speakers, len(raw_segments),
    )

    # Map whisperx segments → TranscriptSegment
    segments: list[TranscriptSegment] = []
    total_duration = 0.0

    for seg in raw_segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()

        if not text:
            continue

        # Speaker label: whisperx uses "SPEAKER_00" format
        speaker = seg.get("speaker")
        if speaker:
            speaker = str(speaker)

        # Word-level timestamps
        words: list[WordTiming] | None = None
        raw_words = seg.get("words")
        if raw_words:
            words = []
            for w in raw_words:
                # whisperx alignment uses "score" for confidence, may also have "word"
                word_text = str(w.get("word", ""))
                w_start = w.get("start")
                w_end = w.get("end")
                # Some words may lack timestamps after alignment
                if w_start is not None and w_end is not None:
                    words.append(WordTiming(
                        start=float(w_start),
                        end=float(w_end),
                        word=word_text,
                        confidence=w.get("score"),
                    ))

        segments.append(TranscriptSegment(
            start=start,
            end=end,
            text=text,
            speaker=speaker,
            language=detected_language,
            words=words if words else None,
        ))

        if end > total_duration:
            total_duration = end

    speaker_count = num_speakers or len(
        {s.speaker for s in segments if s.speaker}
    )

    logger.info(
        "Mapped %d segments, duration=%.1fs, speakers=%d",
        len(segments), total_duration, speaker_count,
    )

    return TranscriptResult(
        segments=segments,
        detected_language=detected_language,
        duration_secs=total_duration,
        speaker_count=speaker_count,
    )
