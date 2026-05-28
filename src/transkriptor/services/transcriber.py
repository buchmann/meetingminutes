import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

logger = logging.getLogger(__name__)

_model_cache = {}


@dataclass
class RawWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    language: str | None = None
    words: list[RawWord] = field(default_factory=list)


@dataclass
class RawTranscript:
    segments: list[RawSegment]
    detected_language: str | None = None
    duration_secs: float = 0.0


def clear_model_cache():
    """Free transcription models from memory. Call between pipeline steps."""
    _model_cache.clear()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except (ImportError, AttributeError):
        pass
    import gc
    gc.collect()
    logger.info("Whisper model cache cleared")


def detect_engine() -> Literal["mlx-whisper", "faster-whisper"]:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401
            logger.info("Detected macOS Apple Silicon — using mlx-whisper (GPU)")
            return "mlx-whisper"
        except ImportError:
            pass
    return "faster-whisper"


def detect_faster_whisper_device() -> tuple[str, str]:
    """Returns (device, compute_type) for faster-whisper."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected — using faster-whisper on cuda/float16")
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def transcribe(
    audio_path: Path,
    model_size: str = "large-v3",
    compute_type: str = "auto",
    engine: str = "auto",
    language: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> RawTranscript:
    if engine == "auto":
        engine = detect_engine()

    if engine == "mlx-whisper":
        return _transcribe_mlx(audio_path, model_size, language, progress_callback)
    else:
        return _transcribe_faster_whisper(audio_path, model_size, compute_type, language, progress_callback)


def _transcribe_mlx(
    audio_path: Path,
    model_size: str,
    language: str | None,
    progress_callback: Callable[[int], None] | None,
) -> RawTranscript:
    import mlx_whisper

    model_map = {
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "medium": "mlx-community/whisper-medium-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "tiny": "mlx-community/whisper-tiny-mlx",
    }
    model_id = model_map.get(model_size, f"mlx-community/whisper-{model_size}-mlx")

    lang = language if language != "auto" else None
    logger.info("Transcribing with mlx-whisper model=%s (GPU)...", model_id)

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_id,
        language=lang,
        word_timestamps=True,
        verbose=False,
    )

    detected_lang = result.get("language", lang)
    raw_segments = result.get("segments", [])
    total_duration = 0.0

    segments = []
    for seg in raw_segments:
        words = []
        for w in seg.get("words", []):
            words.append(RawWord(
                start=w["start"], end=w["end"],
                word=w["word"], probability=w.get("probability", 0.0),
            ))
        segments.append(RawSegment(
            start=seg["start"], end=seg["end"],
            text=seg["text"].strip(), language=detected_lang,
            words=words,
        ))
        if seg["end"] > total_duration:
            total_duration = seg["end"]
        if progress_callback and total_duration > 0:
            pct = min(int((seg["end"] / max(total_duration, 1)) * 100), 100)
            progress_callback(pct)

    if progress_callback:
        progress_callback(100)

    return RawTranscript(
        segments=segments, detected_language=detected_lang,
        duration_secs=total_duration,
    )


def _transcribe_faster_whisper(
    audio_path: Path,
    model_size: str,
    compute_type: str,
    language: str | None,
    progress_callback: Callable[[int], None] | None,
) -> RawTranscript:
    from faster_whisper import WhisperModel

    if compute_type == "auto":
        device, compute_type = detect_faster_whisper_device()
    else:
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

    cache_key = (model_size, device, compute_type)
    if cache_key not in _model_cache:
        logger.info("Loading Whisper model %s (device=%s, compute_type=%s)...", model_size, device, compute_type)
        _model_cache[cache_key] = WhisperModel(model_size, device=device, compute_type=compute_type)

    model = _model_cache[cache_key]

    lang = language if language != "auto" else None
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=lang,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments = []
    detected_lang = info.language
    total_duration = info.duration

    for seg in segments_gen:
        words = []
        if seg.words:
            words = [
                RawWord(start=w.start, end=w.end, word=w.word, probability=w.probability)
                for w in seg.words
            ]
        segments.append(RawSegment(
            start=seg.start, end=seg.end,
            text=seg.text.strip(), language=detected_lang,
            words=words,
        ))
        if progress_callback and total_duration > 0:
            pct = min(int((seg.end / total_duration) * 100), 100)
            progress_callback(pct)

    return RawTranscript(
        segments=segments, detected_language=detected_lang,
        duration_secs=total_duration,
    )
