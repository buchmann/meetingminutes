import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_pipeline_cache = {}


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


def diarize(
    audio_path: Path,
    hf_token: str,
    device: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[DiarizationSegment]:
    from transkriptor.services._torchaudio_compat import patch
    patch()

    import torch
    from pyannote.audio import Pipeline

    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    cache_key = device
    if cache_key not in _pipeline_cache:
        logger.info("Loading pyannote diarization pipeline (device=%s)...", device)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if device != "cpu":
            pipeline.to(torch.device(device))
        _pipeline_cache[cache_key] = pipeline

    pipeline = _pipeline_cache[cache_key]

    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    diarization = pipeline(str(audio_path), **kwargs)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append(DiarizationSegment(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
        ))

    logger.info("Diarization found %d speakers, %d segments", len(set(s.speaker for s in results)), len(results))
    return results
