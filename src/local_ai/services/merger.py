from local_ai.models import TranscriptSegment, WordTiming
from local_ai.services.diarizer import DiarizationSegment
from local_ai.services.transcriber import RawSegment


def merge_transcript_diarization(
    transcript_segments: list[RawSegment],
    diarization_segments: list[DiarizationSegment],
) -> list[TranscriptSegment]:
    merged = []
    for seg in transcript_segments:
        speaker = _find_best_speaker(seg.start, seg.end, diarization_segments)
        words = None
        if seg.words:
            words = [
                WordTiming(start=w.start, end=w.end, word=w.word, confidence=w.probability)
                for w in seg.words
            ]
        merged.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=speaker,
            language=seg.language,
            words=words,
        ))
    return merged


def segments_without_diarization(transcript_segments: list[RawSegment]) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            language=seg.language,
            words=[
                WordTiming(start=w.start, end=w.end, word=w.word, confidence=w.probability)
                for w in seg.words
            ] if seg.words else None,
        )
        for seg in transcript_segments
    ]


def _find_best_speaker(
    start: float, end: float, diarization_segments: list[DiarizationSegment]
) -> str | None:
    overlap_by_speaker: dict[str, float] = {}
    for d in diarization_segments:
        overlap_start = max(start, d.start)
        overlap_end = min(end, d.end)
        if overlap_start < overlap_end:
            overlap = overlap_end - overlap_start
            overlap_by_speaker[d.speaker] = overlap_by_speaker.get(d.speaker, 0.0) + overlap

    if not overlap_by_speaker:
        return None
    return max(overlap_by_speaker, key=overlap_by_speaker.get)
