def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(transcript: dict | None) -> str:
    if not transcript or not transcript.get("segments"):
        return ""

    lines = []
    for i, seg in enumerate(transcript["segments"], 1):
        start = _srt_ts(seg["start"])
        end = _srt_ts(seg["end"])
        speaker = seg.get("speaker")
        text = seg["text"]
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)
