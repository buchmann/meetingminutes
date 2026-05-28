def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def export_txt(job: dict, transcript: dict | None, summary: dict | None) -> str:
    lines = [
        "Transkriptor - Meeting Transcript",
        f"File: {job['filename']}",
        f"Date: {job['created_at'][:10]}",
    ]
    if job.get("duration_secs"):
        dur = int(job["duration_secs"])
        lines.append(f"Duration: {dur // 60}:{dur % 60:02d}")
    if job.get("speaker_count"):
        lines.append(f"Speakers: {job['speaker_count']}")
    lines.append("")

    if summary:
        lines.append("--- SUMMARY ---")
        lines.append(summary.get("overall_summary", ""))
        lines.append("")

        participants = summary.get("participants", [])
        if participants:
            lines.append("Participants: " + ", ".join(participants))
            lines.append("")

        timeline = summary.get("timeline", [])
        if timeline:
            lines.append("Timeline:")
            for entry in timeline:
                lines.append(f"  {entry}")
            lines.append("")

        topics = summary.get("key_topics", [])
        if topics:
            lines.append("Topics Discussed:")
            for t in topics:
                if isinstance(t, dict):
                    ts = f" [{t['timestamp_start']}]" if t.get("timestamp_start") else ""
                    status = f" ({t['status']})" if t.get("status") else ""
                    lines.append(f"  * {t.get('name', '')}{ts}{status}")
                    if t.get("summary"):
                        lines.append(f"    {t['summary']}")
                    for sp in t.get("sub_points", []) or []:
                        if isinstance(sp, dict):
                            lines.append(f"      - {sp.get('text', '')}")
                            if sp.get("detail"):
                                lines.append(f"        {sp['detail']}")
                        else:
                            lines.append(f"      - {sp}")
                    if t.get("remaining"):
                        lines.append("    Remaining:")
                        for r in t["remaining"]:
                            lines.append(f"      - {r}")
                    if t.get("speakers_involved"):
                        lines.append(f"    Speakers: {', '.join(t['speakers_involved'])}")
                else:
                    lines.append(f"  * {t}")
            lines.append("")

        items = summary.get("action_items", [])
        if items:
            lines.append("Action Items:")
            for item in items:
                desc = item.get("description", str(item)) if isinstance(item, dict) else str(item)
                assignee = item.get("assignee", "") if isinstance(item, dict) else ""
                deadline = item.get("deadline", "") if isinstance(item, dict) else ""
                prefix = f"[{assignee}] " if assignee else ""
                suffix = f" (Deadline: {deadline})" if deadline else ""
                lines.append(f"  - {prefix}{desc}{suffix}")
            lines.append("")

        decisions = summary.get("key_decisions", [])
        if decisions:
            lines.append("Key Decisions:")
            for d in decisions:
                lines.append(f"  - {d}")
            lines.append("")

        next_steps = summary.get("next_steps", [])
        if next_steps:
            lines.append("Next Steps:")
            for step in next_steps:
                lines.append(f"  - {step}")
            lines.append("")

        open_questions = summary.get("open_questions", [])
        if open_questions:
            lines.append("Open Questions:")
            for q in open_questions:
                lines.append(f"  ? {q}")
            lines.append("")

    if transcript and transcript.get("segments"):
        lines.append("--- TRANSCRIPT ---")
        for seg in transcript["segments"]:
            ts = _ts(seg["start"])
            speaker = seg.get("speaker") or "UNKNOWN"
            lines.append(f"[{ts}] {speaker}: {seg['text']}")

    return "\n".join(lines) + "\n"
