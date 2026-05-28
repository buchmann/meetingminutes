def export_json(job: dict, transcript: dict | None, summary: dict | None) -> dict:
    return {
        "metadata": {
            "filename": job["filename"],
            "created_at": job["created_at"],
            "duration_secs": job.get("duration_secs"),
            "detected_language": job.get("detected_language"),
            "speaker_count": job.get("speaker_count"),
            "whisper_model": job.get("whisper_model"),
            "processing_secs": job.get("processing_secs"),
        },
        "transcript": transcript,
        "summary": summary,
    }
