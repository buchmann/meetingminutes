from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MERGING = "merging"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


class WordTiming(BaseModel):
    start: float
    end: float
    word: str
    confidence: float | None = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    language: str | None = None
    confidence: float | None = None
    words: list[WordTiming] | None = None


class TranscriptResult(BaseModel):
    segments: list[TranscriptSegment]
    detected_language: str | None = None
    duration_secs: float
    speaker_count: int = 0


class ActionItem(BaseModel):
    description: str
    assignee: str | None = None
    deadline: str | None = None


class TopicDetail(BaseModel):
    name: str
    summary: str
    timestamp_start: str | None = None
    speakers_involved: list[str] | None = None


class SummaryResult(BaseModel):
    overall_summary: str
    key_topics: list[str] | list[TopicDetail]
    action_items: list[ActionItem]
    key_decisions: list[str]
    participants: list[str]
    timeline: list[str] | None = None
    language: str


class JobResponse(BaseModel):
    id: str
    filename: str
    status: JobStatus
    progress_pct: int
    status_message: str | None = None
    error_message: str | None = None
    duration_secs: float | None = None
    detected_language: str | None = None
    speaker_count: int | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    processing_secs: float | None = None


class JobDetailResponse(JobResponse):
    transcript: TranscriptResult | None = None
    summary: SummaryResult | None = None
