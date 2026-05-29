# Architecture & Code Guide

## Project Structure

```
transkriptor-app/
├── pyproject.toml                    # Package metadata + dependencies
├── requirements-remote.txt           # Slim deps for K8s (no torch/pyannote)
├── Dockerfile                        # Production image (remote-backend only)
├── .env.example                      # Configuration template
├── k8s/                              # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml                # All TRANSKRIPTOR_* env vars
│   ├── deployment.yaml               # Single-replica pod
│   ├── service.yaml                  # ClusterIP on port 80
│   ├── ingress.yaml                  # nginx ingress with upload tuning
│   └── pvc.yaml                      # Persistent storage for SQLite + uploads
├── src/transkriptor/
│   ├── __init__.py
│   ├── __main__.py                   # Entry point: python -m transkriptor
│   ├── app.py                        # FastAPI factory + lifespan
│   ├── config.py                     # Pydantic Settings (env-based config)
│   ├── database.py                   # SQLite schema + async CRUD
│   ├── models.py                     # Pydantic models (Job, Transcript, Summary)
│   ├── tracing.py                    # OpenTelemetry + Instana setup
│   ├── routers/
│   │   ├── pages.py                  # HTML page routes (/, /jobs/{id}, /settings)
│   │   ├── jobs.py                   # REST API (CRUD, SSE, GPU metrics, style)
│   │   ├── exports.py                # Export downloads (TXT, SRT, JSON)
│   │   └── chat.py                   # Text Improver page + API
│   ├── services/
│   │   ├── audio.py                  # ffmpeg preprocessing (→ 16kHz mono WAV)
│   │   ├── transcriber.py            # Local faster-whisper / mlx-whisper
│   │   ├── remote_transcriber.py     # Remote WhisperX API client
│   │   ├── diarizer.py               # Local pyannote speaker diarization
│   │   ├── merger.py                 # Align transcript segments with speakers
│   │   ├── summarizer.py             # LLM summarization (Ollama / OpenAI API)
│   │   ├── text_improver.py          # Text correction via LLM
│   │   ├── style_analyzer.py         # Writing style profile builder
│   │   ├── pipeline.py               # Orchestrates full processing flow
│   │   ├── recorder.py               # Live audio recording (ffmpeg + BlackHole)
│   │   └── _torchaudio_compat.py     # torchaudio compatibility shim
│   ├── exporters/
│   │   ├── txt.py                    # Plain text export
│   │   ├── srt.py                    # SRT subtitle export
│   │   └── json_export.py            # JSON export
│   └── templates/
│       ├── base.html                 # Layout with nav (PicoCSS + HTMX)
│       ├── index.html                # Main page: upload, record, GPU widget, jobs
│       ├── job_detail.html           # Transcript viewer + summary display
│       ├── settings.html             # Style profile management
│       ├── chat.html                 # Text Improver interface
│       └── partials/
│           └── job_list.html         # HTMX partial for auto-refreshing job list
└── static/
    ├── css/app.css                   # Custom styles (GPU widget, chat, speakers)
    ├── js/app.js                     # Client-side utilities
    └── vendor/htmx.min.js           # HTMX library
```

## Application Lifecycle

### Startup (`app.py`)

1. `create_app()` is called from `__main__.py`
2. OpenTelemetry tracing is initialized (`tracing.py`)
3. FastAPI app is created with lifespan context manager
4. On startup (lifespan):
   - `Settings.ensure_dirs()` creates `data/uploads/` and `data/outputs/`
   - SQLite database is initialized (schema migration is idempotent)
   - Stuck jobs from previous crashes are recovered (set to `failed`)
   - `Pipeline` and `Recorder` instances are created and attached to `app.state`
5. Routers are registered: `pages`, `jobs`, `exports`, `chat`
6. Static files are mounted from `static/`

### Request Flow

```
Browser
  |
  ├─ GET /                    → pages.router → index.html (HTMX)
  ├─ GET /chat                → chat.router  → chat.html
  ├─ POST /api/jobs           → jobs.router  → creates job → pipeline.process_job()
  ├─ GET /api/jobs/{id}/progress → SSE stream (EventSourceResponse)
  ├─ POST /api/chat/improve   → chat.router  → text_improver.improve_text()
  └─ GET /api/gpu/metrics     → jobs.router  → DCGM + GPU Manager proxy
```

## Services

### Pipeline (`services/pipeline.py`)

The orchestrator. Manages the full audio→transcript→summary flow.

Key design decisions:
- **GPU Lock**: An `asyncio.Lock()` serializes GPU-bound work. Only one job can use WhisperX or vLLM at a time.
- **Non-fatal summarization**: If the LLM fails, the transcript is still saved. The job completes with a warning.
- **Cached re-summarization**: Jobs with existing transcripts can skip transcription and only re-run the LLM step.
- **GPU swapping**: Before each GPU-bound step, the pipeline calls the GPU Manager to activate the needed service.

The pipeline flow:
```
process_job(job_id)
  → acquire GPU lock
  → check for cached transcript
  → if no cache:
      → preprocess audio (ffmpeg)
      → GPU swap → WhisperX
      → transcribe (remote or local)
  → save transcript to DB
  → GPU swap → vLLM
  → summarize (LLM)
  → save summary to DB
  → release GPU lock
```

### Summarizer (`services/summarizer.py`)

Generates structured meeting minutes from transcripts. This is the most complex service.

**Model Profile System**: Different LLMs have different context windows. The summarizer dynamically budgets tokens:

```python
_MODEL_PROFILES = {
    "granite":     {"context_window": 8192,  "max_output_tokens": 2048, "prompt_reserve_tokens": 800},
    "gpt-oss-120b": {"context_window": 32768, "max_output_tokens": 16000, "prompt_reserve_tokens": 1500},
    "default":     {"context_window": 16384, "max_output_tokens": 8000, "prompt_reserve_tokens": 1500},
}
```

**Budget calculation**: `available_for_transcript = context_window - prompt_tokens - max_output_tokens`

For Granite (8k context): ~5400 tokens for prompt+transcript, ~2048 for output. Transcripts exceeding the budget are truncated.

**Prompt selection**: 
- Full prompt (32k+ models): rich JSON schema with `sub_points`, `status`, `remaining`, `speakers_involved`
- Compact prompt (8k models): simplified schema focusing on essential fields

**Duration-adaptive tiers**: Short (<5 min), medium (5-30 min), long (>30 min) meetings get different detail levels.

**LLM call chain**: OpenAI SDK (preferred, auto-instrumented) → httpx fallback → Ollama fallback.

**Post-processing**: Cleans garbled characters (MoE models sometimes inject Arabic/Greek), validates JSON, builds `SummaryResult` model.

### Text Improver (`services/text_improver.py`)

Corrects spelling, grammar, and clarity in pasted text while preserving the user's writing style.

- **Language detection**: Counts German stopwords — if >10% of words are German stopwords, treats as German
- **Style profile**: Loads from `data/style_profile.txt`, injected into the prompt
- **Response cleaning**: Strips `<think>` blocks and common LLM preambles ("Here's the corrected version:")
- **Input limit**: 6000 chars max to fit within Granite's 8k context window
- **Temperature**: 0.3 (slightly creative for natural-sounding corrections)

### Remote Transcriber (`services/remote_transcriber.py`)

Sends audio to the WhisperX API on DGX Spark. The remote endpoint does transcription + alignment + diarization in a single GPU-accelerated call. Returns a fully diarized `TranscriptResult` — no separate merge step needed.

Timeout is very generous (3600s read) because large audio files can take 15-30 minutes to upload and process over a slow network (~40KB/s).

### Recorder (`services/recorder.py`)

Records live system audio via ffmpeg + macOS AVFoundation. Uses BlackHole or Aggregate Device to capture meeting audio. The recording is saved as WAV and auto-submitted to the transcription pipeline when stopped.

## Database

Single `jobs` table in SQLite (via aiosqlite):

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | 12-char hex UUID |
| `filename` | TEXT | Original upload filename |
| `status` | TEXT | pending/preprocessing/transcribing/diarizing/merging/summarizing/completed/failed |
| `progress_pct` | INTEGER | 0-100 |
| `transcript_json` | TEXT | Full transcript as JSON (TranscriptResult) |
| `summary_json` | TEXT | Summary as JSON (SummaryResult) |
| `duration_secs` | REAL | Audio duration |
| `speaker_count` | INTEGER | Number of detected speakers |
| `processing_secs` | REAL | Total processing time |

## Frontend

Built with HTMX + PicoCSS — no JavaScript framework, no build step.

- **HTMX**: Auto-refreshing job list (`hx-trigger="every 5s"`), SSE progress bars, dynamic partials
- **PicoCSS**: Classless CSS framework for clean default styling
- **SSE**: Real-time progress updates during processing via `EventSourceResponse`
- **GPU Widget**: JavaScript fetches `/api/gpu/metrics` every 15s, renders bar charts
- **Chat**: JavaScript manages the improve/copy/history workflow, supports Ctrl+Enter submit

## Deployment Modes

### Remote (Production)

The Dockerfile installs only `requirements-remote.txt` — no torch, pyannote, or whisper. All ML work is offloaded to DGX Spark. The pod is lightweight (~256MB RAM).

```
TRANSKRIPTOR_TRANSCRIPTION_BACKEND=remote
TRANSKRIPTOR_SUMMARY_BACKEND=openai
```

### Local (Development)

Full `pyproject.toml` dependencies including torch, faster-whisper, pyannote.audio. Everything runs on the local machine. Needs ~16GB RAM for whisper + diarization.

```
TRANSKRIPTOR_TRANSCRIPTION_BACKEND=local
TRANSKRIPTOR_SUMMARY_BACKEND=ollama
```
