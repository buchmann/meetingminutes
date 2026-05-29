# Transkriptor

Local, privacy-first meeting transcription app. Upload or record audio, get speaker-attributed transcripts with timestamps and AI-generated meeting minutes. All processing runs on your own hardware — nothing leaves your network.

## What It Does

- **Transcription** — WhisperX on GPU produces word-level timestamps with automatic language detection (English/German/mixed)
- **Speaker Diarization** — identifies who said what, with color-coded speaker labels
- **AI Summaries** — structured meeting minutes with topics, action items, decisions, timeline, and next steps
- **Text Improver** — paste emails or Slack messages, get corrected versions in your personal writing style
- **Live Recording** — record system audio directly from the browser via BlackHole/Aggregate Device
- **Export** — download transcripts as TXT, SRT (subtitles), or JSON
- **GPU Monitoring** — live dashboard showing GPU utilization, temperature, power, and running services

## Architecture

```
Browser (HTMX + PicoCSS)
    |
    v
Kubernetes (.35 cluster, nginx ingress)
    |
    v
Transkriptor Pod (FastAPI, SQLite, SSE progress)
    |
    +---> DGX Spark GPU Server (192.168.178.190)
    |       |--- WhisperX         (port 8003)  — transcription + diarization
    |       |--- vLLM Granite 8B  (port 8001)  — summarization (default, 8k context)
    |       |--- vLLM GPT-OSS 120B (port 8000) — summarization (alt, 32k context)
    |       |--- GPU Manager      (port 9090)  — orchestrates GPU memory allocation
    |       |--- DCGM Exporter    (port 9400)  — GPU metrics
    |       |--- OTEL Collector   (port 4317)  — telemetry relay to Instana
    |
    +---> Instana Agent (observability)
            |--- Application traces (pipeline spans, LLM calls)
            |--- LLM metrics (token counts, latency)
            |--- GPU metrics (utilization, temp, power)
```

The app runs in two modes:
- **Remote mode** (production): lightweight pod in Kubernetes, GPU work offloaded to DGX Spark
- **Local mode** (development): everything on one machine with faster-whisper + pyannote + Ollama

## Processing Pipeline

```
Audio Upload / Recording
    |
    v
1. Preprocess         — ffmpeg converts to 16kHz mono WAV        [0-5%]
    |
    v
2. GPU Swap           — GPU Manager activates WhisperX            [5%]
    |
    v
3. Transcribe         — WhisperX large-v3 with word timestamps   [5-80%]
   + Diarize            (transcription + diarization in one call)
    |
    v
4. GPU Swap           — GPU Manager activates vLLM (Granite or 120B) [80%]
    |
    v
5. Summarize          — Granite 3.3 8B generates structured JSON  [80-98%]
    |
    v
6. Store + Export     — results in SQLite, exports generated      [98-100%]
```

GPU-bound work runs under an async lock so jobs don't fight over GPU memory. The GPU Manager handles the WhisperX/vLLM coexistence — both can run simultaneously when memory allows (~40GB free on 128GB DGX Spark).

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for building)
- Kubernetes cluster with nginx ingress
- DGX Spark (or any NVIDIA GPU server) running WhisperX + vLLM

### Local Development

```bash
# Clone and install
git clone https://github.com/buchmann/meetingminutes.git
cd meetingminutes
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your settings

# Run
python -m transkriptor
# Open http://127.0.0.1:8000
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Build and push image
docker build -t mbx1010/transkriptor:latest .
docker push mbx1010/transkriptor:latest
kubectl rollout restart deployment/transkriptor -n transkriptor
```

### DGX Spark Services

The GPU server needs these containers running:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| WhisperX | `mekopa/whisperx-blackwell:otel` | 8003 | Transcription + diarization |
| vLLM Granite 8B | `vllm/vllm-openai:latest` | 8001 | LLM summarization (default, 8k ctx) |
| vLLM GPT-OSS 120B | `vllm/vllm-openai:latest` | 8000 | LLM summarization (large, 32k ctx) |
| GPU Manager | `python:3.12-slim` | 9090 | GPU memory orchestration |
| DCGM Exporter | `nvcr.io/nvidia/k8s/dcgm-exporter` | 9400 | GPU metrics |
| OTEL Collector | `otel-collector-contrib` | 4317/4318 | Telemetry relay |

> **Note:** Only one vLLM model runs at a time. Granite (61GB) can coexist with WhisperX; the 120B model (90GB) needs the full GPU. The GPU Manager handles switching automatically.

## Configuration

All settings use the `TRANSKRIPTOR_` prefix and can be set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIPTION_BACKEND` | `local` | `local` or `remote` (DGX Spark) |
| `WHISPERX_URL` | `http://192.168.178.190:8003` | Remote WhisperX endpoint |
| `SUMMARY_BACKEND` | `ollama` | `ollama` or `openai` (vLLM) |
| `OPENAI_BASE_URL` | `http://192.168.178.190:8001/v1` | vLLM endpoint |
| `OPENAI_MODEL` | `ibm/granite-3-3-8b-instruct` | LLM model (`ibm/granite-3-3-8b-instruct` or `openai/gpt-oss-120b`) |
| `GPU_MANAGER_URL` | (empty) | GPU Manager endpoint |
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `OTEL_ENDPOINT` | `http://localhost:4318` | OTLP HTTP endpoint |
| `LANGUAGE` | `auto` | `en`, `de`, or `auto` |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload file size |

See `src/transkriptor/config.py` for the complete list.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/jobs` | Upload audio + start processing |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/jobs/{id}` | Job detail with transcript + summary |
| `DELETE` | `/api/jobs/{id}` | Delete job + files |
| `GET` | `/api/jobs/{id}/progress` | SSE stream for live progress |
| `GET` | `/api/jobs/{id}/export/{fmt}` | Download as txt/srt/json |
| `POST` | `/api/jobs/{id}/resummarize` | Re-run summarization only |
| `POST` | `/api/jobs/reprocess` | Reprocess existing file |
| `POST` | `/api/chat/improve` | Text improvement via LLM |
| `GET` | `/api/gpu/metrics` | GPU status from DCGM + GPU Manager |
| `GET` | `/api/health` | Service health check |
| `POST` | `/api/style/analyze` | Build writing style profile |

## Documentation

- **[Architecture & Code Guide](docs/architecture.md)** — code structure, services, data flow
- **[Model Integration](docs/models.md)** — how Granite, WhisperX, and vLLM work together
- **[Observability & Tracing](docs/tracing.md)** — OpenTelemetry setup, Instana integration, GPU metrics

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.12, asyncio |
| Frontend | HTMX, PicoCSS, Jinja2 templates |
| Database | SQLite via aiosqlite |
| Transcription | WhisperX (large-v3) on NVIDIA Blackwell GPU |
| LLM | IBM Granite 3.3 8B (8k ctx) or GPT-OSS 120B (32k ctx) via vLLM |
| Observability | OpenTelemetry, Instana, DCGM Exporter |
| Deployment | Docker, Kubernetes, nginx ingress |
| GPU Server | NVIDIA DGX Spark (128GB unified memory) |
