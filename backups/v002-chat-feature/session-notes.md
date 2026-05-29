# Transkriptor Session Backup — 2026-05-29

## Session Summary

This session continued from a previous long session that covered:
- Switching to Granite as default model everywhere
- Fixing 503 errors, context window overflows, empty summaries
- Fixing k8s deployment to correct cluster (.35 only)
- Fixing 28MB file upload through nginx ingress
- OTEL tracing/metrics fix for Instana
- Model profile system for context window budgeting
- Compact prompts for Granite's 8k context

## What was done in THIS session

### 1. Text Improver Chat Feature (NEW)
Implemented a full chat interface for correcting/improving pasted text (emails, Slack messages).

**New files created:**
- `src/transkriptor/services/text_improver.py` — Core service
  - Language detection (German stopword heuristic)
  - Prompt building (EN/DE) with optional style profile
  - LLM call via OpenAI SDK or httpx with OTEL tracing
  - Response cleaning: strips `<think>` tags and LLM preambles
  - 6000-char input limit for Granite 8k context
  - Temperature 0.3

- `src/transkriptor/routers/chat.py` — API router
  - `GET /chat` — renders chat page
  - `POST /api/chat/improve` — accepts JSON `{text: "..."}`, returns improved text

- `src/transkriptor/templates/chat.html` — UI template
  - Textarea input with char counter (6000 max)
  - "Improve" button + Ctrl/Cmd+Enter shortcut
  - Side-by-side result: improved text + collapsible original
  - Copy button
  - Session history with click-to-reload
  - Style profile badge

**Modified files:**
- `src/transkriptor/app.py` — registered chat router
- `src/transkriptor/templates/base.html` — added "Text Improver" nav link
- `static/css/app.css` — chat-specific styles

### 2. GPU Metrics Integration (NEW)
Connected DCGM exporter metrics to both Instana and the Transkriptor UI.

**OTEL Collector (on DGX Spark 192.168.178.190):**
- Added `prometheus/dcgm` receiver to `/home/manfred/otel-collector-config.yaml`
- Scrapes `http://192.168.178.190:9400/metrics` every 15s
- Separate `metrics/gpu` pipeline forwards to Instana agent at `:4328`
- Metrics: GPU util, memory util, temp, power, SM clock, encoder/decoder util

**Transkriptor UI:**
- New `GET /api/gpu/metrics` endpoint in `routers/jobs.py`
  - Pulls from DCGM exporter (port 9400) + gpu-manager (port 9090)
  - Returns: gpu_util, mem_util, temperature, power_watts, sm_clock_mhz,
    memory_available_gb, active_vllm_profile, whisperx_running, vllm_running, coexistence

- GPU status widget on main page (`templates/index.html`)
  - 4 bar charts: GPU%, Mem%, Temp, Power
  - Service badges: WhisperX, vLLM profile, memory free, co-run status
  - Auto-refreshes every 15s

- CSS styles in `static/css/app.css` for the GPU widget

## Architecture Overview

```
User Browser
    |
    v
nginx ingress (.35 cluster, port 80/443)
    |
    v
Transkriptor Pod (k8s, namespace: transkriptor)
    |--- /chat page + /api/chat/improve → text_improver.py → Granite LLM
    |--- /api/gpu/metrics → DCGM exporter + gpu-manager on Spark
    |--- /api/jobs → pipeline → WhisperX + Granite summarizer
    |
    v
DGX Spark (192.168.178.190)
    |--- vLLM Granite (port 8001)
    |--- WhisperX (port 8003)
    |--- GPU Manager (port 9090)
    |--- DCGM Exporter (port 9400) → OTEL Collector → Instana Agent
    |--- OTEL Collector (ports 4317/4318) → Instana Agent (port 4328)
    |--- Instana Agent → Instana Backend
```

## Key Config Values
- k8s context: `kubernetes-admin@kubernetes` (master .35)
- Build machine: 192.168.178.61
- Docker image: `mbx1010/transkriptor:latest`
- DGX Spark: 192.168.178.190
- Granite model: `ibm/granite-3-3-8b-instruct` on port 8001
- OTEL collector config: `/home/manfred/otel-collector-config.yaml` on Spark
- Style profile: `data/style_profile.txt` on the pod

## Deployment Pipeline
```
1. Edit code locally on Mac
2. rsync → manfred@192.168.178.61:/tmp/transkriptor-build/
3. ssh .61: docker build -t mbx1010/transkriptor:latest .
4. ssh .61: docker push mbx1010/transkriptor:latest
5. kubectl rollout restart deployment/transkriptor -n transkriptor
```
