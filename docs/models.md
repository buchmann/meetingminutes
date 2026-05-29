# Model Integration

How Transkriptor works with WhisperX, Granite, and the GPU Manager on DGX Spark.

## Hardware

**NVIDIA DGX Spark** — 128GB unified memory (CPU + GPU shared), Blackwell B200 architecture.

Unified memory means WhisperX (~5GB) and vLLM Granite (~61GB) can coexist simultaneously, with ~40GB free. The GPU Manager monitors this and enables co-running when memory allows.

## Models

### WhisperX (Transcription + Diarization)

| Property | Value |
|----------|-------|
| Service | `mekopa/whisperx-blackwell:otel` |
| Port | 8003 |
| Model | `large-v3` (1.6B parameters) |
| GPU Memory | ~5GB |
| Languages | English, German, auto-detect |

WhisperX combines three models in one pipeline:
1. **Whisper large-v3** — speech-to-text with word-level timestamps
2. **wav2vec2** — forced alignment for precise word timing
3. **pyannote** — speaker diarization (who spoke when)

The API accepts audio via multipart upload and returns a fully diarized transcript in one call:

```
POST http://192.168.178.190:8003/transcribe
Content-Type: multipart/form-data

file=@meeting.wav
language=auto
min_speakers=2
max_speakers=6
```

Response: JSON with segments, each containing speaker label, start/end timestamps, text, and word-level timing.

### IBM Granite 3.3 8B (Summarization + Text Improvement)

| Property | Value |
|----------|-------|
| Service | `vllm/vllm-openai:latest` |
| Port | 8001 |
| Model | `ibm/granite-3-3-8b-instruct` |
| Architecture | Mixture of Experts (MoE) hybrid |
| Context Window | 8192 tokens |
| GPU Memory | ~61GB |
| vLLM Flag | `--max-model-len 8192` |

Granite is served via vLLM with an OpenAI-compatible API. Transkriptor uses the OpenAI Python SDK to call it:

```
POST http://192.168.178.190:8001/v1/chat/completions
{
    "model": "ibm/granite-3-3-8b-instruct",
    "messages": [{"role": "user", "content": "..."}],
    "temperature": 0.1,        # Low for factual summarization
    "max_tokens": 2048,
    "response_format": {"type": "json_object"},
    "extra_body": {
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": false}
    }
}
```

Key parameters:
- **Temperature 0.1** for summarization (factual accuracy), **0.3** for text improvement (more natural)
- **`response_format: json_object`** forces valid JSON output
- **`enable_thinking: false`** disables Granite's chain-of-thought `<think>` blocks
- **`repetition_penalty: 1.1`** prevents repetitive output from MoE models

## Context Window Management

Granite's 8192-token context window requires careful budgeting:

```
Total: 8192 tokens
├── Prompt template:   ~800 tokens (compact prompt)
├── Transcript:        ~5344 tokens (~21,376 chars)
└── Output:            ~2048 tokens (summary JSON)
```

### How the budget is calculated

```python
profile = {"context_window": 8192, "max_output_tokens": 2048, "prompt_reserve_tokens": 800}
chars_per_token = 4  # English text average

prompt_tokens = len(template) // 4 + 800  # template + safety margin
available = 8192 - prompt_tokens - 2048    # what's left for transcript
max_chars = available * 4                   # convert to characters
```

For a typical meeting transcript, this allows ~20,000 characters (~11 minutes of conversation). Longer transcripts are truncated.

### Prompt selection

The summarizer automatically picks the right prompt based on context window:

| Context Window | Prompt Style | JSON Schema |
|----------------|-------------|-------------|
| <= 8192 | Compact | `overall_summary`, `key_topics` (name + summary only), `action_items`, `key_decisions`, `timeline`, `participants`, `next_steps`, `open_questions` |
| > 8192 | Full | Same + `sub_points`, `status`, `remaining`, `speakers_involved`, `timestamp_start` per topic |

### Duration-adaptive detail

| Meeting Length | Topics | Timeline Entries | Detail Level |
|---------------|--------|-----------------|--------------|
| < 5 min | 2-3 | 2-4 | Brief (2-3 sentences per topic) |
| 5-30 min | 3-6 | 4-8 | Standard (short paragraph per topic) |
| > 30 min | 5-10 | 6-15 | Detailed (full paragraph per topic) |

## GPU Manager

The GPU Manager is a lightweight FastAPI service running on the Spark that orchestrates GPU memory allocation.

### Endpoints

| Method | Path | Action |
|--------|------|--------|
| `GET` | `/status` | Current state (memory free, active services) |
| `POST` | `/gpu/whisperx` | Activate WhisperX for transcription |
| `POST` | `/gpu/vllm-small` | Activate vLLM Granite (small profile) |
| `POST` | `/gpu/vllm` | Activate vLLM 120B (large profile) |

### Coexistence logic

The GPU Manager checks available memory before starting a service:

```
WhisperX:  ~5GB  → always fits alongside Granite
Granite:   ~61GB → fits alongside WhisperX (128 - 61 - 5 = 62GB free)
120B:      ~90GB → cannot coexist with anything
```

When both WhisperX and Granite are loaded, the pipeline skips GPU swap calls entirely — no container restarts needed between transcription and summarization steps.

### Startup timeout

Granite (61GB MoE model) takes ~375 seconds to load into GPU memory. The GPU Manager waits up to 480 seconds for the health check to pass before timing out.

## Auto-detection

The pipeline auto-detects which vLLM profile to use:

```python
def _resolve_vllm_profile(settings):
    if "granite" in model.lower() or ":8001" in base_url:
        return "vllm-small"   # → POST /gpu/vllm-small
    return "vllm"             # → POST /gpu/vllm (120B)
```

The summarizer similarly auto-selects the model profile:

```python
def _get_model_profile(model, base_url):
    if "granite" in model.lower() or ":8001" in base_url:
        return _MODEL_PROFILES["granite"]     # 8k context
    if "120b" in model.lower():
        return _MODEL_PROFILES["gpt-oss-120b"]  # 32k context
    return _MODEL_PROFILES["default"]           # 16k context
```

## LLM Call Chain

Both the summarizer and text improver follow the same call priority:

```
1. OpenAI SDK (openai.AsyncOpenAI)
   ├── Auto-instrumented by traceloop-sdk for Instana GenAI spans
   ├── Manual gen_ai.* span attributes for token counts
   └── OTEL metrics recording (llm.usage.*, llm.request.count)

2. httpx fallback (if openai package unavailable)
   └── Raw HTTP POST to /v1/chat/completions

3. Ollama fallback (if backend="ollama")
   └── ollama.AsyncClient for local models
```

## Writing Style Profile

The Text Improver and summarizer can apply a personal writing style. The style profile is generated by analyzing writing samples (pasted text or Apple Mail sent emails) and stored at `data/style_profile.txt`.

The profile is a 150-250 word instruction paragraph like:
> "Write in the following style: Use direct, informal tone. Prefer short sentences. Mix technical terms with plain language..."

This gets injected into the LLM prompt so corrections and summaries match the user's voice.
