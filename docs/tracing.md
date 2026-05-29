# Observability & Tracing

Transkriptor uses OpenTelemetry for distributed tracing, LLM metrics, and GPU monitoring, all forwarded to IBM Instana for visualization.

## Overview

```
Transkriptor Pod (K8s)
    |
    |  OTLP HTTP traces + metrics
    |
    v
OTEL Collector (DGX Spark :4317/:4318)
    |
    |  ├── App traces (pipeline spans, LLM calls)
    |  ├── App metrics (llm.usage.*, llm.request.count)
    |  └── GPU metrics (DCGM_FI_DEV_*)      ← scraped from :9400
    |
    v
Instana Agent (DGX Spark :4328)
    |
    v
Instana Backend → Dashboards
```

## Components

### 1. Transkriptor Tracing (`tracing.py`)

Initialized at app startup by `setup_tracing()`. Handles three concerns:

#### TracerProvider (spans)

The Instana init container in K8s sets its own `TracerProvider`. Transkriptor detects this and avoids overriding it:

```python
instana_agent_host = os.environ.get("INSTANA_AGENT_HOST")
existing_provider = trace.get_tracer_provider()

if instana_agent_host and not isinstance(existing_provider, TracerProvider):
    # Instana's provider is already active — use it for traces
    logger.info("Using Instana TracerProvider")
else:
    # No Instana — set our own TracerProvider with OTLP exporter
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(...)))
    trace.set_tracer_provider(provider)
```

This prevents the `Overriding of current TracerProvider is not allowed` error that occurs when both Instana and the app try to set a provider.

#### MeterProvider (LLM metrics)

Always creates its own `MeterProvider` (Instana doesn't provide one). Sends metrics to the local Instana agent when available:

```python
if instana_agent_host:
    metrics_endpoint = f"http://{instana_agent_host}:4318/v1/metrics"
else:
    metrics_endpoint = f"{settings.otel_endpoint}/v1/metrics"
```

Creates these OTEL instruments:

| Metric | Type | Description |
|--------|------|-------------|
| `llm.usage.input_tokens` | Gauge | Prompt tokens for last request |
| `llm.usage.output_tokens` | Gauge | Completion tokens for last request |
| `llm.usage.total_tokens` | Gauge | Total tokens for last request |
| `llm.response.duration` | Gauge | LLM response latency in ms |
| `llm.request.count` | Counter | Cumulative request count |

Metrics are flushed every 15 seconds via `PeriodicExportingMetricReader`.

#### Auto-instrumentation

Instruments three libraries at startup:

| Library | Instrumentor | What it traces |
|---------|-------------|----------------|
| FastAPI | `FastAPIInstrumentor` | Incoming HTTP requests |
| httpx | `HTTPXClientInstrumentor` | Outgoing HTTP calls to Spark |
| OpenAI SDK | `OpenAIInstrumentor` (traceloop) | LLM API calls with gen_ai.* attributes |

The OpenAI instrumentor comes from `traceloop-sdk` (OpenLLMetry) and automatically creates spans with `gen_ai.request.model`, `gen_ai.usage.prompt_tokens`, etc.

### 2. Pipeline Spans

The pipeline creates a span tree for each job:

```
pipeline.process_job                    (root span)
├── pipeline.preprocess                 (ffmpeg conversion)
├── pipeline.transcribe_remote          (WhisperX API call)
│   └── (httpx auto-span)              (HTTP POST to :8003)
└── pipeline.summarize                  (LLM call)
    └── chat ibm/granite-3-3-8b-instruct  (OpenAI SDK span)
```

Each span carries relevant attributes:

```python
root_span.set_attribute("job.id", job_id)
root_span.set_attribute("job.filename", filename)
root_span.set_attribute("transcription.backend", "remote")
root_span.set_attribute("transcript.speakers", 3)
root_span.set_attribute("job.processing_secs", 253.7)
```

### 3. LLM Span Attributes

The summarizer creates manual spans with both OTEL GenAI conventions and Instana's expected attributes:

```python
with _tracer.start_as_current_span("chat ibm/granite-3-3-8b-instruct") as span:
    # OTEL GenAI semantic conventions
    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.request.temperature", 0.1)
    span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
    span.set_attribute("gen_ai.content.prompt", prompt[:4096])
    span.set_attribute("gen_ai.content.completion", content[:4096])

    # Instana aliases (their GenAI plugin reads these)
    span.set_attribute("llm.request.type", "chat")
    span.set_attribute("llm.request.model", model)
    span.set_attribute("llm.usage.input_tokens", prompt_tokens)
    span.set_attribute("llm.usage.output_tokens", completion_tokens)
    span.set_attribute("llm.usage.total_tokens", total)
```

Both naming conventions are set because:
- Instana's GenAI dashboard reads `llm.*` attributes
- OTEL ecosystem tools read `gen_ai.*` attributes
- The OTEL Collector's `transform/genai_to_llm` processor also maps `gen_ai.*` → `llm.*` for vLLM's native spans

### 4. LLM Metrics Recording

After each LLM call, metrics are recorded via the `LLMMetrics` wrapper:

```python
llm_metrics = get_llm_metrics()
if llm_metrics:
    llm_metrics.record(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total,
        duration_ms=duration_ms,
        service_name="transkriptor",
        model_id="ibm/granite-3-3-8b-instruct",
        ai_system="openai",
    )
```

Each metric gets tagged with both OTEL (`gen_ai.request.model`) and Instana (`model_id`, `ai_system`) attribute names so the Instana GenAI dashboard can filter by model.

### 5. OTEL Collector (DGX Spark)

The collector sits on the Spark and serves two purposes:

**a) Application telemetry relay**: Receives OTLP traces/metrics/logs from apps and forwards to Instana agent.

**b) GPU metrics scraper**: Scrapes Prometheus metrics from the DCGM exporter.

Config file: `/home/manfred/otel-collector-config.yaml`

```yaml
receivers:
  otlp/receiver:          # App traces + metrics (ports 4317/4318)
    protocols:
      grpc: { endpoint: "0.0.0.0:4317" }
      http: { endpoint: "0.0.0.0:4318" }

  prometheus/dcgm:        # GPU metrics (scraped from :9400)
    config:
      scrape_configs:
        - job_name: dcgm-exporter
          scrape_interval: 15s
          static_configs:
            - targets: ["192.168.178.190:9400"]

processors:
  transform/genai_to_llm:  # Map gen_ai.* → llm.* for Instana
    trace_statements:
      - set(attributes["llm.usage.input_tokens"],
            attributes["gen_ai.usage.prompt_tokens"])
        where attributes["gen_ai.usage.prompt_tokens"] != nil
      # ... (similar for output_tokens, total_tokens, duration, model)

exporters:
  otlphttp/instana:        # Forward to Instana agent
    endpoint: http://192.168.178.190:4328

pipelines:
  traces:     otlp → transform/genai_to_llm → batch → instana
  metrics/app: otlp → batch → instana
  metrics/gpu: prometheus/dcgm → batch → instana
  logs:       otlp → batch → instana
```

### 6. DCGM Exporter (GPU Metrics)

NVIDIA Data Center GPU Manager exporter runs as a container, exposing Prometheus metrics on port 9400.

Metrics scraped:

| Metric | Type | What it measures |
|--------|------|-----------------|
| `DCGM_FI_DEV_GPU_UTIL` | gauge | GPU compute utilization (%) |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | gauge | GPU memory bandwidth utilization (%) |
| `DCGM_FI_DEV_GPU_TEMP` | gauge | GPU temperature (Celsius) |
| `DCGM_FI_DEV_POWER_USAGE` | gauge | Power draw (Watts) |
| `DCGM_FI_DEV_SM_CLOCK` | gauge | Streaming multiprocessor clock (MHz) |
| `DCGM_FI_DEV_ENC_UTIL` | gauge | Video encoder utilization (%) |
| `DCGM_FI_DEV_DEC_UTIL` | gauge | Video decoder utilization (%) |
| `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | counter | Total energy since boot (mJ) |

These flow through the OTEL Collector's `metrics/gpu` pipeline to Instana, where they appear in custom dashboards.

### 7. Transkriptor GPU Widget

The app also exposes GPU metrics directly via `GET /api/gpu/metrics`, which pulls from both DCGM (raw metrics) and the GPU Manager (service status):

```json
{
    "gpu_util": 45.0,
    "mem_util": 52.0,
    "temperature": 58.0,
    "power_watts": 78.3,
    "sm_clock_mhz": 2411.0,
    "memory_available_gb": 40.7,
    "active_vllm_profile": "small",
    "whisperx_running": true,
    "vllm_running": true,
    "coexistence": true
}
```

The frontend renders this as bar charts with auto-refresh (every 15 seconds).

## Resource Attributes

The `INSTANA_PLUGIN=genai` resource attribute is critical — it tells Instana to show the service in the GenAI monitoring view:

```python
resource_attrs = {
    "service.name": "transkriptor",
    "service.version": "0.1.0",
    "INSTANA_PLUGIN": "genai",
}
```

Also set via environment variable for any library that reads `OTEL_RESOURCE_ATTRIBUTES`:
```
OTEL_RESOURCE_ATTRIBUTES=INSTANA_PLUGIN=genai
```

## Troubleshooting

### No traces in Instana

1. Check OTEL is enabled: `TRANSKRIPTOR_OTEL_ENABLED=true`
2. Check endpoint: `TRANSKRIPTOR_OTEL_ENDPOINT=http://192.168.178.190:4328`
3. Check collector health: `curl http://192.168.178.190:13133/health`
4. Check collector logs: `docker logs otel-collector --tail 20`

### No LLM metrics

1. Verify `TRACELOOP_METRICS_ENABLED=true` in configmap
2. Check that `llm.usage.*` metrics are being recorded in app logs (look for `LLM metrics recorded`)
3. Verify the metrics endpoint is correct (should be Instana agent, not Spark directly)

### TracerProvider conflict

If you see `Overriding of current TracerProvider is not allowed`:
- The Instana init container already set a TracerProvider
- `tracing.py` should detect this via `INSTANA_AGENT_HOST` env var
- If the env var is missing, set it in the configmap

### GPU metrics not showing

1. DCGM exporter running: `curl http://192.168.178.190:9400/metrics`
2. Collector config has `prometheus/dcgm` receiver
3. Collector was restarted after config change: `docker restart otel-collector`
4. Check for scrape errors in collector logs
