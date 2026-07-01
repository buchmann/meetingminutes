# Instana GenAI & vLLM Monitoring — reference + gap analysis for local-ai

Extracted from the official *IBM Instana Observability* documentation (961 pp.,
GenAI chapter ~pp. 2384–2442) and mapped to the **local-ai / DGX Spark** setup.

Instana observes GenAI in **three layers**, each with its own ingest path:

| Layer | What | Instana sensor | How it's fed |
|-------|------|----------------|--------------|
| **1. Application LLM calls** | traces + token/cost/latency of your app's LLM interactions | **OpenTelemetry Gen AI sensor** (v1.0.6, 2026-06-25) | **OpenLLMetry** (traceloop-sdk) → OTLP |
| **2. vLLM server traces** | per-request inference spans | Gen AI sensor / LLM Task view | `vllm serve --otlp-traces-endpoint` |
| **3. vLLM infra metrics** | throughput, KV-cache, queue, latency | **vLLM sensor** → `oTelVLLM` entity | Prometheus `/metrics` → OTel collector **with vLLM resource attrs** |
| **(+) GPU** | DCGM GPU metrics | `oTelDcgm` entity | dcgm-exporter → OTel collector |

> The **OpenTelemetry Gen AI sensor is backend-side** — Instana runs and updates
> it automatically (nothing to install on your side). You only produce correctly
> tagged OTLP telemetry; the sensor turns it into the GenAI dashboard + entities.

---

## Layer 1 — Application GenAI via OpenLLMetry  *(you already have this)*

Instana's words: *"Add two lines of code to instrument your application with
OpenLLMetry, then configure environment variables to point the telemetry to
Instana."*

**Required env (all modes):**
```
OTEL_RESOURCE_ATTRIBUTES="INSTANA_PLUGIN=genai"
TRACELOOP_LOGGING_ENABLED=true
TRACELOOP_METRICS_ENABLED=true
```
**Agent mode** (telemetry via a local/host Instana agent):
```
TRACELOOP_BASE_URL=<instana-agent-host>:4317
OTEL_EXPORTER_OTLP_INSECURE=true
```
**Agentless mode** (direct to backend):
```
TRACELOOP_BASE_URL=<instana-otlp-endpoint>:4317
TRACELOOP_HEADERS="x-instana-key=<agent-key>,x-instana-host=<instana-host>"
OTEL_EXPORTER_OTLP_INSECURE=false
```

**local-ai status:** ✅ Implemented — `src/local_ai/tracing.py` uses the OpenAI/
Ollama OpenLLMetry instrumentors, `OTEL_RESOURCE_ATTRIBUTES=INSTANA_PLUGIN=genai`,
`TRACELOOP_LOGGING/METRICS_ENABLED=true` (see `k8s/configmap.yaml`). Runtime uses
**agent mode** via the Kubernetes operator-injected agent (192.168.200.12). App
LLM calls already land in the GenAI dashboard.

---

## Layer 2 — vLLM server-side traces  *(you already have this)*

Instana's exact recipe:
```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://<instana-agent-host>:4317"
export OTEL_SERVICE_NAME="<your-service-name>"
vllm serve <model> --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
```
Also install the OTLP exporter packages (the doc pins 1.26.x; we use 1.42.x to
match the image):
```
opentelemetry-sdk / -api / -exporter-otlp / -semantic-conventions-ai
```

**local-ai status:** ✅ Implemented 2026-06-30 — gpt-oss (`vllm-node-mxfp4:otel`,
`--otlp-traces-endpoint …:4318/v1/traces`, http/protobuf) and Granite both emit
server spans. We route them through the **Spark OTel collector** (which has the
`transform/genai_to_llm` processor) instead of straight to the agent — equivalent
result, and it lets us relabel/transform.

---

## Layer 3 — vLLM infrastructure metrics  ⚠️ **GAP — action needed**

This is what creates the proper **`oTelVLLM` entity** and the vLLM performance
dashboard (throughput, KV-cache, latency, queue). Instana offers two options:

- **Option 1 — ODCV** (Java "OTel Data Collector for vLLM", `otel-dc-vllm`): a
  dedicated collector that reads vLLM `/metrics` by semantic convention.
- **Option 2 — OpenTelemetry collector** (IDOT or open-source contrib): scrape
  vLLM `/metrics` and forward, **with a `resource` processor that stamps the vLLM
  identity attributes**. ← we already run a collector, so this is the low-effort path.

**The critical part we're missing** — the doc's Option 2 (agent mode) `resource`
processor:
```yaml
processors:
  resource:
    attributes:
      - { key: service.name,        value: vllm,          action: insert }
      - { key: service.namespace,   value: genai,         action: insert }
      - { key: service.instance.id, value: <vllm-host>,   action: insert }
      - { key: server.address,      value: <vllm-address>,action: insert }
      - { key: server.port,         value: "8000",        action: insert }
      - { key: vllm.entity.type,    value: vllm,          action: insert }   # ← makes the oTelVLLM entity
      - { key: INSTANA_PLUGIN,      value: vllm,          action: insert }   # ← routes to the vLLM sensor
exporters:
  otlphttp: { endpoint: "http://<instana-agent-host>:4318", tls: { insecure: true } }
service:
  pipelines:
    metrics/prometheus: { receivers: [prometheus], processors: [resource, batch], exporters: [otlphttp] }
```

**local-ai status:** ⚠️ **Partial.** Our Spark collector's `prometheus/vllm`
pipeline scrapes `:8000`/`:8001` and forwards, but with **only `batch`** (labels
`host.name` + `service.name`). It is **missing `INSTANA_PLUGIN=vllm` and
`vllm.entity.type=vllm`**, so the metrics arrive as generic *Custom OpenTelemetry
gauges* rather than under a first-class **`oTelVLLM`** entity with the vLLM sensor
dashboard.

**Fix:** add a `resource/vllm` processor with those two keys (plus
service.namespace=genai, server.address/port) to the `metrics/vllm` pipeline in
`/home/manfred/otel-collector-config.yaml`, then `docker restart otel-collector`.
This is a config-only change on the Spark; no container rebuild.

---

## GPU metrics (DCGM)  *(you already have this)*

Doc: install OTel Data Collector → view under **Infrastructure › Analyze
Infrastructure › OTEL Dcgm**. Metrics: `DCGM_FI_DEV_GPU_TEMP`, `POWER_USAGE`,
`GPU_UTIL`, `MEM_COPY_UTIL`, `FB_USED/FB_FREE`, `SM_CLOCK`, `MEM_CLOCK`.

**local-ai status:** ✅ `prometheus/dcgm` scrape → `oTelDcgm` entity, 10 panels
live. Note GB10 does **not** emit `FB_USED/FB_FREE` (unified LPDDR5X) or `PROF_*`.

---

## Summary — where local-ai stands vs. the Instana doc

| Capability | Doc approach | local-ai | Action |
|-----------|--------------|----------|--------|
| App LLM traces + token metrics | OpenLLMetry + `INSTANA_PLUGIN=genai` | ✅ | — |
| GenAI backend sensor (1.0.6) | backend-managed | ✅ (auto) | — |
| vLLM server-side traces | `--otlp-traces-endpoint` | ✅ | — |
| **vLLM infra metrics → `oTelVLLM`** | resource attrs `INSTANA_PLUGIN=vllm`, `vllm.entity.type=vllm` | ⚠️ partial | **add resource processor** |
| GPU DCGM → `oTelDcgm` | OTel collector scrape | ✅ | — |
| Instana agent reachable | agent up on the Spark | ✅ (restarted 06-30) | consider `restart=unless-stopped` |

**Bottom line:** the GenAI *observability* pipeline is essentially complete. The
one meaningful gap is **Layer 3**: to get vLLM into a proper `oTelVLLM` entity and
the vLLM sensor dashboard, add the `INSTANA_PLUGIN=vllm` + `vllm.entity.type=vllm`
resource attributes to the collector's vLLM metrics pipeline.

*(Separate topic: "Instana chat not answering" is Instana's **active** GenAI
assistant calling your model — not observability. The model itself answers
correctly; see the browser test page `~/Desktop/gpt-oss-chat.html`.)*
