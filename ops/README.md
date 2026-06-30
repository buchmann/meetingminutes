# DGX Spark — operational setup (`ops/`)

These files reproduce the **DGX Spark** side of local-ai. They are NOT part of the
app image — they live on the Spark host (`manfred@192.168.178.190`). Versioned here
so a Spark reinstall is reproducible.

The Spark (NVIDIA GB10, **128 GB unified memory**, compute capability **sm_121**)
runs the GPU services the k8s app calls over the LAN.

## Services & ports

| Service | Port | Container | Image | Notes |
|---|---|---|---|---|
| WhisperX (transcribe + diarize) | 8003 | `whisper-server` | `mekopa/whisperx-blackwell:otel` | started on demand by gpu-manager |
| vLLM Granite 4.0-H-Small | 8001 | `vllm-granite-small` | `vllm/vllm-openai:latest` | BF16, 32K ctx |
| vLLM gpt-oss-120b | 8000 | `vllm-gpt-oss-cutlass` | `vllm-node-mxfp4` (custom) | MXFP4/CUTLASS, ~60 tok/s |
| vLLM bge-m3 embeddings | 8002 | `vllm-embed` | `vllm/vllm-openai:latest` | for Notes RAG |
| GPU manager | 9090 | `gpu-manager` | `python:3.12-slim` | swaps models (see below) |

**Memory rule:** only ONE large model fits at a time (granite ≈92 GB OR gpt-oss ≈83 GB,
not both). The gpu-manager stops the other when activating one. WhisperX (~5 GB) +
embeddings (~7 GB) can coexist with one LLM.

## GPU manager (`gpu-manager.py`)

A small FastAPI app that starts/stops the vLLM/whisper containers so they don't
fight over GPU memory. The app calls it before transcribe/summarize.

**Launch** (mounts itself read-only + the docker socket):

```bash
docker run -d --name gpu-manager --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /home/manfred/gpu-manager.py:/app/app.py:ro \
  -p 9090:9090 python:3.12-slim \
  bash -c "pip install --no-cache-dir fastapi uvicorn docker httpx \
    opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
    opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-asgi >/dev/null 2>&1 \
    && uvicorn app:app --host 0.0.0.0 --port 9090 --app-dir /app"
```

It runs the file at `/home/manfred/gpu-manager.py` — keep that in sync with
`ops/gpu-manager.py` here. After editing, `docker restart gpu-manager` (no --reload).

**Endpoints** (POST): `/gpu/whisperx`, `/gpu/vllm-small` (Granite), `/gpu/vllm/large`
(gpt-oss), `/gpu/vllm-bf16`, `/gpu/vllm-fp8`; `GET /status`.

**Profile → container mapping** (in `gpu-manager.py`): `small` → `vllm-granite-small`,
`large` → `vllm-gpt-oss-cutlass`, `fp8` → `vllm-granite-fp8` (defunct — FP8 crashes on
sm_121, do not use). The app's active-model switch hits `vllm-small` or `vllm/large`.

> History: the stock `/gpu/vllm-small` handler was hard-wired to the non-existent
> `vllm-granite-fp8` and would stop granite then 404 — fixed to `small`. The `large`
> profile container was repointed from the broken stock `vllm-gpt-oss-120b` to the
> working CUTLASS `vllm-gpt-oss-cutlass`.

## vLLM containers

### Granite 4.0-H-Small (port 8001, production-capable, BF16 32K)
```bash
docker run -d --name vllm-granite-small --restart unless-stopped \
  --runtime nvidia --gpus all --network host --ipc host \
  -v /home/manfred/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model ibm-granite/granite-4.0-h-small \
  --served-model-name ibm/granite-3-3-8b-instruct \
  --port 8001 --gpu-memory-utilization 0.72 --max-model-len 32768 \
  --dtype bfloat16 --enable-prefix-caching --enable-chunked-prefill --max-num-seqs 8 \
  --otlp-traces-endpoint http://192.168.178.190:4318/v1/traces
```
> Stock `vllm/vllm-openai` already bundles the OTLP exporter, so Granite needs no
> `:otel` rebuild. Endpoint = Spark OTEL collector on **:4318** (gRPC alt: :4317).

### gpt-oss-120b (port 8000, default model, MXFP4/CUTLASS)
Requires the custom **`vllm-node-mxfp4`** image — build it from
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker):
```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker && ./build-and-copy.sh --exp-mxfp4   # FULL source compile, ~1h10m on the Spark
```
Then run (the stock `vllm/vllm-openai` image does NOT work on GB10 — Marlin MXFP4
produces garbage; only this CUTLASS build is correct):
```bash
docker run -d --name vllm-gpt-oss-cutlass --restart no \
  --runtime nvidia --gpus all --network host --ipc host \
  -v /home/manfred/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
  -e OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf \
  -e OTEL_SERVICE_NAME=vllm-gpt-oss-120b \
  vllm-node-mxfp4:otel bash -c -i "vllm serve openai/gpt-oss-120b \
    --served-model-name gpt-oss-120b openai/gpt-oss-120b --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss \
    --gpu-memory-utilization 0.70 --enable-prefix-caching \
    --load-format fastsafetensors --quantization mxfp4 \
    --mxfp4-backend CUTLASS --mxfp4-layers moe,qkv,o,lm_head \
    --attention-backend FLASHINFER --kv-cache-dtype fp8 \
    --max-num-batched-tokens 8192 --max-model-len 32768 \
    --otlp-traces-endpoint http://192.168.178.190:4318/v1/traces"
```

> **Server-side OTLP tracing (`vllm-node-mxfp4:otel`):** the base `vllm-node-mxfp4`
> build ships `opentelemetry-{api,sdk}` but **not** an OTLP span exporter, so vLLM
> would crash on startup with `--otlp-traces-endpoint`. The `:otel` tag is that image
> plus the exporter — created once with:
> ```bash
> docker exec vllm-gpt-oss-cutlass pip install --no-cache-dir opentelemetry-exporter-otlp-proto-http==1.42.1
> docker commit vllm-gpt-oss-cutlass vllm-node-mxfp4:otel
> ```
> vLLM's tracer picks the exporter from `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`
> (`grpc` default → port 4317, or `http/protobuf` → port 4318/v1/traces). We use
> **http/protobuf → the Spark OTEL collector on :4318** (same path the app uses).
> `--served-model-name gpt-oss-120b openai/gpt-oss-120b` exposes both names (the second
> is the alias Instana queries). `OTEL_SERVICE_NAME` is how the spans show up in Instana.

### bge-m3 embeddings (port 8002)
```bash
docker run -d --name vllm-embed --restart unless-stopped \
  --runtime nvidia --gpus all --network host --ipc host \
  -v /home/manfred/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  BAAI/bge-m3 --served-model-name bge-m3 --port 8002 \
  --gpu-memory-utilization 0.06 --dtype bfloat16
```

### WhisperX (port 8003)
Third-party image; started on demand by the gpu-manager. CMD:
`uvicorn app.main_gpu:app --host 0.0.0.0 --port 8003`. Model `large-v3`, batch 16
baked into the image (built for sm_90 → runs but sub-optimal on sm_121).

## Switching the app's LLM

The app (Settings → Sprachmodell, admin) persists the active model in its DB and
hits the gpu-manager to swap. Manually:
```bash
curl -X POST http://localhost:9090/gpu/vllm/large    # → gpt-oss-120b (stops granite)
curl -X POST http://localhost:9090/gpu/vllm-small    # → Granite (stops gpt-oss)
curl     http://localhost:9090/status
```
A swap reloads the model (~3-5 min).
