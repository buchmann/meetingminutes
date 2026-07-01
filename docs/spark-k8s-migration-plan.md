# DGX Spark: Docker → Kubernetes (k3s) migration plan

Move the Spark inference/monitoring stack from raw Docker to a **standalone
single-node k3s** (already installed, PoC passed 2026-07-01). Goal: declarative,
self-healing, cleaner Instana/GPU observability — without breaking the live app
(https://local-ai.lab.allwaysbeginner.com depends on the Spark's vLLM).

## Ground truth (validated)
- Node: `gx10-870f`, **aarch64**, Ubuntu 24.04, driver **580.126.09**, GB10, 119 GB unified.
- **k3s v1.36.2+k3s1** installed minimal (`--disable traefik --disable servicelb --disable metrics-server`); coexists with Docker (live vLLM stayed HTTP 200 through install).
- NVIDIA **device-plugin v0.17.4** → `nvidia.com/gpu: 1` allocatable, **no driver reinstall**. GPU pods need `runtimeClassName: nvidia`.
- **No passwordless sudo** on the Spark → user runs privileged steps.

## Current containers → target workloads
| Docker container | Image | Port(s) | GPU | k8s target |
|---|---|---|---|---|
| vllm-gpt-oss-cutlass | vllm-node-mxfp4:otel | 8000 | ✔ | Deployment `vllm-gptoss` (replicas 0/1) |
| vllm-granite-small | vllm/vllm-openai | 8001 | ✔ | Deployment `vllm-granite` (replicas 0/1) |
| vllm-embed | vllm/vllm-openai | 8002 | ✔ (shared) | Deployment `vllm-embed` (usually 1) |
| whisper-server | whisperx image | 8003 | ✔ | Deployment `whisper` (replicas 0/1) |
| dcgm-exporter | dcgm-exporter | 9400 | ✔ | GPU-operator DCGM **or** DaemonSet |
| otel-collector | otelcol-contrib | 4317/4318 | – | Deployment `otel-collector` + ConfigMap |
| instana-agent | icr.io/instana/agent | 4328 | – | **Instana agent DaemonSet** (Helm) |
| gpu-manager.py | (host FastAPI) | 9090 | – | Deployment `gpu-manager` (rewritten, RBAC) |

## Key design decisions

### 1. GPU sharing = time-slicing (critical)
Docker shared the single GPU across containers; k8s `nvidia.com/gpu:1` is exclusive.
Restore sharing with device-plugin **time-slicing** so embeddings + one LLM (+ transient
whisper) can co-schedule:
```yaml
# ConfigMap consumed by the device plugin
version: v1
sharing:
  timeSlicing:
    resources:
      - name: nvidia.com/gpu
        replicas: 4      # advertise the 1 physical GPU as 4 schedulable slices
```
Memory is **not** partitioned (all slices share 119 GB) — which matches reality. The
"only one *large* model at a time" rule stays an orchestration policy (below), not a
scheduler guarantee. (MPS is an alternative for better compute isolation.)

### 2. Networking = hostNetwork (preserve the contract)
Keep `hostNetwork: true` on the serving pods so the app keeps calling
`192.168.178.190:8000/8001/8002/8003` **unchanged** — no Service/Ingress/URL churn on a
single node. Only one process may bind a port at a time → drives the per-service cutover.

### 3. Storage = hostPath to the existing HF cache
Mount `/home/manfred/.cache/huggingface` as a `hostPath` volume so model weights are
**not re-downloaded**. Single node → hostPath is appropriate (no PV provisioner needed).

### 4. Images into k3s containerd (separate from Docker!)
k3s uses its own containerd image store. Bring images over one of two ways:
- **Bake a proper image** for the CUTLASS+OTLP build (replace the `docker commit` :otel hack):
  `Dockerfile: FROM vllm-node-mxfp4 → pip install opentelemetry-exporter-otlp-proto-http==1.42.1`.
- Import: `docker save vllm-node-mxfp4:otel | sudo k3s ctr images import -` (or push to a registry).

### 5. Model switching = replica scaling (replaces gpu-manager swap)
`nvidia.com/gpu:1` (with time-slicing) no longer force-serialises, so the "one large
model" policy is explicit: activating a model scales its Deployment to 1 and the other
large model(s) to 0. Two options:
- **Rewrite `gpu-manager.py`** to use the k8s Python client (`scale deployment`) instead
  of the Docker SDK; run it as a Deployment with a ServiceAccount + RBAC (patch/scale on
  deployments), keep `hostNetwork` :9090 so the app's existing `POST :9090/gpu/vllm/large`
  calls work **unchanged** (minimal blast radius). ← recommended
- Or point the app's model-switch directly at the k3s API (more app change).

## Phased rollout (each phase reversible)

**Phase 1 — Cluster prep** *(no disruption)*
- Namespace `spark-ai`; label node; create `nvidia` RuntimeClass if missing.
- Device-plugin **time-slicing** ConfigMap (replicas: 4) + restart plugin.
- `hostPath` volume manifest for the HF cache; ConfigMaps for collector config; Secrets.
- Build/import the `vllm-node-mxfp4:otel` image into k3s containerd.

**Phase 2 — Serving cutover** *(brief per-service downtime)*
For each of gptoss / granite / embed / whisper: write the Deployment (hostNetwork,
runtimeClassName nvidia, `nvidia.com/gpu:1`, hostPath cache, env + serve flags copied
1:1 from `ops/README.md`), then **stop the Docker container and apply the Deployment**.
Verify `:8000/health` etc. after each. Set `--restart no` semantics via k8s (no auto-run
of the heavy model unless scaled).

**Phase 3 — Monitoring on k8s** *(cleaner + fixes restart bug)*
- **Instana agent DaemonSet** via Helm (auto-restart, native k8s + host + container
  monitoring) → retires the `restart=no` docker agent that died silently for 3 weeks.
- **NVIDIA GPU Operator** (with `driver.enabled=false`, `toolkit.enabled=false` — host has
  both) for the standard DCGM exporter → `oTelDcgm`.
- **OTel collector** Deployment with our existing pipelines (traces, metrics/app,
  metrics/gpu, metrics/vllm incl. the `resource/vllm` INSTANA_PLUGIN=vllm tagging).

**Phase 4 — Cutover finish**
- Rewrite + deploy `gpu-manager` (scaling controller, RBAC).
- Decommission the Docker containers; k3s already `systemd`-enabled on boot.
- Update `ops/` manifests + `docs/monitoring.md`.

## Risks / gotchas
- **Time-slicing is mandatory** or concurrent embed+LLM breaks (exclusive GPU).
- **119 GB unified memory is unmanaged by k8s** → keep the one-large-model policy; set pod
  `resources.requests/limits` for CPU/RAM but GPU memory stays a manual budget.
- **Port cutover**: a hostNetwork pod and its Docker twin can't both bind :8000 — stop the
  container first (short downtime; app users see a blip).
- **k3s image store ≠ Docker** — import images explicitly.
- Control-plane overhead ~1–2 GB (negligible on 119 GB).
- All privileged steps need the user (no passwordless sudo).

## Rollback
Per-service: scale the Deployment to 0 and `docker start <container>` (the Docker
containers/images remain until Phase 4). Whole PoC/cluster: `sudo /usr/local/bin/k3s-uninstall.sh`.
