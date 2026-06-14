"""Tiny GPU service manager for DGX Spark.

Exposes HTTP endpoints to manage GPU-heavy containers (whisperx / vLLM)
on the GB10's unified CPU+GPU memory (128 GB physical, ~119 GB visible).

Supports three vLLM model profiles:
  - vllm-large  → openai/gpt-oss-120b on port 8000 (~80 GB)
  - vllm-small  → ibm-granite/granite-4.0-h-small on port 8001 (~61 GB, BF16)
  - vllm-fp8    → ibm-granite/granite-4.0-h-small-FP8 on port 8002 (~32 GB, FP8)

When enough free memory is available, services can coexist.
If memory is tight the manager falls back to stop-one-start-the-other.

Runs as a Docker container with the Docker socket mounted.
"""

import asyncio
import logging
import os
import socket
import time

import docker
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpu-manager")

# --------------- Container registry ---------------
WHISPERX_CONTAINER = "whisper-server"
WHISPERX_HEALTH_URL = "http://localhost:8003/health"

VLLM_LARGE_CONTAINER = "vllm-gpt-oss-cutlass"
VLLM_LARGE_HEALTH_URL = "http://localhost:8000/v1/models"

VLLM_SMALL_CONTAINER = "vllm-granite-small"
VLLM_SMALL_HEALTH_URL = "http://localhost:8001/v1/models"

VLLM_FP8_CONTAINER = "vllm-granite-fp8"
VLLM_FP8_HEALTH_URL = "http://localhost:8002/v1/models"

# Backwards-compatible alias
VLLM_CONTAINER = VLLM_LARGE_CONTAINER
VLLM_HEALTH_URL = VLLM_LARGE_HEALTH_URL

# All vLLM containers
ALL_VLLM_CONTAINERS = {
    "large": {"name": VLLM_LARGE_CONTAINER, "health": VLLM_LARGE_HEALTH_URL, "port": 8000},
    "small": {"name": VLLM_SMALL_CONTAINER, "health": VLLM_SMALL_HEALTH_URL, "port": 8001},
    "fp8":   {"name": VLLM_FP8_CONTAINER,   "health": VLLM_FP8_HEALTH_URL,   "port": 8002},
}

# --------------- Memory management ---------------
MEMORY_SAFETY_MARGIN_GB = 5.0

_SERVICE_MEMORY_ESTIMATES_GB: dict[str, float] = {
    "whisperx": 5.0,
    "vllm": 80.0,
    "vllm-large": 80.0,
    "vllm-small": 61.0,
    "vllm-fp8": 32.0,
}


def _get_available_memory_gb() -> float:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 * 1024)
                    return round(gb, 2)
    except Exception as exc:
        logger.warning("Failed to read /proc/meminfo: %s", exc)
    return 0.0


def _estimate_service_memory_gb(name: str) -> float:
    return _SERVICE_MEMORY_ESTIMATES_GB.get(name, 0.0)


def _can_coexist(target_service: str) -> bool:
    avail = _get_available_memory_gb()
    needed = _estimate_service_memory_gb(target_service) + MEMORY_SAFETY_MARGIN_GB
    can = avail >= needed
    logger.info(
        "Memory check for %s: available=%.1f GB, needed=%.1f GB (est %.1f + margin %.1f) -> %s",
        target_service, avail, needed,
        _estimate_service_memory_gb(target_service), MEMORY_SAFETY_MARGIN_GB,
        "coexist" if can else "swap",
    )
    return can
# -------------------------------------------------

# --------------- OpenTelemetry tracing ---------------
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "gpu-manager")
_OTEL_ENDPOINT = os.getenv(
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "http://192.168.178.190:4318/v1/traces",
)
_TRACING_ENABLED = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _container_id = os.getenv("HOSTNAME", socket.gethostname())
    resource = Resource.create({
        "service.name": _SERVICE_NAME,
        "service.version": "1.2.0",
        "container.id": _container_id,
        "host.id": "dgx-spark",
        "host.name": "dgx-spark",
        "service.instance.id": _container_id,
    })
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=_OTEL_ENDPOINT))
    )
    trace.set_tracer_provider(provider)
    _TRACING_ENABLED = True
    logger.info("OTEL tracing enabled -> %s (service: %s)", _OTEL_ENDPOINT, _SERVICE_NAME)
except Exception as exc:
    logger.warning("OTEL tracing disabled: %s", exc)
# -----------------------------------------------------

app = FastAPI(title="GPU Manager", version="1.2")
client = docker.from_env()

if _TRACING_ENABLED:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)


def _container_running(name: str) -> bool:
    try:
        c = client.containers.get(name)
        return c.status == "running"
    except docker.errors.NotFound:
        return False


def _stop_container(name: str) -> str:
    try:
        c = client.containers.get(name)
        if c.status == "running":
            logger.info("Stopping %s...", name)
            c.stop(timeout=30)
            logger.info("%s stopped", name)
            return "stopped"
        return "already_stopped"
    except docker.errors.NotFound:
        return "not_found"


def _start_container(name: str) -> str:
    try:
        c = client.containers.get(name)
        if c.status != "running":
            logger.info("Starting %s...", name)
            c.start()
            logger.info("%s started", name)
            return "started"
        return "already_running"
    except docker.errors.NotFound:
        raise HTTPException(404, f"Container {name} not found")


async def _wait_healthy(name: str, health_url: str, timeout: int = 120) -> bool:
    import httpx

    logger.info("Waiting for %s to become healthy (timeout=%ds)...", name, timeout)
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=10.0) as http:
        while time.monotonic() - start < timeout:
            try:
                resp = await http.get(health_url)
                if resp.status_code == 200:
                    elapsed = time.monotonic() - start
                    logger.info("%s healthy after %.1fs", name, elapsed)
                    return True
            except Exception:
                pass
            await asyncio.sleep(3)

    logger.warning("%s did not become healthy within %ds", name, timeout)
    return False


@app.get("/status")
async def get_status():
    """Show which GPU services are running and memory info."""
    whisperx_up = _container_running(WHISPERX_CONTAINER)
    vllm_large_up = _container_running(VLLM_LARGE_CONTAINER)
    vllm_small_up = _container_running(VLLM_SMALL_CONTAINER)
    vllm_fp8_up = _container_running(VLLM_FP8_CONTAINER)

    active_vllm = None
    if vllm_large_up:
        active_vllm = "large"
    elif vllm_fp8_up:
        active_vllm = "fp8"
    elif vllm_small_up:
        active_vllm = "small"

    return {
        "whisperx": whisperx_up,
        "vllm": vllm_large_up or vllm_small_up or vllm_fp8_up,
        "vllm_large": vllm_large_up,
        "vllm_small": vllm_small_up,
        "vllm_fp8": vllm_fp8_up,
        "active_vllm_profile": active_vllm,
        "memory_available_gb": _get_available_memory_gb(),
        "coexistence": whisperx_up and (vllm_large_up or vllm_small_up or vllm_fp8_up),
    }


@app.post("/gpu/whisperx")
async def activate_whisperx():
    """Start whisperx - only stop vLLM if memory is too tight."""
    any_vllm_running = any(
        _container_running(info["name"]) for info in ALL_VLLM_CONTAINERS.values()
    )

    if any_vllm_running and _can_coexist("whisperx"):
        logger.info("Enough memory - starting whisperx without stopping vLLM")
        vllm_action = "kept_running"
    else:
        vllm_action = {}
        for profile, info in ALL_VLLM_CONTAINERS.items():
            cname = info["name"]
            if _container_running(cname):
                vllm_action[cname] = _stop_container(cname)
            else:
                vllm_action[cname] = "not_running"
        if any(a == "stopped" for a in vllm_action.values()):
            logger.info("Stopped vLLM container(s) to free memory for whisperx")
            await asyncio.sleep(3)

    whisperx_action = _start_container(WHISPERX_CONTAINER)
    healthy = await _wait_healthy(WHISPERX_CONTAINER, WHISPERX_HEALTH_URL, timeout=120)

    if not healthy:
        raise HTTPException(503, "whisperx failed to become healthy")

    return {
        "active": "whisperx",
        "vllm_action": vllm_action,
        "whisperx_action": whisperx_action,
        "healthy": healthy,
        "coexistence": any(
            _container_running(info["name"]) for info in ALL_VLLM_CONTAINERS.values()
        ),
        "memory_available_gb": _get_available_memory_gb(),
    }


@app.post("/gpu/vllm")
async def activate_vllm():
    """Start the default vLLM model (FP8 Granite). Backwards-compatible endpoint."""
    return await _activate_vllm_profile("small")


@app.post("/gpu/vllm-small")
async def activate_vllm_small():
    """Start the Granite model — uses FP8 for speed."""
    return await _activate_vllm_profile("small")


@app.post("/gpu/vllm-fp8")
async def activate_vllm_fp8():
    """Start the FP8-quantized Granite model on port 8002."""
    return await _activate_vllm_profile("fp8")


@app.post("/gpu/vllm-bf16")
async def activate_vllm_bf16():
    """Start the original BF16 Granite model on port 8001 (for comparison)."""
    return await _activate_vllm_profile("small")


@app.post("/gpu/vllm/{profile}")
async def activate_vllm_by_profile(profile: str):
    """Start a specific vLLM profile: 'large' (120B), 'small' (BF16), or 'fp8'."""
    if profile not in ALL_VLLM_CONTAINERS:
        raise HTTPException(400, f"Unknown profile '{profile}'. Use 'large', 'small', or 'fp8'.")
    return await _activate_vllm_profile(profile)


async def _activate_vllm_profile(profile: str):
    """Core logic: activate a vLLM profile, stopping the other if needed."""
    info = ALL_VLLM_CONTAINERS[profile]
    container_name = info["name"]
    health_url = info["health"]
    mem_key = f"vllm-{profile}"

    # Stop ALL OTHER vLLM profiles (can't run two vLLM on same GPU)
    for other_profile, other_info in ALL_VLLM_CONTAINERS.items():
        if other_profile != profile:
            other_name = other_info["name"]
            if _container_running(other_name):
                logger.info("Stopping other vLLM profile '%s' (%s)", other_profile, other_name)
                _stop_container(other_name)
                await asyncio.sleep(3)

    # Handle whisperx coexistence
    whisperx_running = _container_running(WHISPERX_CONTAINER)
    if whisperx_running and _can_coexist(mem_key):
        logger.info("Enough memory - starting vLLM-%s without stopping whisperx", profile)
        whisperx_action = "kept_running"
    else:
        whisperx_action = _stop_container(WHISPERX_CONTAINER)
        if whisperx_action == "stopped":
            logger.info("Stopped whisperx to free memory for vLLM-%s", profile)
            await asyncio.sleep(3)

    # FP8 loads faster (~200s) vs BF16 (~375s) vs large (~500s)
    timeout = 300 if profile == "fp8" else (480 if profile == "small" else 600)

    vllm_action = _start_container(container_name)
    healthy = await _wait_healthy(container_name, health_url, timeout=timeout)

    if not healthy:
        raise HTTPException(503, f"vLLM-{profile} ({container_name}) failed to become healthy")

    return {
        "active": f"vllm-{profile}",
        "profile": profile,
        "container": container_name,
        "port": info["port"],
        "whisperx_action": whisperx_action,
        "vllm_action": vllm_action,
        "healthy": healthy,
        "coexistence": whisperx_action == "kept_running",
        "memory_available_gb": _get_available_memory_gb(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
