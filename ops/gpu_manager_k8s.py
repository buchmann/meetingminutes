"""GPU service manager for DGX Spark — Kubernetes-native (v2).

Scales k8s Deployments in namespace `spark-ai` instead of docker containers,
preserving the HTTP contract the app already calls on :9090:

  profile mapping
    whisperx -> Deployment/whisper       (:8003)
    large    -> Deployment/vllm-gptoss    (:8000, gpt-oss-120b)  [default LLM]
    small    -> Deployment/vllm-granite   (:8001, granite)
    fp8      -> Deployment/vllm-granite   (:8001, alias; no separate fp8 deploy)

Only ONE large LLM fits the 119 GB unified memory, so activating a vLLM profile
scales the other large deployment to 0. whisper (~6 GB) coexists with a running
LLM when memory allows (checked via /proc/meminfo), else the LLM is scaled down.

Runs as a k8s Deployment (hostNetwork, :9090) with a ServiceAccount allowed to
scale deployments in `spark-ai`.
"""

import asyncio
import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from kubernetes import client, config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpu-manager")

NAMESPACE = os.getenv("SPARK_NS", "spark-ai")
config.load_incluster_config()
_apps = client.AppsV1Api()

WHISPER = {"deploy": "whisper", "health": "http://localhost:8003/health", "port": 8003}
VLLM = {
    "large": {"deploy": "vllm-gptoss",  "health": "http://localhost:8000/health", "port": 8000},
    "small": {"deploy": "vllm-granite", "health": "http://localhost:8001/health", "port": 8001},
    "fp8":   {"deploy": "vllm-granite", "health": "http://localhost:8001/health", "port": 8001},
}

MEMORY_SAFETY_MARGIN_GB = 5.0
_MEM_EST_GB = {"whisperx": 6.0, "vllm-large": 83.0, "vllm-small": 88.0, "vllm-fp8": 88.0}


def _mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / (1024 * 1024), 2)
    except Exception as exc:
        logger.warning("meminfo read failed: %s", exc)
    return 0.0


def _running(deploy: str) -> bool:
    try:
        d = _apps.read_namespaced_deployment(deploy, NAMESPACE)
        return bool(d.status.ready_replicas and d.status.ready_replicas > 0)
    except client.ApiException as exc:
        if exc.status == 404:
            return False
        raise


def _scale(deploy: str, n: int) -> str:
    try:
        _apps.patch_namespaced_deployment_scale(deploy, NAMESPACE, {"spec": {"replicas": n}})
        logger.info("scaled %s -> %d", deploy, n)
        return f"scaled_{n}"
    except client.ApiException as exc:
        if exc.status == 404:
            raise HTTPException(404, f"Deployment {deploy} not found")
        raise


async def _wait_healthy(url: str, timeout: int) -> bool:
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=8.0) as http:
        while time.monotonic() - start < timeout:
            try:
                if (await http.get(url)).status_code == 200:
                    logger.info("healthy: %s (%.0fs)", url, time.monotonic() - start)
                    return True
            except Exception:
                pass
            await asyncio.sleep(3)
    logger.warning("not healthy within %ds: %s", timeout, url)
    return False


def _can_coexist(mem_key: str) -> bool:
    avail = _mem_avail_gb()
    needed = _MEM_EST_GB.get(mem_key, 0.0) + MEMORY_SAFETY_MARGIN_GB
    ok = avail >= needed
    logger.info("coexist %s: avail=%.1f needed=%.1f -> %s", mem_key, avail, needed, ok)
    return ok


app = FastAPI(title="GPU Manager (k8s)", version="2.0")


@app.get("/status")
async def status():
    large = _running(VLLM["large"]["deploy"])
    small = _running(VLLM["small"]["deploy"])
    wx = _running(WHISPER["deploy"])
    return {
        "whisperx": wx,
        "vllm": large or small,
        "vllm_large": large,
        "vllm_small": small,
        "active_vllm_profile": "large" if large else ("small" if small else None),
        "memory_available_gb": _mem_avail_gb(),
        "coexistence": wx and (large or small),
    }


@app.post("/gpu/whisperx")
async def activate_whisperx():
    any_vllm = _running(VLLM["large"]["deploy"]) or _running(VLLM["small"]["deploy"])
    if any_vllm and _can_coexist("whisperx"):
        vllm_action = "kept_running"
    else:
        vllm_action = {}
        for p in ("large", "small"):
            d = VLLM[p]["deploy"]
            if _running(d):
                vllm_action[d] = _scale(d, 0)
        if vllm_action:
            await asyncio.sleep(3)
    _scale(WHISPER["deploy"], 1)
    healthy = await _wait_healthy(WHISPER["health"], timeout=180)
    if not healthy:
        raise HTTPException(503, "whisperx failed to become healthy")
    return {"active": "whisperx", "vllm_action": vllm_action, "healthy": healthy,
            "memory_available_gb": _mem_avail_gb()}


@app.post("/gpu/vllm")
async def activate_vllm():
    return await _activate("small")


@app.post("/gpu/vllm-small")
async def activate_vllm_small():
    return await _activate("small")


@app.post("/gpu/vllm-fp8")
async def activate_vllm_fp8():
    return await _activate("fp8")


@app.post("/gpu/vllm-bf16")
async def activate_vllm_bf16():
    return await _activate("small")


@app.post("/gpu/vllm/{profile}")
async def activate_vllm_by_profile(profile: str):
    if profile not in VLLM:
        raise HTTPException(400, f"Unknown profile '{profile}'. Use 'large', 'small', or 'fp8'.")
    return await _activate(profile)


async def _activate(profile: str):
    info = VLLM[profile]
    target = info["deploy"]
    mem_key = f"vllm-{profile}"

    # Scale down any OTHER large vLLM deployment (mutual exclusion on GPU memory).
    for p, i in VLLM.items():
        if i["deploy"] != target and _running(i["deploy"]):
            _scale(i["deploy"], 0)
            await asyncio.sleep(3)

    wx = _running(WHISPER["deploy"])
    if wx and _can_coexist(mem_key):
        wx_action = "kept_running"
    else:
        wx_action = _scale(WHISPER["deploy"], 0) if wx else "not_running"
        if wx:
            await asyncio.sleep(3)

    _scale(target, 1)
    timeout = 300 if profile == "fp8" else (480 if profile == "small" else 600)
    healthy = await _wait_healthy(info["health"], timeout=timeout)
    if not healthy:
        raise HTTPException(503, f"vllm-{profile} ({target}) failed to become healthy")

    return {"active": f"vllm-{profile}", "profile": profile, "deployment": target,
            "port": info["port"], "whisperx_action": wx_action, "healthy": healthy,
            "memory_available_gb": _mem_avail_gb()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
