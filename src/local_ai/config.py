import platform
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


def _default_diarization_device() -> str:
    """Auto-detect best device for pyannote diarization.

    On macOS, use CPU to avoid MPS memory contention with mlx-whisper.
    On Linux/Windows with CUDA, use the GPU.
    """
    system = platform.system()
    if system == "Darwin":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Settings(BaseSettings):
    model_config = {"env_prefix": "LOCAL_AI_", "env_file": ".env", "env_file_encoding": "utf-8"}

    data_dir: Path = Field(default=Path("data"))

    # Transcription backend: "local" or "remote"
    transcription_backend: str = "local"

    # Remote whisperx API (e.g. DGX Spark with whisperx-blackwell)
    # Handles transcription + diarization in one call
    whisperx_url: str = "http://192.168.178.190:8003"

    # Local transcription settings
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "auto"
    whisper_engine: str = "auto"

    language: Literal["en", "de", "auto"] = "auto"

    diarization_enabled: bool = True
    hf_token: str = ""
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarization_device: str = Field(default_factory=_default_diarization_device)

    # Summarization backend: "ollama" or "openai"
    summary_backend: str = "ollama"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:70b"

    # OpenAI-compatible API settings (e.g. DGX Spark, vLLM, etc.)
    openai_base_url: str = "http://192.168.178.190:8001/v1"
    openai_api_key: str = "none"
    openai_model: str = "ibm/granite-3-3-8b-instruct"

    # Embedding model (vLLM on the DGX, OpenAI-compatible /v1/embeddings).
    # Used for semantic search in Notes & Manuals. Empty → keyword-only fallback.
    embedding_base_url: str = "http://192.168.178.190:8002/v1"
    embedding_model: str = "bge-m3"

    summarization_enabled: bool = True
    summary_language: str = "auto"

    # Web search via self-hosted SearXNG (ad-free metasearch, JSON API).
    # In-cluster service; empty disables the Web Search feature.
    searxng_url: str = "http://searxng:8080"

    # Fernet key (urlsafe-base64, 32 bytes) used to encrypt stored email
    # account passwords at rest. From the LOCAL_AI_EMAIL_ENC_KEY secret.
    # Empty disables the Email Digest feature (no place to store creds safely).
    email_enc_key: str = ""

    # GPU manager (DGX Spark: swaps whisperx ↔ vLLM since both can't run simultaneously)
    gpu_manager_url: str = ""  # e.g. "http://192.168.178.190:9090"
    # vLLM profile for GPU manager: "large" (120B), "small" (granite 8B), or "auto" (detect from model/port)
    vllm_profile: str = "auto"
    # GPU-manager endpoint for the ACTIVE model — set at runtime by apply_llm().
    vllm_gpu_endpoint: str = "vllm-small"
    # Active LLM key (one of LLM_MODELS). Persisted in DB; applied at startup.
    active_llm: str = "gptoss"

    # OpenTelemetry tracing
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "local-ai"

    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_size_mb: int = 500

    # ── Multi-user / authentication ──────────────────────────────────
    # Seed admin created on first startup when no users exist yet.
    admin_username: str = "admin"
    admin_password: str = ""  # set in .env to enable seeding; blank = skip seeding
    # Session cookie lifetime in hours (default 30 days).
    session_ttl_hours: int = 720
    # Mark the session cookie Secure (enable when served over HTTPS).
    session_cookie_secure: bool = False

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "local_ai.db"

    @property
    def style_profile_path(self) -> Path:
        return self.data_dir / "style_profile.txt"

    def ensure_dirs(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ── Switchable LLMs (DGX Spark) ────────────────────────────────────────
# The active model is chosen in Settings, persisted in DB, and applied to the
# Settings object at startup / on switch via apply_llm(). All LLM features read
# settings.openai_base_url / openai_model, so switching here switches everything.
# Only ONE can be GPU-resident at a time (128GB unified memory) — switching
# triggers a GPU-manager swap (gpu_endpoint) and a ~3-5 min model reload.
LLM_MODELS: dict[str, dict] = {
    "gptoss": {
        "label": "GPT-OSS 120B",
        "detail": "CUTLASS MXFP4 · ~60 tok/s · strongest analysis & German",
        "base_url": "http://192.168.178.190:8000/v1",
        "model": "gpt-oss-120b",
        "gpu_endpoint": "vllm/large",
    },
    "granite": {
        "label": "Granite 4.0-H-Small",
        "detail": "32K context · ~10 tok/s · lighter / lower power",
        "base_url": "http://192.168.178.190:8001/v1",
        "model": "ibm/granite-3-3-8b-instruct",
        "gpu_endpoint": "vllm-small",
    },
}
DEFAULT_LLM = "gptoss"


def apply_llm(settings: "Settings", key: str) -> dict:
    """Point the Settings object at the chosen model. Returns the model dict."""
    model = LLM_MODELS.get(key) or LLM_MODELS[DEFAULT_LLM]
    settings.active_llm = key if key in LLM_MODELS else DEFAULT_LLM
    settings.openai_base_url = model["base_url"]
    settings.openai_model = model["model"]
    settings.vllm_gpu_endpoint = model["gpu_endpoint"]
    return model
