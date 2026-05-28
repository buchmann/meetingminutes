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
    model_config = {"env_prefix": "TRANSKRIPTOR_", "env_file": ".env", "env_file_encoding": "utf-8"}

    data_dir: Path = Field(default=Path("data"))

    whisper_model: str = "large-v3"
    whisper_compute_type: str = "auto"
    whisper_engine: str = "auto"

    language: Literal["en", "de", "auto"] = "auto"

    diarization_enabled: bool = True
    hf_token: str = ""
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarization_device: str = Field(default_factory=_default_diarization_device)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:70b"
    summarization_enabled: bool = True
    summary_language: str = "auto"

    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_size_mb: int = 500

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "transkriptor.db"

    def ensure_dirs(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
