import asyncio
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioInfo:
    wav_path: Path
    duration_secs: float
    sample_rate: int
    channels: int


async def preprocess_audio(input_path: Path, output_dir: Path) -> AudioInfo:
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "audio.wav"

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
        str(wav_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()[-500:]}")

    duration = await _get_duration(wav_path)
    return AudioInfo(wav_path=wav_path, duration_secs=duration, sample_rate=16000, channels=1)


async def _get_duration(path: Path) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    info = json.loads(stdout)
    return float(info["format"]["duration"])
