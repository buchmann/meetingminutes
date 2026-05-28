"""Live audio recording via ffmpeg + BlackHole / system audio devices."""

import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def list_audio_devices() -> list[dict]:
    """List available macOS audio input devices via ffmpeg avfoundation."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
        # Device list goes to stderr
        output = result.stderr
        devices = []
        in_audio = False
        for line in output.splitlines():
            if "audio devices:" in line.lower():
                in_audio = True
                continue
            if in_audio and "[" in line and "]" in line:
                # Parse lines like: [AVFoundation indev @ 0x...] [1] BlackHole 16ch
                parts = line.split("] ")
                if len(parts) >= 2:
                    idx_part = parts[-2].split("[")[-1]  # "1"
                    name = parts[-1].strip()
                    try:
                        devices.append({"index": int(idx_part), "name": name})
                    except ValueError:
                        pass
        return devices
    except Exception as e:
        logger.error("Failed to list audio devices: %s", e)
        return []


class Recorder:
    """Manages ffmpeg recording process."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._process: subprocess.Popen | None = None
        self._output_path: Path | None = None
        self._started_at: str | None = None

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def status(self) -> dict:
        return {
            "recording": self.is_recording,
            "started_at": self._started_at,
            "output_path": str(self._output_path) if self._output_path else None,
        }

    def start(self, device_index: int = 0, device_name: str = "") -> dict:
        """Start recording from the specified audio device.

        Args:
            device_index: AVFoundation audio device index (default 0 = Aggregate Device)
            device_name: Optional device name for the filename
        """
        if self.is_recording:
            raise RuntimeError("Already recording")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = device_name.replace(" ", "_").replace("/", "_") if device_name else f"device{device_index}"
        filename = f"recording_{timestamp}_{safe_name}.wav"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self.output_dir / filename

        # Record 16kHz mono WAV — ready for whisper, no preprocessing needed
        cmd = [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-i", f":{device_index}",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            str(self._output_path),
        ]

        logger.info("Starting recording: %s (device :%d)", filename, device_index)
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._started_at = datetime.now().isoformat()

        return {
            "recording": True,
            "filename": filename,
            "device_index": device_index,
            "started_at": self._started_at,
        }

    def stop(self) -> dict:
        """Stop recording and return the output file path."""
        if not self.is_recording:
            raise RuntimeError("Not recording")

        # Send 'q' to ffmpeg to gracefully stop
        logger.info("Stopping recording...")
        try:
            self._process.stdin.write(b"q")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)

        output_path = self._output_path
        started_at = self._started_at

        self._process = None
        self._started_at = None

        if output_path and output_path.exists():
            size = output_path.stat().st_size
            logger.info("Recording saved: %s (%d bytes)", output_path.name, size)
            return {
                "recording": False,
                "file_path": str(output_path),
                "filename": output_path.name,
                "file_size_bytes": size,
                "started_at": started_at,
            }
        else:
            raise RuntimeError("Recording file not found")
