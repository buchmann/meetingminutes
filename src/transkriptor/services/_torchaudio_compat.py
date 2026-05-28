"""Compatibility shim for torchaudio 2.9+ and torch 2.6+ where:
- torchaudio legacy API (info, load, AudioMetaData, list_audio_backends) was removed
- torch.load defaults to weights_only=True, breaking pyannote model loading

Call patch() before importing pyannote.audio.
"""

import torchaudio

_patched = False


def patch():
    global _patched
    if _patched:
        return
    _patched = True

    import soundfile as sf
    import torch

    # --- Step 1: Restore torchaudio legacy API (must happen before pyannote import) ---

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample=16, encoding="PCM_S"):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "info"):
        def _info(filepath, backend=None):
            info = sf.info(str(filepath))
            return torchaudio.AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
            )
        torchaudio.info = _info

    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends():
            return ["soundfile"]
        torchaudio.list_audio_backends = _list_audio_backends

    if not hasattr(torchaudio, "_original_load"):
        original = torchaudio.load

        def _load(filepath, *args, **kwargs):
            kwargs.pop("backend", None)
            frame_offset = kwargs.pop("frame_offset", 0)
            num_frames = kwargs.pop("num_frames", -1)
            try:
                return original(filepath, *args, frame_offset=frame_offset,
                                num_frames=num_frames, **kwargs)
            except (ImportError, RuntimeError):
                start = frame_offset
                stop = None if num_frames == -1 else frame_offset + num_frames
                data, sr = sf.read(str(filepath), dtype="float32", always_2d=True,
                                   start=start, stop=stop)
                waveform = torch.from_numpy(data.T)
                return waveform, sr

        torchaudio._original_load = original
        torchaudio.load = _load

    # --- Step 2: Allowlist pyannote globals for torch.load (torch 2.6+ weights_only=True) ---

    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
    except ImportError:
        pass
