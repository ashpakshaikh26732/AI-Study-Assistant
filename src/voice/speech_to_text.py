"""Speech-to-text with Whisper (CPU or GPU, no ffmpeg needed for WAV input)."""
from __future__ import annotations

import io
import wave

import numpy as np

WHISPER_RATE = 16000


def load_whisper_model(config: dict):
    """Load the Whisper ASR pipeline on the GPU if there is one, else the CPU.

    Load it lazily (first time the mic is used) - it isn't needed for typed chat.
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=config["voice"]["whisper_model"],
        chunk_length_s=30,
        device=device,
    )


def wav_bytes_to_array(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit PCM WAV bytes to a mono float32 array in [-1, 1] plus its sample rate."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"Unsupported WAV sample width: {width * 8}-bit (need 16-bit)")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def resample(samples: np.ndarray, rate: int, target: int = WHISPER_RATE) -> np.ndarray:
    """Linear-interpolation resample (plenty for speech)."""
    if rate == target or len(samples) == 0:
        return samples
    new_len = int(round(len(samples) * target / rate))
    old_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(new_x, old_x, samples).astype(np.float32)


def transcribe_audio(audio_bytes: bytes, asr_pipeline) -> str:
    """Transcribe recorded audio to text; returns "" if nothing was recognised.

    WAV input (what the mic widget produces with ``format="wav"``) is decoded
    in-process. Other formats are handed to the pipeline as-is, which needs ffmpeg.
    """
    try:
        samples, rate = wav_bytes_to_array(audio_bytes)
        audio = {"raw": resample(samples, rate), "sampling_rate": WHISPER_RATE}
    except (wave.Error, ValueError, EOFError):
        audio = audio_bytes
    result = asr_pipeline(audio)
    return (result.get("text", "") if isinstance(result, dict) else "").strip()
