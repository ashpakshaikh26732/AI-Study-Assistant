import io
import wave

import numpy as np

from src.voice.speech_to_text import resample, transcribe_audio, wav_bytes_to_array
from src.voice.text_to_speech import _speakable, convert_text_to_speech


def make_wav(samples: np.ndarray, rate: int = 44100, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((samples * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


def test_wav_decoding_mono_and_stereo():
    tone = np.sin(np.linspace(0, 20, 4410)).astype(np.float32)
    mono, rate = wav_bytes_to_array(make_wav(tone))
    assert rate == 44100 and len(mono) == 4410 and np.allclose(mono, tone, atol=1e-3)
    stereo, _ = wav_bytes_to_array(make_wav(np.repeat(tone, 2), channels=2))
    assert len(stereo) == 4410


def test_resample_to_whisper_rate():
    assert len(resample(np.zeros(44100, dtype=np.float32), 44100)) == 16000
    same = np.ones(10, dtype=np.float32)
    assert resample(same, 16000) is same


def test_transcribe_passes_16khz_array_to_the_model():
    seen = {}

    def fake_asr(audio):
        seen.update(audio)
        return {"text": "  what is a gru  "}

    text = transcribe_audio(make_wav(np.zeros(44100, dtype=np.float32)), fake_asr)
    assert text == "what is a gru" and seen["sampling_rate"] == 16000 and len(seen["raw"]) == 16000


def test_non_wav_input_is_forwarded_untouched():
    received = []
    transcribe_audio(b"webm-bytes", lambda audio: received.append(audio) or {"text": "x"})
    assert received == [b"webm-bytes"]


def test_tts_text_is_cleaned_for_speech():
    assert _speakable("**GRU** has *two* gates [1][2].\n\n# Heading") == "GRU has two gates . Heading"


def test_tts_handles_empty_and_failure(monkeypatch):
    assert convert_text_to_speech("") is None
    import src.voice.text_to_speech as tts

    def boom(_text):
        raise ConnectionError("offline")

    monkeypatch.setattr(tts, "_synthesize", boom)
    assert convert_text_to_speech("hello there") is None  # degrades gracefully instead of crashing
