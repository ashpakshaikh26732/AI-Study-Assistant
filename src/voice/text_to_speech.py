"""Text-to-speech with gTTS (needs an internet connection)."""
from __future__ import annotations

import functools
import io
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

MAX_SPOKEN_CHARS = 1500  # keep spoken replies short - long ones are slow to synthesise


def _speakable(text: str) -> str:
    """Strip markdown/citation markers so they aren't read out loud."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_SPOKEN_CHARS]


@functools.lru_cache(maxsize=32)
def _synthesize(text: str) -> Optional[bytes]:
    from gtts import gTTS

    buffer = io.BytesIO()
    gTTS(text, lang="en").write_to_fp(buffer)
    return buffer.getvalue()


def convert_text_to_speech(text_to_speak: str) -> Optional[bytes]:
    """MP3 bytes for ``text_to_speak``, or None if empty / offline / gTTS unavailable."""
    text = _speakable(text_to_speak or "")
    if not text:
        return None
    try:
        return _synthesize(text)
    except Exception as exc:  # no network, gTTS missing, rate-limited...
        log.warning("Text-to-speech failed: %s", exc)
        return None
