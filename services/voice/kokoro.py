"""Kokoro TTS client for the local serve (scripts/serve_kokoro.py, CS_KOKORO_URL).

synthesize(text) -> pcm16_mono_16k bytes; b"" when CS_KOKORO_URL is unset.
ElevenLabs is the default call voice; Kokoro is the optional local one.
Target first audio < 500 ms for a short sentence.
"""

from __future__ import annotations

import os
from typing import Protocol


class TtsBackend(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class HttpKokoro:
    def synthesize(self, text: str) -> bytes:
        text = (text or "").strip()
        if not text:
            return b""
        url = os.environ.get("CS_KOKORO_URL", "").strip()
        if not url:
            return b""
        import json
        import urllib.request

        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read()


def get_tts() -> TtsBackend:
    return HttpKokoro()
