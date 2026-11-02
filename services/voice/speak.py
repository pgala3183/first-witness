"""Speak call lines out loud in the console with ElevenLabs (no phone involved).

GET /voice/say?who=fw|dispatcher&text=... returns MP3 for the browser to play.
The API key stays on the server (ELEVENLABS_API_KEY in .env); the browser
never sees it.

Voices (set in .env):
  ELEVENLABS_VOICE_ID             First Witness: calm, clear, measured
  ELEVENLABS_DISPATCHER_VOICE_ID  the simulated campus police dispatcher
If the dispatcher voice is unset, only First Witness lines are spoken.

Repeated lines are cached in memory, so replays cost no extra characters.
Stdlib only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import OrderedDict

MAX_CHARS = 400
_CACHE: "OrderedDict[tuple[str, str], bytes]" = OrderedDict()
_CACHE_MAX = 200

# Persona: First Witness speaks steadily (high stability, no added style);
# the dispatcher sounds brisker and more natural (lower stability).
PERSONAS = {
    "fw": {"env": ("ELEVENLABS_VOICE_ID", "CS_ELEVENLABS_VOICE_ID"),
           "settings": {"stability": 0.75, "similarity_boost": 0.8, "style": 0.0, "use_speaker_boost": True}},
    "dispatcher": {"env": ("ELEVENLABS_DISPATCHER_VOICE_ID",),
                   "settings": {"stability": 0.45, "similarity_boost": 0.75, "style": 0.15, "use_speaker_boost": True}},
}


def _voice(who: str) -> str:
    for name in PERSONAS[who]["env"]:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return ""


def status() -> dict:
    key = bool(os.environ.get("ELEVENLABS_API_KEY", "").strip())
    return {"ready": key and bool(_voice("fw")), "dispatcher": key and bool(_voice("dispatcher"))}


def say(text: str, who: str = "fw") -> tuple[int, bytes, str]:
    """(http status, body, content type). Never raises."""
    who = who if who in PERSONAS else "fw"
    text = " ".join(str(text or "").split())[:MAX_CHARS]
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice = _voice(who)
    if not text:
        return 400, b'{"error":"text is required"}', "application/json"
    if not key or not voice:
        return 503, b'{"error":"ElevenLabs key or voice not set"}', "application/json"
    hit = _CACHE.get((who, text))
    if hit is not None:
        _CACHE.move_to_end((who, text))
        return 200, hit, "audio/mpeg"
    model = os.environ.get("CS_ELEVENLABS_MODEL", "").strip() or "eleven_flash_v2_5"
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
        data=json.dumps({"text": text, "model_id": model, "voice_settings": PERSONAS[who]["settings"]}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            audio = r.read()
    except urllib.error.HTTPError as e:
        return 502, json.dumps({"error": f"elevenlabs http {e.code}"}).encode(), "application/json"
    except Exception as e:  # noqa: BLE001 - network: report, don't crash the API
        return 502, json.dumps({"error": type(e).__name__}).encode(), "application/json"
    if not audio:
        return 502, b'{"error":"empty audio"}', "application/json"
    _CACHE[(who, text)] = audio
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return 200, audio, "audio/mpeg"
