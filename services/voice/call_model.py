"""Call agent: the fine-tuned Qwen3-4B on Fireworks answers the dispatcher.

answer(facts, history, question) -> str | None. None means "use the keyword
rules": not configured, timeout, HTTP error or an empty reply.

SYSTEM and user_msg() are exactly what the model was trained on;
finetune/build_dataset.py imports them from here, so training and live
answers can't drift apart.

Env: CS_CALL_MODEL_URL, CS_CALL_MODEL, CS_CALL_MODEL_TIMEOUT_S (default 1.5),
CS_CALL_MODEL_REASONING_EFFORT (e.g. none, for a reasoning base model),
and the bearer key in FIREWORKS_API_KEY. Self-check: python -m services.voice.call_model
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from services import activity
from services.usage import add as usage_add

SPEAKER = "First Witness"

SYSTEM = (
    "You are First Witness, an automated campus camera system speaking on a live phone call with a "
    "police or campus-security dispatcher. Answer the dispatcher's last question in one or two short "
    "spoken sentences. Use ONLY the FACTS JSON. Rules: "
    "1) Never state a number, weapon, place, building or name that is not in FACTS. "
    "2) If FACTS can't answer it (injuries, identity, intent, sounds, anything off camera), say you "
    "can't verify it. "
    "3) Never say an officer is coming, responding or nearby unless officer.status is \"accepted\"; then "
    "give their name and ETA. Never say anyone has arrived. "
    "4) Never say a warning was sent unless zone_alert.status is \"sent\". Never say doors are locked. "
    "5) If now.visible is false, say you lost sight of them and use last_seen. "
    "6) No camera ids, JSON or scores. Plain words a dispatcher can repeat."
)

MAX_TOKENS = 80
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)


def dumps(snap: dict[str, Any]) -> str:
    """Compact JSON the model sees (fewer tokens = faster, cheaper)."""
    return json.dumps(snap, separators=(",", ":"), ensure_ascii=False)


def user_msg(snap: dict[str, Any], history: list[tuple[str, str]], question: str) -> str:
    """history: (speaker, text) pairs, speaker "Dispatcher" or SPEAKER."""
    lines = [f"{who}: {text}" for who, text in history] or ["(start of call)"]
    return f"FACTS: {dumps(snap)}\nCALL SO FAR:\n" + "\n".join(lines) + f"\nDISPATCHER: {question}"


def _config() -> tuple[str, str, str, float]:
    url = os.environ.get("CS_CALL_MODEL_URL", "").strip().rstrip("/")
    model = os.environ.get("CS_CALL_MODEL", "").strip()
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    try:
        timeout = float(os.environ.get("CS_CALL_MODEL_TIMEOUT_S", "") or 1.5)
    except ValueError:
        timeout = 1.5
    return url, model, key, timeout


def configured() -> bool:
    url, model, key, _ = _config()
    return bool(url and model and key and "<" not in model)


def usage_id(model: str) -> str:
    """Stable dashboard id: the model's last path segment."""
    return model.rsplit("/", 1)[-1]


def clean(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def answer(
    facts: dict[str, Any], history: list[tuple[str, str]], question: str
) -> str | None:
    if not configured():
        return None
    url, model, key, timeout = _config()
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(facts, history, question)},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        # A reasoning base model would otherwise think through MAX_TOKENS and say nothing.
        **({"reasoning_effort": effort} if (effort := os.environ.get("CS_CALL_MODEL_REASONING_EFFORT", "").strip()) else {}),
    }).encode("utf-8")
    req = urllib.request.Request(
        url + "/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    raw: dict[str, Any] | None = None
    for attempt in range(2):
        try:
            with activity.busy("callmodel"), urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            # A scaled-to-zero deployment answers 503 DEPLOYMENT_SCALING_UP once.
            scaling = e.code == 503 and b"DEPLOYMENT_SCALING_UP" in (e.read() or b"")
            if scaling and attempt == 0:
                time.sleep(0.3)
                continue
            usage_add(usage_id(model), requests=1, errors=1)
            return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            usage_add(usage_id(model), requests=1, errors=1)
            return None
    if not isinstance(raw, dict):
        return None
    use = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    usage_add(
        usage_id(model),
        requests=1,
        live=1,
        tokens_in=int(use.get("prompt_tokens") or 0),
        tokens_out=int(use.get("completion_tokens") or 0),
        ms=(time.perf_counter() - t0) * 1000,
    )
    text = clean(
        str((((raw.get("choices") or [{}])[0].get("message") or {}).get("content")) or "")
    )
    return text or None


if __name__ == "__main__":
    from services.usage import swap

    assert clean("<think>hmm</think> I can't verify that.") == "I can't verify that."
    assert clean("<think>never closed") == ""
    assert usage_id("accounts/me/models/first-witness-call-4b") == "first-witness-call-4b"
    msg = user_msg({"a": 1}, [("Dispatcher", "Who is this?"), (SPEAKER, "First Witness.")], "Say again?")
    assert msg == ('FACTS: {"a":1}\nCALL SO FAR:\nDispatcher: Who is this?\n'
                   "First Witness: First Witness.\nDISPATCHER: Say again?")
    assert user_msg({}, [], "Hi").endswith("(start of call)\nDISPATCHER: Hi")
    saved = {k: os.environ.pop(k, None) for k in ("CS_CALL_MODEL_URL", "CS_CALL_MODEL", "FIREWORKS_API_KEY")}
    assert not configured() and answer({}, [], "Hi") is None
    # Unreachable server: falls back (None) quickly and counts an error.
    os.environ.update(CS_CALL_MODEL_URL="http://127.0.0.1:9/v1", CS_CALL_MODEL="m", FIREWORKS_API_KEY="k",
                      CS_CALL_MODEL_TIMEOUT_S="0.5")
    swap()
    assert answer({}, [], "Hi") is None
    assert swap() == {"m": {"requests": 1, "errors": 1}}
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v
    if configured():
        t = time.perf_counter()
        print("live:", answer({"site": "test"}, [], "Who is this?"), f"{(time.perf_counter() - t) * 1000:.0f} ms")
    print("call_model self-check OK")
