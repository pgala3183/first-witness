"""Call agent: the fine-tuned model on Fireworks answers the dispatcher.

answer(facts, history, question) -> str | None. None means "use the keyword
rules": not configured, timeout, error or an empty reply.

SYSTEM and user_msg() are exactly what the model was trained on;
finetune/build_dataset.py imports them from here, so training and live
answers can't drift apart.

Two backends (CS_CALL_MODEL_BACKEND):
* chat (default): an OpenAI-style chat endpoint, e.g. a Fireworks deployment.
  Env: CS_CALL_MODEL_URL, CS_CALL_MODEL, CS_CALL_MODEL_REASONING_EFFORT.
* training-pool: the LoRA checkpoint on Fireworks' serverless training pool,
  which needs no deployment. Prompts are rendered with the model's chat
  template (chat_template.py, as in training) and sent as token ids.
  Env: CS_CALL_MODEL (base model id), CS_CALL_MODEL_SNAPSHOT (checkpoint path
  from finetune/results/serverless_run.json).
Both: CS_CALL_MODEL_TIMEOUT_S (default 1.5) and FIREWORKS_API_KEY.
Self-check: python -m services.voice.call_model
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from services import activity
from services.usage import add as usage_add

from .chat_template import END

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


def backend() -> str:
    return os.environ.get("CS_CALL_MODEL_BACKEND", "").strip().lower() or "chat"


def _snapshot() -> str:
    return os.environ.get("CS_CALL_MODEL_SNAPSHOT", "").strip()


def configured() -> bool:
    url, model, key, _ = _config()
    if backend() == "training-pool":
        return bool(model and key and _snapshot() and "<" not in _snapshot())
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
    if backend() == "training-pool":
        return _pool_answer(facts, history, question)
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


# ---------------------------------------------------------------- training pool

_pool: dict[str, Any] = {}
_pool_lock = threading.Lock()
# Requests run here so a slow one can be abandoned at the timeout; it finishes unseen.
_pool_exec = ThreadPoolExecutor(max_workers=4, thread_name_prefix="callmodel")


def _pool_client() -> tuple[Any, Any, Any]:
    """(chat template, sampler, sdk) for the configured checkpoint, built once."""
    _, model, key, _ = _config()
    ident = (model, _snapshot())
    with _pool_lock:
        if _pool.get("ident") != ident:
            from fireworks.training import sdk

            from .chat_template import chat_for, connect

            chat = chat_for(model)
            sampler = connect(key).create_sampling_client(model_path=_snapshot())
            _pool.update(ident=ident, chat=chat, sampler=sampler, sdk=sdk)
        return _pool["chat"], _pool["sampler"], _pool["sdk"]


def _pool_sample(messages: list[dict[str, str]], max_tokens: int) -> tuple[str, int, int]:
    chat, sampler, sdk = _pool_client()
    prompt = chat.prompt_ids(messages)
    params = sdk.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=[END])
    res = sampler.sample(prompt=sdk.ModelInput.from_ints(prompt), num_samples=1, sampling_params=params)
    res = res.result() if hasattr(res, "result") else res
    out = [int(t) for t in res.sequences[0].tokens]
    return chat.decode(out), len(prompt), len(out)


def _pool_answer(facts: dict[str, Any], history: list[tuple[str, str]], question: str) -> str | None:
    _, model, _, timeout = _config()
    uid = usage_id(model) + "-ft"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg(facts, history, question)}]
    t0 = time.perf_counter()
    try:
        with activity.busy("callmodel"):
            text, tokens_in, tokens_out = _pool_exec.submit(_pool_sample, messages, MAX_TOKENS).result(timeout)
    except FutureTimeout:
        usage_add(uid, requests=1, errors=1)
        return None
    except Exception:  # noqa: BLE001  (network, auth, missing SDK: the keyword rules answer)
        usage_add(uid, requests=1, errors=1)
        return None
    usage_add(uid, requests=1, live=1, tokens_in=tokens_in, tokens_out=tokens_out,
              ms=(time.perf_counter() - t0) * 1000)
    return clean(text) or None


def warmup() -> bool:
    """Build the sampler and answer once, so the first live question isn't a cold start.

    Call it off the event loop when a live call starts; never raises."""
    if not configured() or backend() != "training-pool":
        return False
    try:
        _pool_sample([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user_msg({}, [], "Who is this?")}], 8)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    from services.usage import swap

    assert clean("<think>hmm</think> I can't verify that.") == "I can't verify that."
    assert clean("<think>never closed") == ""
    assert usage_id("accounts/me/models/first-witness-call-4b") == "first-witness-call-4b"
    msg = user_msg({"a": 1}, [("Dispatcher", "Who is this?"), (SPEAKER, "First Witness.")], "Say again?")
    assert msg == ('FACTS: {"a":1}\nCALL SO FAR:\nDispatcher: Who is this?\n'
                   "First Witness: First Witness.\nDISPATCHER: Say again?")
    assert user_msg({}, [], "Hi").endswith("(start of call)\nDISPATCHER: Hi")
    keys = ("CS_CALL_MODEL_URL", "CS_CALL_MODEL", "FIREWORKS_API_KEY", "CS_CALL_MODEL_BACKEND",
            "CS_CALL_MODEL_SNAPSHOT", "CS_CALL_MODEL_TIMEOUT_S")
    saved = {k: os.environ.pop(k, None) for k in keys}
    assert not configured() and answer({}, [], "Hi") is None
    # Unreachable server: falls back (None) quickly and counts an error.
    os.environ.update(CS_CALL_MODEL_URL="http://127.0.0.1:9/v1", CS_CALL_MODEL="m", FIREWORKS_API_KEY="k",
                      CS_CALL_MODEL_TIMEOUT_S="0.5")
    swap()
    assert answer({}, [], "Hi") is None
    assert swap() == {"m": {"requests": 1, "errors": 1}}

    # Training pool (offline): needs a snapshot; slow or failing answers fall back.
    os.environ.update(CS_CALL_MODEL_BACKEND="training-pool", CS_CALL_MODEL="accounts/fireworks/models/b")
    assert not configured() and not warmup()
    os.environ["CS_CALL_MODEL_SNAPSHOT"] = "acct/run-1/fw-ep2"
    assert configured()
    real_sample = _pool_sample

    def slow(_m: Any, _n: int) -> tuple[str, int, int]:
        time.sleep(1.0)
        return "late", 1, 1

    def broken(_m: Any, _n: int) -> tuple[str, int, int]:
        raise RuntimeError("pool down")

    def good(messages: Any, _n: int) -> tuple[str, int, int]:
        assert messages[0]["content"] == SYSTEM and messages[1]["content"].endswith("DISPATCHER: Hi")
        return "No officer has accepted yet.", 500, 7

    swap()
    for fake, want in ((slow, None), (broken, None), (good, "No officer has accepted yet.")):
        globals()["_pool_sample"] = fake
        t = time.perf_counter()
        assert answer({}, [], "Hi") == want, fake.__name__
        assert time.perf_counter() - t < 0.9, "the timeout must cut a slow answer off"
    use = swap()["b-ft"]
    assert use["requests"] == 3 and use["errors"] == 2 and use["tokens_in"] == 500, use
    globals()["_pool_sample"] = real_sample
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v
    if configured():
        print("warmup:", warmup())
        t = time.perf_counter()
        print("live:", answer({"site": "test"}, [], "Who is this?"), f"{(time.perf_counter() - t) * 1000:.0f} ms")
    print("call_model self-check OK")
