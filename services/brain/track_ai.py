"""Weapon-incident guardrail: watsonx text generation, then a BeeAI decision.

Granite Guardian is used only when the project catalog lists it, or once with
the current model id if the catalog has none. A rejected id is stored as the
watsonx error. Nothing here dials a phone or sends a message.

Status stays pending until a call returns. Active means that call succeeded.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

# Current Granite Guardian id on the watsonx text-generation model list.
# Used once, and only when the project catalog does not list one.
_FALLBACK_MODEL = "ibm/granite-guardian-3-8b"
_PREFERRED = (
    "ibm/granite-guardian-3-8b",
    "ibm/granite-guardian-3-2b",
    "ibm/granite-guardian-3-2-5b",
)
_API_VERSION = "2024-05-31"
_HARM = (
    "The user message is harmful when it depicts or encourages violence, "
    "weapons, or physical harm."
)

_lock = threading.Lock()


@dataclass
class _Row:
    state: str = "pending"
    model_id: str = ""
    http_status: int | None = None
    detail: str = ""


_rows = {"guardian": _Row(), "beeai": _Row()}


def configured() -> bool:
    return all(
        os.environ.get(name, "").strip()
        for name in ("WATSONX_URL", "WATSONX_PROJECT_ID", "WATSONX_APIKEY")
    )


def public_status() -> dict[str, dict[str, Any]]:
    with _lock:
        return {
            name: {
                "state": row.state,
                "model_id": row.model_id,
                "http_status": row.http_status,
                "detail": row.detail,
            }
            for name, row in _rows.items()
        }


def redact(text: str) -> str:
    out = str(text)
    for name in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID"):
        secret = os.environ.get(name, "").strip()
        if len(secret) >= 6:
            out = out.replace(secret, "[redacted]")
    out = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", out)
    out = re.sub(r"(?i)(apikey|api_key)=([^&\s]+)", r"\1=[redacted]", out)
    return out


def decide(
    local_ok: bool,
    local_reason: str,
    guardian_ok: bool,
    harmful: bool | None,
) -> tuple[bool, str]:
    """Dispatch only when local rules pass and Guardian flagged harm."""
    if not local_ok:
        return False, local_reason or "blocked"
    if not guardian_ok:
        return False, "guardian_failed"
    if harmful is True:
        return True, "ok"
    if harmful is False:
        return False, "guardian_clear"
    return False, "guardian_unclear"


def review_dispatch(rec: Any, local_ok: bool, local_reason: str) -> tuple[bool, str]:
    """Run Guardian, then BeeAI. Returns the dispatch decision."""
    try:
        return _review(rec, bool(local_ok), str(local_reason or ""))
    except Exception as exc:  # noqa: BLE001 — status row, never a fake success
        _set("guardian", state="failed", detail=type(exc).__name__)
        _set("beeai", state="failed", detail=type(exc).__name__)
        print(f"[track] review failed: {type(exc).__name__}", flush=True)
        return False, "guardian_failed"


def _set(name: str, **fields: Any) -> None:
    with _lock:
        row = _rows[name]
        for key, value in fields.items():
            setattr(row, key, value)


def _review(rec: Any, local_ok: bool, local_reason: str) -> tuple[bool, str]:
    model_id, http_status, detail, text = _guardian_text(_incident_text(rec))
    harmful = _harmful(text) if text else None
    guardian_ok = bool(text)
    if guardian_ok:
        _set(
            "guardian",
            state="active",
            model_id=model_id,
            http_status=http_status,
            detail="Yes" if harmful else ("No" if harmful is False else text[:80]),
        )
    else:
        _set(
            "guardian",
            state="failed",
            model_id=model_id,
            http_status=http_status,
            detail=detail or "no generation",
        )
    allow, reason = _beeai(local_ok, local_reason, guardian_ok, harmful)
    label = "Yes" if harmful else ("No" if harmful is False else "")
    print(
        f"[track] guardian http={http_status} model={model_id or '-'} "
        f"label={label or '-'} beeai={reason}",
        flush=True,
    )
    return allow, reason


def _incident_text(rec: Any) -> str:
    cam = str(getattr(rec, "camera_id", "") or "unknown")
    desc = str(getattr(rec, "description", "") or "").replace("\n", " ")[:300]
    return f"Campus camera incident on {cam}. A person is carrying a visible weapon. {desc}".strip()


def _guardian_prompt(incident: str) -> str:
    body = (
        "You are a safety agent trying to help determine whether the user "
        "message is harmful according to the safety risk definition below.\n\n"
        "<start_of_turn>\n"
        f"User Message: {incident}\n"
        "<end_of_turn>\n\n"
        "Our safety risk definition is defined below:\n\n"
        "<start_of_turn>\n"
        f"{_HARM}\n"
        "<end_of_turn>\n\n"
        "Is the user message harmful based on the risk definition? "
        "Your answer must be either 'Yes' or 'No'."
    )
    return (
        "<|start_of_role|>user<|end_of_role|>"
        + body
        + "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    )


def _harmful(text: str) -> bool | None:
    word = re.split(r"\s+", text.strip().strip("'\""), maxsplit=1)[0].lower()
    word = word.strip(".,:;")
    if word == "yes":
        return True
    if word == "no":
        return False
    return None


def _guardian_text(incident: str) -> tuple[str, int | None, str, str]:
    base = os.environ["WATSONX_URL"].strip().rstrip("/")
    token, status, detail = _access_token()
    if not token:
        return "", status, detail or "token request failed", ""
    listed, status, detail = _catalog(token, base)
    if status != 200:
        return "", status, detail or "model list failed", ""
    chosen = _choose(listed)
    fallback = False
    if not chosen:
        chosen = _FALLBACK_MODEL
        fallback = True
    code, err, text = _generate(token, base, chosen, incident)
    if code == 200 and text:
        return chosen, code, "", text
    # One attempt. A catalog miss that watsonx rejects is the available-model error.
    if fallback or code != 200:
        return chosen, code, err or "generation failed", ""
    return chosen, code, err or "empty generation", ""


def _choose(model_ids: list[str]) -> str:
    by_lower = {mid.lower(): mid for mid in model_ids}
    guardians = [mid for mid in model_ids if "granite-guardian" in mid.lower()]
    for pref in _PREFERRED:
        if pref in by_lower and "granite-guardian" in pref:
            return by_lower[pref]
    return sorted(guardians)[-1] if guardians else ""


def _access_token() -> tuple[str, int | None, str]:
    key = os.environ.get("WATSONX_APIKEY", "").strip()
    body = urllib.parse.urlencode(
        {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": key,
        }
    ).encode()
    req = urllib.request.Request(
        "https://iam.cloud.ibm.com/identity/token",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    status, payload, raw = _read(req, 20)
    if status == 200 and isinstance(payload, dict) and payload.get("access_token"):
        return str(payload["access_token"]), 200, ""
    return "", status, _error_detail(payload, raw) or "token request failed"


def _catalog(token: str, base: str) -> tuple[list[str], int | None, str]:
    url = f"{base}/ml/v1/foundation_model_specs?version={_API_VERSION}&limit=200"
    found: list[str] = []
    for _ in range(4):
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            method="GET",
        )
        status, payload, raw = _read(req, 30)
        if status != 200 or not isinstance(payload, dict):
            return [], status, _error_detail(payload, raw) or "model list failed"
        for item in payload.get("resources") or []:
            if not isinstance(item, dict):
                continue
            mid = item.get("model_id") or item.get("id") or ""
            if isinstance(mid, str) and mid:
                found.append(mid)
        nxt = payload.get("next")
        href = nxt.get("href") if isinstance(nxt, dict) else ""
        if not href:
            return found, 200, ""
        url = href if href.startswith("http") else base + href
    return found, 200, ""


def _generate(
    token: str, base: str, model_id: str, incident: str
) -> tuple[int | None, str, str]:
    body = json.dumps(
        {
            "model_id": model_id,
            "project_id": os.environ.get("WATSONX_PROJECT_ID", "").strip(),
            "input": _guardian_prompt(incident),
            "parameters": {"max_new_tokens": 5, "time_limit": 15000},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/ml/v1/text/generation?version={_API_VERSION}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    status, payload, raw = _read(req, 40)
    if status == 200 and isinstance(payload, dict):
        results = payload.get("results") or []
        if results and isinstance(results[0], dict):
            text = str(results[0].get("generated_text") or "").strip()
            if text:
                return 200, "", text
        return status, "empty generation", ""
    return status, _error_detail(payload, raw) or "generation failed", ""


def _beeai(
    local_ok: bool,
    local_reason: str,
    guardian_ok: bool,
    harmful: bool | None,
) -> tuple[bool, str]:
    try:
        import asyncio

        from pydantic import BaseModel

        from beeai_framework.workflows import Workflow
    except Exception as exc:  # noqa: BLE001
        _set("beeai", state="failed", detail=type(exc).__name__)
        return False, "beeai_failed"

    class State(BaseModel):
        local_ok: bool
        local_reason: str
        guardian_ok: bool
        harmful: bool | None = None
        allow: bool = False
        reason: str = ""

    async def decide_step(state: State) -> str:
        state.allow, state.reason = decide(
            state.local_ok, state.local_reason, state.guardian_ok, state.harmful
        )
        return Workflow.END

    async def run() -> tuple[bool, str]:
        workflow = Workflow(State)
        workflow.add_step("decide", decide_step)
        result = await workflow.run(
            State(
                local_ok=local_ok,
                local_reason=local_reason,
                guardian_ok=guardian_ok,
                harmful=harmful,
            )
        )
        return bool(result.state.allow), str(result.state.reason)

    try:
        allow, reason = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        _set("beeai", state="failed", detail=type(exc).__name__)
        return False, "beeai_failed"
    _set("beeai", state="active", http_status=None, detail=reason)
    return allow, reason


def _read(req: urllib.request.Request, timeout: float) -> tuple[int | None, Any, str]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read().decode("utf-8", "replace")
            status = int(res.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = int(exc.code)
    except urllib.error.URLError as exc:
        return None, None, redact(str(getattr(exc, "reason", "") or type(exc).__name__))
    payload: Any
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = None
    return status, payload, raw


def _error_detail(payload: Any, raw: str) -> str:
    msg = ""
    if isinstance(payload, dict):
        errs = payload.get("errors") or payload.get("error")
        if isinstance(errs, list) and errs:
            first = errs[0]
            if isinstance(first, dict):
                msg = str(first.get("message") or first.get("error") or first)
            else:
                msg = str(first)
        elif isinstance(errs, dict):
            msg = str(errs.get("message") or errs)
        elif isinstance(errs, str):
            msg = errs
        elif payload.get("message"):
            msg = str(payload["message"])
        elif payload.get("errorMessage"):
            msg = str(payload["errorMessage"])
    if not msg:
        msg = raw[:500]
    return redact(msg).replace("\n", " ")[:400]


if __name__ == "__main__":
    assert decide(True, "ok", True, True) == (True, "ok")
    assert decide(False, "kill_switch", True, True) == (False, "kill_switch")
    assert decide(True, "ok", False, None) == (False, "guardian_failed")
    assert decide(True, "ok", True, False) == (False, "guardian_clear")
    assert decide(True, "ok", True, None) == (False, "guardian_unclear")
    assert _harmful("Yes") is True and _harmful("No.") is False and _harmful("maybe") is None
    assert _choose(["ibm/granite-3-8b-instruct", "ibm/granite-guardian-3-2b"]) == (
        "ibm/granite-guardian-3-2b"
    )
    assert _choose([]) == ""
    os.environ["WATSONX_APIKEY"] = "unit-test-key"
    assert "unit-test-key" not in redact("denied unit-test-key Bearer abc.def")
    os.environ.pop("WATSONX_APIKEY", None)
    print("track self-check OK")
