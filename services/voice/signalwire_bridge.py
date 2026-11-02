"""SignalWire Compatibility API bridge for the outbound demo call.

Real calls and SMS stay off unless CS_SIGNALWIRE_ENABLED is explicitly set.
Outbound destinations are limited to the one verified demo number.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

_TRUE = {"1", "true", "yes"}
_EMERGENCY_DIGITS = frozenset({"911", "112", "999", "000", "110", "988", "933"})
_E164 = re.compile(r"^\+[1-9]\d{7,14}$")


@dataclass(frozen=True)
class SignalWireConfig:
    space: str
    project_id: str
    api_token: str
    from_number: str
    to_number: str
    public_base: str
    enabled: bool


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _paired_env(primary: str, legacy: str) -> str:
    new = os.environ.get(primary, "").strip()
    old = os.environ.get(legacy, "").strip()
    if new and old and new != old:
        raise ValueError("alias mismatch")
    return new or old


def _digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def verified_demo_number() -> str:
    """The only destination calls and SMS may use. Not logged."""
    return _paired_env("SIGNALWIRE_TO_NUMBER", "CS_DEMO_TO_NUMBER")


def destination_block_reason(number: str) -> str | None:
    """Reject emergency numbers, short codes, and every other destination."""
    raw = (number or "").strip()
    digits = _digits(raw)
    if not digits or digits in _EMERGENCY_DIGITS or raw in _EMERGENCY_DIGITS:
        return "emergency_or_missing_destination"
    if len(digits) < 8:
        return "short_code"
    if not _E164.fullmatch(raw):
        return "not_e164"
    allowed = verified_demo_number()
    if not allowed or raw != allowed:
        return "not_allowlisted"
    return None


def sms_fallback_call_enabled() -> bool:
    """A failed SMS may place a call only when this separate flag is set."""
    return _truthy("CS_SIGNALWIRE_SMS_FALLBACK_CALL")


def _signalwire_enabled() -> bool:
    if _truthy("CS_KILL_SWITCH"):
        return False
    return _truthy("CS_SIGNALWIRE_ENABLED")


def _credentials() -> tuple[str, str, str, str, str] | None:
    space = _first_env("SIGNALWIRE_SPACE")
    project = _first_env("SIGNALWIRE_PROJECT_ID")
    token = _first_env("SIGNALWIRE_API_TOKEN")
    frm = _paired_env("SIGNALWIRE_FROM_NUMBER", "SIGNALWIRE_FROM")
    to = verified_demo_number()
    if not all((space, project, token, frm, to)):
        return None
    if not _E164.fullmatch(frm):
        return None
    reason = destination_block_reason(to)
    if reason:
        raise ValueError(reason)
    space = space.removeprefix("https://").removeprefix("http://").rstrip("/")
    return space, project, token, frm, to


def load_config() -> SignalWireConfig | None:
    if not _signalwire_enabled():
        return None
    creds = _credentials()
    base = os.environ.get("CS_PUBLIC_BASE", "").strip().rstrip("/")
    if creds is None or not base:
        return None
    space, project, token, frm, to = creds
    return SignalWireConfig(space, project, token, frm, to, base, True)


def load_sms_config() -> SignalWireConfig | None:
    """SMS uses the same explicit enable flag as calls."""
    if not _signalwire_enabled():
        return None
    creds = _credentials()
    if creds is None:
        return None
    space, project, token, frm, to = creds
    return SignalWireConfig(space, project, token, frm, to, "", True)


def _reject_destination(number: str) -> dict[str, Any] | None:
    reason = destination_block_reason(number)
    if reason is None:
        return None
    return {"ok": False, "skipped": True, "reason": reason}


def cxml_connect_stream(stream_wss_url: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<Stream url="{stream_wss_url}" />'
        "</Connect></Response>"
    )


def build_call_request(cfg: SignalWireConfig, incident_id: str) -> Request:
    endpoint = (
        f"https://{cfg.space}/api/laml/2010-04-01/Accounts/"
        f"{quote(cfg.project_id, safe='')}/Calls.json"
    )
    voice_url = f"{cfg.public_base}/signalwire/voice?{urlencode({'incident_id': incident_id})}"
    body = urlencode(
        {"To": cfg.to_number, "From": cfg.from_number, "Url": voice_url, "Method": "POST"}
    ).encode()
    auth = base64.b64encode(f"{cfg.project_id}:{cfg.api_token}".encode()).decode()
    return Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )


def place_call(incident_id: str, *, config: SignalWireConfig | None = None) -> dict[str, Any]:
    cfg = config if config is not None else load_config()
    if cfg is None:
        return {"ok": False, "skipped": True, "reason": "signalwire_not_configured"}
    blocked = _reject_destination(cfg.to_number)
    if blocked is not None:
        return blocked
    with urlopen(build_call_request(cfg, incident_id), timeout=15) as response:
        payload = json.loads(response.read().decode())
    sid = payload.get("sid") or payload.get("call_sid")
    return {"ok": bool(sid), "call_sid": sid, "incident_id": incident_id}


def send_sms(body: str, *, config: SignalWireConfig | None = None) -> dict[str, Any]:
    """Text the verified demo handset. No number is included in the result."""
    cfg = config if config is not None else load_sms_config()
    if cfg is None:
        return {"ok": False, "skipped": True, "reason": "signalwire_not_configured"}
    blocked = _reject_destination(cfg.to_number)
    if blocked is not None:
        return blocked
    endpoint = (
        f"https://{cfg.space}/api/laml/2010-04-01/Accounts/"
        f"{quote(cfg.project_id, safe='')}/Messages.json"
    )
    auth = base64.b64encode(f"{cfg.project_id}:{cfg.api_token}".encode()).decode()
    req = Request(
        endpoint,
        data=urlencode({"From": cfg.from_number, "To": cfg.to_number, "Body": body[:1500]}).encode(),
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode())
    return {"ok": bool(payload.get("sid")), "sid": payload.get("sid"), "status": payload.get("status")}


def sos_call(text: str, *, config: SignalWireConfig | None = None) -> dict[str, Any]:
    """Optional call after SMS failure. Disabled unless the fallback flag is set."""
    if not sms_fallback_call_enabled():
        return {"ok": False, "skipped": True, "reason": "sms_fallback_call_disabled"}
    cfg = config if config is not None else load_config()
    if cfg is None:
        return {"ok": False, "skipped": True, "reason": "signalwire_not_configured"}
    blocked = _reject_destination(cfg.to_number)
    if blocked is not None:
        return blocked
    endpoint = (
        f"https://{cfg.space}/api/laml/2010-04-01/Accounts/"
        f"{quote(cfg.project_id, safe='')}/Calls.json"
    )
    url = f"{cfg.public_base}/signalwire/sos?{urlencode({'text': text[:500]})}"
    auth = base64.b64encode(f"{cfg.project_id}:{cfg.api_token}".encode()).decode()
    req = Request(
        endpoint,
        data=urlencode({"To": cfg.to_number, "From": cfg.from_number, "Url": url, "Method": "POST"}).encode(),
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode())
    return {"ok": bool(payload.get("sid")), "sid": payload.get("sid"), "via": "call"}


def cxml_say(text: str) -> str:
    from xml.sax.saxutils import escape

    said = escape(text)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Say>{said}</Say><Pause length=\"1\"/><Say>{said}</Say><Hangup/></Response>"
    )


def hangup_call(call_sid: str, *, config: SignalWireConfig | None = None) -> bool:
    """End a call that this demo placed. Reset must leave no call running."""
    cfg = config if config is not None else load_config()
    if cfg is None or not call_sid:
        return False
    number = getattr(cfg, "to_number", None)
    if isinstance(number, str) and _reject_destination(number) is not None:
        return False
    endpoint = (
        f"https://{cfg.space}/api/laml/2010-04-01/Accounts/"
        f"{quote(cfg.project_id, safe='')}/Calls/{quote(call_sid, safe='')}.json"
    )
    auth = base64.b64encode(f"{cfg.project_id}:{cfg.api_token}".encode()).decode()
    req = Request(
        endpoint,
        data=urlencode({"Status": "completed"}).encode(),
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urlopen(req, timeout=10) as response:
        return 200 <= response.status < 300


def service_ready(url: str) -> bool:
    """Probe the service's /health endpoint; configuration alone is not readiness."""
    if not url:
        return False
    try:
        parts = urlsplit(url)
        health_url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
        with urlopen(health_url, timeout=1) as response:
            payload = json.loads(response.read().decode())
        return payload.get("ok") is True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _missing_settings() -> list[str]:
    groups = (
        ("SIGNALWIRE_SPACE", ("SIGNALWIRE_SPACE",)),
        ("SIGNALWIRE_PROJECT_ID", ("SIGNALWIRE_PROJECT_ID",)),
        ("SIGNALWIRE_API_TOKEN", ("SIGNALWIRE_API_TOKEN",)),
        ("SIGNALWIRE_FROM_NUMBER", ("SIGNALWIRE_FROM_NUMBER", "SIGNALWIRE_FROM")),
        ("SIGNALWIRE_TO_NUMBER", ("SIGNALWIRE_TO_NUMBER", "CS_DEMO_TO_NUMBER")),
        ("CS_PUBLIC_BASE", ("CS_PUBLIC_BASE",)),
    )
    missing = []
    for label, keys in groups:
        if not any(os.environ.get(key, "").strip() for key in keys):
            missing.append(label)
    return missing


def status() -> dict[str, Any]:
    enabled = _truthy("CS_SIGNALWIRE_ENABLED")
    kill_switch = _truthy("CS_KILL_SWITCH")
    missing = _missing_settings()
    if not enabled:
        missing.insert(0, "CS_SIGNALWIRE_ENABLED")
    if kill_switch:
        missing.insert(0, "kill_switch_on")
    try:
        configured = load_config() is not None
    except ValueError:
        configured = False
        missing.append("destination_rejected")
    return {
        "configured": configured,
        "enabled_flag": enabled,
        "kill_switch": kill_switch,
        "sms_fallback_call": sms_fallback_call_enabled(),
        "missing": missing,
        "parakeet": bool(os.environ.get("CS_PARAKEET_URL")),
        "kokoro": bool(os.environ.get("CS_KOKORO_URL")),
    }
