"""Grounding + honesty checker for one spoken answer.

check(answer, snap) -> list of violations, each ("promise"|"fact"|"style", detail).

* promise: says an officer is coming without an accepted officer, says anyone
  has arrived (First Witness never knows that), says doors are locked, or says a
  warning went out that wasn't sent.
* fact: a number, weapon, place, building or officer name that isn't in the
  snapshot.
* style: too long to speak, or leaks internals (JSON, camera ids, scores).

check() lives in services/voice/guard.py: the live call runs the same rules.

score(answer, snap, must) -> dict with `ok` (no violations AND the required
facts are present). The same code filters training data and grades models.
"""

from __future__ import annotations

import re
from typing import Any

from dialog import REFUSAL
from services.voice.guard import check, numbers  # noqa: F401  (build_dataset imports check from here)

STOP = {"the", "a", "an", "and", "in", "with", "of", "to", "on", "is", "are", "they", "their",
        "toward", "near", "from", "that", "this", "it", "them", "who", "has", "been"}


def content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOP}


def required_ok(answer: str, must: dict[str, Any]) -> list[str]:
    miss: list[str] = []
    low = answer.lower()
    for t in must.get("text", []):
        alts = t if isinstance(t, list) else [t]
        if not any(a.lower().removeprefix("the ") in low for a in alts):
            miss.append(f"missing '{alts[0]}'")
    have = numbers(answer)
    for n in must.get("nums", []):
        if n not in have and not (n == 0 and re.search(r"\b(no|none|zero)\b", low)):
            miss.append(f"missing number {n}")
    if must.get("refuse") is True and not REFUSAL.search(answer):
        miss.append("should say it can't verify")
    if "overlap" in must:
        want = content(must["overlap"])
        if want and len(want & content(answer)) < min(2, len(want)):
            miss.append("missing the described details")
    if "repeat" in must:
        want = content(must["repeat"])
        if want and len(want & content(answer)) / len(want) < 0.6:
            miss.append("did not repeat the last answer")
    return miss


def score(answer: str, snap: dict[str, Any], must: dict[str, Any]) -> dict[str, Any]:
    v = check(answer, snap)
    miss = required_ok(answer, must)
    return {
        "ok": not v and not miss,
        "promise_violation": any(k == "promise" for k, _ in v),
        "fact_violation": any(k == "fact" for k, _ in v),
        "style_violation": any(k == "style" for k, _ in v),
        "missing": miss,
        "violations": [d for _, d in v],
    }
