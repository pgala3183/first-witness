"""What First Witness says on its own: the opening report and live updates.

Built only from the FACTS snapshot (facts.live_snapshot): who is armed and
what they look like, which weapons, where they are, how many people, and
where they are heading in the building. Nothing is guessed, and every
sentence still goes through the honesty guard before it is spoken. The
dispatcher's follow-up questions go to the fine-tuned call model instead.

  opening(snap)            -> sentences for the start of the call
  Narrator().update(snap)  -> sentences when something changed, else []
"""

from __future__ import annotations

import time
from typing import Any

from . import scene
from .guard import check

PRODUCT = scene.PRODUCT
# Quiet time before a routine update (new weapons); handoffs and losing sight are said at once.
MIN_GAP_S = 12.0
# Movement-only updates are rarer, so the call isn't a running commentary.
MOVE_GAP_S = 25.0
# No armed person in view this long before "I've lost sight of them" (weapons drop out
# of view for a few seconds all the time).
LOST_AFTER_S = 8.0
HANDOFF_WAIT_S = 6.0


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _people(n: int, armed: int) -> str:
    if n <= 0:
        return "No one is in view on this camera yet"
    who = "one person" if n == 1 else f"{scene.num(n)} people"
    if armed <= 0:
        return f"I see {who}, no weapon in view right now"
    if n <= 1:
        return f"I see {who}, and they are armed"
    return f"I see {who}; {scene.num(armed)} of them armed"


def _who(subjects: list[str]) -> str:
    """'A man in a black t-shirt with a rifle, and a man in a red polo shirt with a handgun.'"""
    subjects = [s.strip().rstrip(".") for s in subjects if s and s.strip()]
    if not subjects:
        return ""
    shown = subjects[:3]
    text = shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + ", and " + shown[-1]
    more = "" if len(subjects) <= 3 else f", and {scene.num(len(subjects) - 3)} more"
    return _cap(text) + more + "."


def _heading(movement: str | None) -> str:
    text = (movement or "").strip()
    if not text or text.lower().startswith(("no one", "no movement", "nobody")):
        return ""
    return text if text.endswith(".") else text + "."


def _safe(lines: list[str], snap: dict[str, Any]) -> list[str]:
    """Only sentences the honesty guard passes are spoken."""
    return [s for s in lines if s and not check(s, snap)]


def opening(snap: dict[str, Any]) -> list[str]:
    now = snap.get("now") or {}
    seen = now if now.get("visible") else (snap.get("last_seen") or now)
    armed = int(seen.get("armed") or 0)
    what = "an armed person" if armed == 1 else "armed people"
    lines = [
        f"This is {PRODUCT}, the automated camera system at {snap['site']}.",
        f"I'm reporting {what} in {snap['building']}, {snap['site_address']}.",
    ]
    if now.get("visible"):
        weapons = scene.weapons_phrase(now.get("weapons") or {})
        lines.append(f"They're in {now['place']}. {_people(int(now.get('people') or 0), armed)}.")
        if armed:
            lines.append(f"I can see {weapons}.")
    else:
        lines.append(f"I've lost sight of them; they were last seen in {seen.get('place', 'the building')}.")
    lines += [_who(seen.get("subjects") or []), _heading(now.get("movement"))]
    lines.append("I can't verify injuries from the cameras. I'll update you as they move.")
    return _safe(lines, snap)


class Narrator:
    """Tracks what has been said and returns update sentences when the scene changes."""

    def __init__(self, snap: dict[str, Any], clock: Any = time.monotonic) -> None:
        self.clock = clock
        now = snap.get("now") or {}
        self.place = now.get("place")
        self.visible = bool(now.get("visible"))
        self.weapons: dict[str, int] = dict(now.get("weapons") or {})
        self.armed = int(now.get("armed") or 0)
        self.movement = _heading(now.get("movement"))
        self.officer = (snap.get("officer") or {}).get("status", "none")
        self.last = self.clock()
        self.last_move = self.clock()
        self.unseen_since: float | None = None
        self.lost_said = False
        self.handoff_since: float | None = None

    def update(self, snap: dict[str, Any]) -> list[str]:
        now = snap.get("now") or {}
        lines: list[str] = []
        urgent = False
        visible = bool(now.get("visible"))
        place = now.get("place")
        weapons = dict(now.get("weapons") or {})
        armed = int(now.get("armed") or 0)
        heading = _heading(now.get("movement"))
        t = self.clock()
        moved_only = False

        if place and place != self.place:
            # Camera handoff: the action is on another camera now. The new camera's first
            # second can still be empty, so wait up to HANDOFF_WAIT_S for people to appear.
            self.handoff_since = self.handoff_since if self.handoff_since is not None else t
            if not int(now.get("people") or 0) and t - self.handoff_since < HANDOFF_WAIT_S:
                return []
            urgent = True
            lines.append(f"Update: they've moved from {self.place} into {place}." if self.place
                         else f"Update: they're in {place}.")
            lines.append(f"{_people(int(now.get('people') or 0), armed)}.")
            lines += [_who(now.get("subjects") or []) if visible else "", heading]
            self.place, self.weapons, self.armed = place, weapons, armed
            self.unseen_since, self.lost_said, self.handoff_since = None, False, None
        elif not visible:
            self.unseen_since = self.unseen_since if self.unseen_since is not None else t
            if not self.lost_said and t - self.unseen_since >= LOST_AFTER_S:
                urgent, self.lost_said = True, True
                lines.append(f"Update: I can't see an armed person in {place} right now. I'm still watching.")
        else:
            self.unseen_since, self.lost_said = None, False
            new_kinds = [k for k in weapons if k not in self.weapons]
            if new_kinds or armed > self.armed:
                lines.append(f"Update: I now see {scene.weapons_phrase(weapons)} in {place}.")
                lines.append(_who(now.get("subjects") or []))
                self.weapons = {k: max(weapons.get(k, 0), self.weapons.get(k, 0)) for k in set(weapons) | set(self.weapons)}
                self.armed = max(armed, self.armed)
            elif heading and heading != self.movement and t - self.last_move >= MOVE_GAP_S:
                moved_only = True
                lines.append(f"Update: {heading[0].lower() + heading[1:]}")

        officer = snap.get("officer") or {}
        if officer.get("status") == "accepted" and self.officer != "accepted":
            eta = officer.get("eta_min")
            lines.append(f"{officer.get('name', 'An officer')} has accepted"
                         + (f" and is about {scene.num(int(eta))} minutes away." if eta is not None else "."))
            urgent = True
        self.officer = officer.get("status", "none")

        if not lines or (not urgent and t - self.last < MIN_GAP_S):
            return []
        self.visible, self.movement = visible, heading or self.movement
        self.last = t
        if moved_only or heading:
            self.last_move = t
        return _safe(lines, snap)


if __name__ == "__main__":
    from .facts import live_snapshot

    for t in (5, 128):
        print(f"--- opening at {t} s")
        for line in opening(live_snapshot(t)):
            print("  ", line)
    fake = [0.0]
    n = Narrator(live_snapshot(10), clock=lambda: fake[0])
    said = []
    for t in range(11, 290):
        fake[0] = float(t)
        for line in n.update(live_snapshot(t, {"status": "accepted", "name": "Officer A. Lee", "eta_min": 3}
                                           if t >= 150 else None)):
            said.append((t, line))
    for t, line in said:
        print(f"{t:4} {line}")
    texts = [line for _, line in said]
    assert any(line.startswith("Update: they've moved from the ground-floor lobby into the east corridor") for line in texts)
    assert any("west corridor near the stairwell" in line for line in texts)
    assert sum("can't see an armed person" in line for line in texts) <= 4, "lost-sight must not flap"
    assert any("Officer A. Lee has accepted" in line for line in texts)
    blob = " ".join(texts + opening(live_snapshot(5))).lower()
    assert not any(w in blob for w in ("frame", "foreground", "left side", "toward the camera"))
    print(f"briefing self-check OK: {len(said)} update lines over 290 s")
