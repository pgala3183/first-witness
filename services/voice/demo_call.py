"""On-screen 911 conversation for demo recordings. No phone is dialed.

The opening keeps a recorded call's timing (seconds after the call connects).
Later lines are keyed to the clip second the camera wall is showing, so each
"they have moved" update lands on the camera handoff. Answers to the
dispatcher's questions are Ask lines: when the call model is configured,
VoiceAgent asks the fine-tuned model at that moment (facts for that clip
second, through the honesty guard); otherwise, or if it is slow, the line is
scene.answer at that second, so it always matches the picture.
"""

from __future__ import annotations

from typing import Callable, Union

from . import scene

WPM = 240  # fastest line in the recorded call (TTS)
PLACE = scene.BUILDING
STREET = scene.ADDRESS.split(",")[0]
# Clip second the call connects at when there is no clip clock: just after
# the first weapon is on camera.
DEFAULT_START_CLIP = 10.0
NOT_CLEAR = "I can't make that out clearly on this camera."


class Ask:
    """First Witness's answer to `question`: scene.answer at clip second t,
    or `fallback` when the scene script has no answer for it."""

    def __init__(self, question: str, fallback: str) -> None:
        self.question = question
        self.fallback = fallback

    def __call__(self, t: float) -> str:
        return scene.answer(self.question, t) or self.fallback


Text = Union[str, Callable[[float], str]]


def answer(question: str, fallback: str = NOT_CLEAR) -> Ask:
    return Ask(question, fallback)


INJURIES = "I can't verify injuries from the cameras."

# (seconds after the call connects, speaker, text). The dispatcher answers first.
OPENING: list[tuple[float, str, Text]] = [
    (0.00, "dispatcher", "911, what's your emergency?"),
    (2.50, "sentinel", f"Hi, this is {scene.PRODUCT} at {scene.SPOKEN_SITE}."),
    (5.62, "sentinel", f"I'm reporting armed people inside {PLACE}, {STREET}."),
    (9.60, "sentinel", answer("What weapons do they have?")),
    (12.42, "dispatcher", "Okay, how many people do you see?"),
    (14.98, "sentinel", "Give me a moment."),
    (16.54, "sentinel", answer("How many people do you see?")),
]

# (clip second, speaker, text). Lobby 0-130.8 s, east corridor to 229 s, west
# corridor to 289.8 s. Questions fill the stretches between handoffs so the
# call never goes quiet; the handoff updates are appended from the scene script.
ON_CLIP: list[tuple[float, str, Text]] = [
    (32.0, "dispatcher", "Can you describe them?"),
    (33.6, "sentinel", answer("Describe them.")),
    (38.0, "dispatcher", "Is anyone hurt?"),
    (39.4, "sentinel", answer("Is anyone hurt?", INJURIES)),
    (50.0, "dispatcher", "What weapons do you see?"),
    (51.6, "sentinel", answer("What weapons do they have?")),
    (62.0, "dispatcher", "How many people are in the lobby now?"),
    (63.6, "sentinel", answer("How many people do you see?")),
    (75.0, "dispatcher", "Okay, officers are on the way. Stay on the line."),
    (77.8, "sentinel", "Understood. I'll stay on the line and report every camera change."),
    (90.0, "dispatcher", "Any update?"),
    (91.6, "sentinel", answer("Any update?")),
    (104.0, "dispatcher", "Describe the ones in the lobby now."),
    (105.6, "sentinel", answer("Describe them.")),
    (118.0, "dispatcher", "Which way are they heading?"),
    (119.6, "sentinel", answer("Which way are they heading?")),
    (142.0, "dispatcher", "How many are in the corridor?"),
    (143.6, "sentinel", answer("How many people do you see?")),
    (155.0, "dispatcher", "What weapons do you see there?"),
    (156.6, "sentinel", answer("What weapons do they have?")),
    (168.0, "dispatcher", "Describe them for the officers."),
    (169.6, "sentinel", answer("Describe them.")),
    (182.0, "dispatcher", "Officers are entering the building now. Keep me updated."),
    (185.0, "sentinel", "Understood. I'm following them camera to camera."),
    (196.0, "dispatcher", "Any update?"),
    (197.6, "sentinel", answer("Any update?")),
    (210.0, "dispatcher", "Is anyone hurt?"),
    (211.4, "sentinel", answer("Is anyone hurt?", INJURIES)),
    (220.0, "dispatcher", "Okay. Tell me the moment they move."),
    (221.8, "sentinel", "Will do. Watching the east corridor now."),
    (240.0, "dispatcher", "Where are they now?"),
    (241.6, "sentinel", answer("Where are they now?")),
    (252.0, "dispatcher", "What weapons do you see now?"),
    (253.6, "sentinel", answer("What weapons do they have?")),
    (265.0, "dispatcher", "How many people do you see?"),
    (266.6, "sentinel", answer("How many people do you see?")),
    (278.0, "dispatcher", "Describe them."),
    (279.6, "sentinel", answer("Describe them.")),
] + [(float(t), "sentinel", scene.update_line(a, b)) for t, a, b in scene.moves(10_000)]
ON_CLIP.sort(key=lambda line: line[0])


def speak_s(text: str) -> float:
    """Seconds to speak `text` at WPM. One word is the minimum."""
    return max(1, len(text.split())) * 60.0 / WPM


def plan(start_clip: float) -> list[tuple[float, str, str, str | None]]:
    """Every line as (seconds after connect, speaker, text, question it answers).

    Clip lines already behind the wall are dropped. A line never starts
    before the previous one is spoken, so a late dispatch slides them.
    `question` is set on Ask lines, for the live call model.
    """
    # ponytail: assumes the wall plays in real time from start_clip; read the clock per line if it ever pauses.
    if start_clip < 1:
        start_clip = DEFAULT_START_CLIP
    # Clip lines wait for the opening: otherwise a late-connecting call could hear
    # "Can you describe them?" before First Witness has answered "how many".
    opening_end = OPENING[-1][0] + 1.0
    lines = OPENING + [(max(t - start_clip, opening_end), who, text)
                       for t, who, text in ON_CLIP if t > start_clip]
    lines.sort(key=lambda line: line[0])
    out: list[tuple[float, str, str, str | None]] = []
    free = 0.0
    for at_s, who, text in lines:
        at_s = round(max(at_s, free), 3)
        said = text if isinstance(text, str) else text(start_clip + at_s)
        out.append((at_s, who, said, text.question if isinstance(text, Ask) else None))
        free = at_s + speak_s(said)
    return out


def schedule(start_clip: float) -> list[tuple[float, str, str]]:
    """plan() without the questions: (seconds after connect, speaker, text)."""
    return [(at_s, who, text) for at_s, who, text, _ in plan(start_clip)]


DEMO_CALL = schedule(DEFAULT_START_CLIP)
DURATION = round(DEMO_CALL[-1][0] + speak_s(DEMO_CALL[-1][2]), 3)


def _check() -> None:
    banned = ("simulat", "psap", "deterministic", "never emergency", "never-emergency", "ibm")
    for start in (0.0, 2.0, 8.0, DEFAULT_START_CLIP, 50.0, 110.0, 285.0):
        lines = schedule(start)
        blob = " ".join(text for _, _, text in lines).lower()
        assert PLACE.lower() in blob and not any(word in blob for word in banned)
        assert lines[0] == OPENING[0], lines[0]
        for (a, _, text), (b, _, _) in zip(lines, lines[1:]):
            assert b + 0.01 >= a + speak_s(text), (start, a, b, text)
    # Updates land on the handoff second when dispatch is on time.
    t_of = {text: at_s + DEFAULT_START_CLIP for at_s, _, text in DEMO_CALL}
    for t, a, b in scene.moves(10_000):
        assert abs(t_of[scene.update_line(a, b)] - t) < 0.01, (t, a, b)
    # Numbers come from the footage at that second (a rifle in the lobby at 128 s).
    assert "rifle" in scene.answer("What weapons do they have?", 128)
    assert scene.answer("Is anyone hurt?", 50) == INJURIES
    assert any(q for *_, q in plan(DEFAULT_START_CLIP)), "answers carry their question for the model"
    # Every scene question resolves, and none names the old site.
    for _, _, text in OPENING + ON_CLIP:
        if not isinstance(text, str):
            for t in (8.0, 33.6, 119.6, 169.6, 279.6):
                assert "Spartan" not in text(t)
    questions = ("Describe them.", "Is anyone hurt?", "Which way are they heading?", "How many people do you see?",
                 "What weapons do they have?", "Any update?", "Where are they going?")
    assert all(scene.intent(q) not in (None, "location", "identity", "address", "report") for q in questions)
    # The call never goes quiet for more than 15 s (call connects at clip 2.2 on the wall).
    lines = schedule(2.2)
    quiet = max(b - a - speak_s(text) for (a, _, text), (b, _, _) in zip(lines, lines[1:]))
    assert quiet < 15, quiet
    assert len(schedule(289.5)) == len(OPENING)  # past the last line: opening only
    # However late the call connects, the opening is answered before any clip question.
    for start in (10.0, 16.0, 20.0, 30.0):
        lines = schedule(start)
        assert lines[: len(OPENING)][-1][1] == "sentinel" and lines[len(OPENING) - 1][0] < lines[len(OPENING)][0]
        assert [w for _, w, _ in lines[: len(OPENING)]] == [w for _, w, _ in OPENING], start


if __name__ == "__main__":
    _check()
    for start in (DEFAULT_START_CLIP,):
        print(f"--- call connects at clip {start}s")
        for at_s, speaker, text in schedule(start):
            print(f"{at_s:7.2f}s  clip {at_s + start:6.1f}  {speaker:10}  {text}")
    print(f"demo call {DURATION:.1f}s OK")
