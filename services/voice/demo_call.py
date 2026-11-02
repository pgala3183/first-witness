"""On-screen campus police dispatch conversation for demo recordings. No phone is dialed.

Copied from the recorded call in sentinel_old_demo.mp4, word for word. The opening keeps that call's timing (seconds after the
call connects). Later lines are keyed to the clip second the camera wall is
showing, so each "they have moved" update lands on the camera handoff.
First Witness's counts and weapons are scene.answer at the second they are said,
the same wording as the recording, so they always match the picture.
"""

from __future__ import annotations

from typing import Callable, Union

from . import scene

WPM = 240  # fastest line in the recorded call (TTS)
PLACE = scene.BUILDING
STREET = scene.ADDRESS.split(",")[0]
# Clip second the old recording connected at (its first update fell on the
# 60 s handoff 27.7 s in). Used when there is no clip clock.
DEFAULT_START_CLIP = 32.3

Text = Union[str, Callable[[float], str]]


def answer(question: str, fallback: str) -> Callable[[float], str]:
    """Our answer to `question` at clip second t, from the scene script."""
    return lambda t: scene.answer(question, t) or fallback


# (seconds after the call connects, speaker, text). The recorded call's
# lines and gaps, with the dispatcher answering first.
OPENING: list[tuple[float, str, Text]] = [
    (0.00, "dispatcher", "Campus police dispatch, what's your emergency?"),
    (2.50, "sentinel", f"Hi, this is {scene.PRODUCT} at {scene.SPOKEN_SITE}."),
    (5.62, "sentinel", f"I'm reporting armed people inside {PLACE}, {STREET}."),
    (9.60, "sentinel", answer("What weapons do they have?", "I see handguns and a knife in the ground-floor lobby.")),
    (12.42, "dispatcher", "Okay, how many people do you see?"),
    (14.98, "sentinel", "Give me a moment."),
    (16.54, "sentinel", answer("How many people do you see?", "Eight people in the ground-floor lobby; six of them armed.")),
]

# (clip second, speaker, text). Questions fill the stretches between camera
# handoffs so the call never goes quiet; every answer is read off the scene
# at that second. The 117 s exchange keeps the old call's spacing.
ON_CLIP: list[tuple[float, str, Text]] = [
    (30.0, "dispatcher", "Can you describe them?"),
    (31.6, "sentinel", answer("Describe them.", "A group of men with handguns, and one with a knife.")),
    (45.0, "dispatcher", "Is anyone hurt?"),
    (46.4, "sentinel", answer("Is anyone hurt?", "I can't verify injuries from the cameras.")),
    (51.0, "dispatcher", "Okay. Are they still in the lobby?"),
    (52.6, "sentinel", "Yes, still in the ground-floor lobby. I'm tracking each of them across every camera."),
    (59.0, "sentinel", "One moment."),
    (67.0, "dispatcher", "Which way are they heading?"),
    (68.6, "sentinel", answer("Which way are they heading?", "They're in the east corridor on the ground floor now.")),
    (77.0, "dispatcher", "How many are in the corridor?"),
    (78.6, "sentinel", answer("How many people do you see?", "Two people in the east corridor on the ground floor.")),
    (85.0, "dispatcher", "Describe the ones in the corridor."),
    (86.6, "sentinel", answer("Describe them.", "Two men, one with a handgun and one with a knife.")),
    (96.0, "dispatcher", "Okay, officers are on the way. Stay on the line."),
    (98.8, "sentinel", "Understood. I'll stay on the line and report every camera change."),
    (105.0, "dispatcher", "Okay, what weapons do they have?"),
    (106.6, "sentinel", "Let me check that."),
    (107.6, "sentinel", answer("What weapons do they have?", "I see a handgun and a rifle in the east corridor on the ground floor.")),
    (113.0, "dispatcher", "Where are they now?"),
    (114.6, "sentinel", "They're in the east corridor on the ground floor."),
    (125.0, "dispatcher", "How many in the west corridor?"),
    (126.6, "sentinel", answer("How many people do you see?", "Two people in the west corridor near the stairwell.")),
    (137.0, "dispatcher", "What weapons do you see there?"),
    (138.6, "sentinel", answer("What weapons do they have?", "I see a handgun in the west corridor near the stairwell.")),
    (150.0, "dispatcher", "Describe them for the officers."),
    (151.6, "sentinel", answer("Describe them.", "A man with a rifle and a man in a light-blue shirt ahead of him.")),
    (170.0, "dispatcher", "Officers are entering the building now. Keep me updated."),
    (173.0, "sentinel", "Understood. I'm following them camera to camera."),
    (185.0, "dispatcher", "Are they still on the west side?"),
    (186.6, "sentinel", "Yes, still in the west corridor near the stairwell. I have them on camera."),
    (197.0, "dispatcher", "Any update?"),
    (198.6, "sentinel", answer("Any update?", "They're still in the west corridor near the stairwell.")),
    (213.0, "dispatcher", "Is anyone hurt?"),
    (214.4, "sentinel", answer("Is anyone hurt?", "I can't verify injuries from the cameras.")),
    (221.0, "dispatcher", "Okay. Tell me the moment they move."),
    (222.8, "sentinel", "Will do. Watching the west corridor now."),
    (240.0, "dispatcher", "Where are they going?"),
    (241.6, "sentinel", "They've turned back. They're in the east corridor on the ground floor, heading toward the lobby."),
    (252.0, "dispatcher", "What weapons do you see now?"),
    (253.6, "sentinel", answer("What weapons do they have?", "I see a handgun in the east corridor on the ground floor.")),
    (280.0, "dispatcher", "How many in the lobby now?"),
    (281.6, "sentinel", answer("How many people do you see?", "Two people in the ground-floor lobby.")),
    (295.0, "dispatcher", "Which way are they heading?"),
    (296.6, "sentinel", "They're back in the ground-floor lobby, the way they came in."),
] + [(float(t), "sentinel", scene.update_line(a, b)) for t, a, b in scene.moves(10_000)]
ON_CLIP.sort(key=lambda line: line[0])


def speak_s(text: str) -> float:
    """Seconds to speak `text` at WPM. One word is the minimum."""
    return max(1, len(text.split())) * 60.0 / WPM


def schedule(start_clip: float) -> list[tuple[float, str, str]]:
    """Every line as seconds after the call connects at clip `start_clip`.

    Clip lines already behind the wall are dropped. A line never starts
    before the previous one is spoken, so a late dispatch slides them.
    """
    # ponytail: assumes the wall plays in real time from start_clip; read the clock per line if it ever pauses.
    if start_clip < 1:
        start_clip = DEFAULT_START_CLIP
    lines = OPENING + [(t - start_clip, who, text) for t, who, text in ON_CLIP if t > start_clip]
    lines.sort(key=lambda line: line[0])
    out: list[tuple[float, str, str]] = []
    free = 0.0
    for at_s, who, text in lines:
        at_s = round(max(at_s, free), 3)
        said = text if isinstance(text, str) else text(start_clip + at_s)
        out.append((at_s, who, said))
        free = at_s + speak_s(said)
    return out


DEMO_CALL = schedule(DEFAULT_START_CLIP)
DURATION = round(DEMO_CALL[-1][0] + speak_s(DEMO_CALL[-1][2]), 3)


def _check() -> None:
    banned = ("simulat", "psap", "deterministic", "never emergency", "never-emergency", "ibm")
    for start in (0.0, 2.0, 8.0, DEFAULT_START_CLIP, 50.0, 110.0, 331.5):
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
    # Same wording as the recording, numbers from the footage at that second.
    assert scene.answer("What weapons do they have?", 24) == "I see four handguns and one knife in the ground-floor lobby."
    assert scene.answer("What weapons do they have?", 107.6) == "I see one handgun and one rifle in the east corridor on the ground floor."
    assert scene.answer("How many people do you see?", 24).startswith("Eight people in the ground-floor lobby;")
    # Every scene question resolves, and none names the old site.
    for _, _, text in OPENING + ON_CLIP:
        if not isinstance(text, str):
            for t in (8.0, 31.6, 68.6, 151.6, 296.6):
                assert "Spartan" not in text(t)
    questions = ("Describe them.", "Is anyone hurt?", "Which way are they heading?", "How many people do you see?",
                 "What weapons do they have?", "Any update?", "Where are they going?")
    assert all(scene.intent(q) not in (None, "location", "identity", "address", "report") for q in questions)
    # The call never goes quiet for more than 15 s (call connects at clip 2.2 on the wall).
    lines = schedule(2.2)
    quiet = max(b - a - speak_s(text) for (a, _, text), (b, _, _) in zip(lines, lines[1:]))
    assert quiet < 15, quiet
    assert len(schedule(331.5)) == len(OPENING)  # past the last handoff: opening only


if __name__ == "__main__":
    _check()
    for start in (8.0, DEFAULT_START_CLIP):
        print(f"--- call connects at clip {start}s")
        for at_s, speaker, text in schedule(start):
            print(f"{at_s:7.2f}s  clip {at_s + start:6.1f}  {speaker:10}  {text}")
    print(f"demo call {DURATION:.1f}s OK")
