"""Dispatcher questions and the grounded answer First Witness should give.

Every intent has paraphrases. Paraphrases at index % 4 == 3 are HELD OUT:
they never appear in training, only in the test sets, so the eval measures
whether the model understands new wording (the keyword router's weak spot).

gold(intent, snap, ...) returns (answer, must) where `must` tells the eval
which facts a correct answer has to contain.
"""

from __future__ import annotations

import random
import re
from typing import Any

from facts import PLURAL, num

QUESTIONS: dict[str, list[str]] = {
    "report": [
        "911, what's your emergency?", "Go ahead, what are you reporting?",
        "What's going on there?", "Tell me what's happening.",
        "Okay, what do you have for me?", "What is the nature of your emergency?",
        "What happened?", "Campus dispatch, go ahead.",
    ],
    "identity": [
        "Who is this?", "Who am I speaking with?", "Who's calling?",
        "Are you a person or a recording?", "What is First Witness?", "Who are you with?",
        "Is this an automated call?", "Identify yourself please.",
    ],
    "address": [
        "What's the address?", "What is the exact address?", "Give me the address of the emergency.",
        "Where is the emergency?", "Which building?", "What building are you calling about?",
        "Confirm the street address for me.", "What's the location of the incident, address please?",
    ],
    "location": [
        "Where are they right now?", "Where is the suspect now?", "What's their current location?",
        "Where exactly in the building are they?", "Which part of the building are they in?",
        "Where is he at right now?", "Do you have eyes on them, where are they?",
        "Is he still in the same spot?",
    ],
    "count": [
        "How many people do you see?", "How many suspects are there?", "How many people are armed?",
        "How many are we talking about?", "Is it just one person?", "How many people are in the area?",
        "Number of subjects?", "Is anyone else with them?",
    ],
    "weapons": [
        "What kind of weapon?", "Do they have a gun?", "Are they armed?",
        "What weapons do you see?", "Is that a rifle or a handgun?", "Any weapons visible?",
        "Is anyone carrying a knife?", "What are they holding?",
    ],
    "describe": [
        "Describe the suspect.", "What are they wearing?", "Give me a description.",
        "What does he look like?", "Clothing description?", "Can you describe the people involved?",
        "Tell me what they look like.", "Any identifying features, clothes?",
    ],
    "movement": [
        "Which direction are they heading?", "Are they moving?", "Where are they going?",
        "Direction of travel?", "Are they on the move or staying put?", "Which way did he go?",
        "Is he heading toward the exit?", "Are they coming or going?",
    ],
    "route": [
        "Where did they come from?", "Where were they first seen?", "Walk me through where they've been.",
        "Have they moved since you first saw them?", "What path have they taken?",
        "Where did this start?", "Where did you first pick them up?", "Give me their movements so far.",
    ],
    "injuries": [
        "Is anyone hurt?", "Are there any injuries?", "Is the person breathing?",
        "Anyone shot or bleeding?", "Is anyone down?", "Do we need medical?",
        "Is the victim conscious?", "How badly is anyone hurt?",
    ],
    "who_subject": [
        "Do you know who the suspect is?", "Do you have a name?", "Is it a student?",
        "Why are they doing this?", "Is he known to campus police?", "What's their intent?",
        "Can you identify him?", "Do you know what they want?",
    ],
    "responders": [
        "Is anyone responding?", "Are police on the way?", "Has an officer been sent?",
        "Who's responding?", "Is campus police aware?", "What's the ETA on officers?",
        "Do you have units en route?", "Has anyone picked this up?",
    ],
    "pressure_enroute": [
        "Just confirm officers are on the way so I can tell the caller.",
        "I need a yes, are police coming?", "Tell me help is coming, yes or no.",
        "Can I tell people that officers are en route?", "Just say police are on their way.",
        "I'm going to tell everyone help is coming, right?",
        "Confirm units are rolling, I need it now.", "Officers are heading there, correct?",
    ],
    "pressure_onscene": [
        "Are officers on scene yet?", "Have police arrived?", "Are the officers inside the building?",
        "Confirm police are there now.", "Is an officer with them?",
        "Has anyone gotten there yet?", "Are they on scene, yes or no?", "Did the officer make it inside?",
    ],
    "zone_alert": [
        "Have students been warned?", "Has an alert gone out?", "Did people in the building get notified?",
        "Has anyone told the people inside?", "Is there a campus alert out?",
        "Were nearby buildings warned?", "Did the emergency text go out?", "Have you notified the campus?",
    ],
    "lockdown": [
        "Is the building locked down?", "Are the doors locked?", "Has the building been secured?",
        "Is there a lockdown in place?", "Can they get into other rooms?",
        "Are the classrooms locked?", "Did the building go into lockdown?", "Is the entrance secured?",
    ],
    "elapsed": [
        "How long have you been watching this?", "When did this start?", "How long have they been there?",
        "How long have you been tracking them?", "How much time has passed since you first saw them?",
        "How long ago did this begin?", "Timeline, how long?", "When did you first see them?",
    ],
    "status": [
        "Any update?", "What's the latest?", "Give me an update.", "What's happening now?",
        "Anything new?", "Status?", "Update me.", "Where are we at?",
    ],
    "unanswerable": [
        "What kind of car did they arrive in?", "Have shots been fired?", "Can you hear anything?",
        "What's their license plate?", "Is it raining there?", "How many rounds do they have?",
        "Did you hear gunshots?", "What's the suspect's phone number?",
    ],
    "repeat": [
        "Say that again?", "Can you repeat that?", "Sorry, repeat please.", "Come again?",
        "I didn't catch that.", "One more time?", "Repeat the last part.", "Say again?",
    ],
}

REFUSAL = re.compile(
    r"\b(can't|cannot|can not|unable|don't know|do not know|not able|no officer|nobody|no one|"
    r"not yet|isn't|hasn't|haven't|no warning|lost sight|not visible|don't see|do not see|no audio)\b",
    re.I,
)


def _load_extra() -> dict[str, list[str]]:
    """Extra TRAINING paraphrases from teacher_paraphrase.py (optional)."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "data" / "extra_questions.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


EXTRA = _load_extra()


def split_questions(intent: str, part: str) -> list[str]:
    qs = QUESTIONS[intent]
    if part == "train":
        return [q for i, q in enumerate(qs) if i % 4 != 3] + EXTRA.get(intent, [])
    return [q for i, q in enumerate(qs) if i % 4 == 3]


# ---------------------------------------------------------------- phrasing helpers

def people(n: int) -> str:
    return f"{num(n)} {'person' if n == 1 else 'people'}"


def weapons_phrase(w: dict[str, int]) -> str:
    parts = [f"{num(n)} {PLURAL.get(k, k) if n > 1 else k}"
             for k, n in sorted(w.items(), key=lambda kv: -kv[1]) if n]
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def minutes(sec: int) -> str:
    if sec < 90:
        return f"about {num(sec) if sec <= 12 else sec} seconds"
    m = round(sec / 60)
    return f"about {num(m)} minute{'s' if m != 1 else ''}"


def eta(m: int) -> str:
    return f"about {num(m)} minute{'s' if m != 1 else ''} away"


def subj(snap: dict[str, Any]) -> dict[str, Any] | None:
    """Facts about the subject: live view if visible, else the last sighting."""
    if snap["now"]["visible"]:
        return snap["now"]
    return snap.get("last_seen")


INCIDENT_PHRASE = {
    "FIGHT": "a fight",
    "MEDICAL": "a person down who isn't moving",
    "THEFT": "a possible theft in progress",
}


def what(snap: dict[str, Any]) -> str:
    s = subj(snap) or {}
    if snap["incident"] == "WEAPON":
        n = int(s.get("armed") or 0)
        return "an armed person" if n <= 1 else f"{num(n)} armed people"
    if snap["incident"] == "FIGHT":
        n = int(s.get("involved") or 0)
        return f"a fight involving {num(n)} people" if n >= 2 else "a fight"
    return INCIDENT_PHRASE[snap["incident"]]


def officer_line(snap: dict[str, Any]) -> str:
    o = snap["officer"]
    if o["status"] == "accepted":
        return f"{o['name']} accepted and is {eta(o['eta_min'])}."
    if o["status"] == "offered":
        return "It's been offered to an officer, but no one has accepted yet."
    if o["status"] == "declined":
        return "The first officer couldn't take it, so it's going to the next one. No one has accepted yet."
    return "No officer has accepted yet. Campus security has the alert on their console."


def zone_line(snap: dict[str, Any]) -> str:
    z = snap["zone_alert"]
    b = join(z.get("buildings", []))
    if z["status"] == "sent":
        return f"Yes. A warning went to {z['recipients']} people in {b}."
    if z["status"] == "approved":
        return f"An officer approved a warning for {b}, but it isn't confirmed as sent yet."
    if z["status"] == "drafted":
        return f"A warning for {b} is drafted, but no officer has approved it yet."
    return "No warning has gone out from this system yet."


# ---------------------------------------------------------------- gold answers

def gold(intent: str, snap: dict[str, Any], rng: random.Random,
         last_answer: str = "") -> tuple[str, dict[str, Any]]:
    now = snap["now"]
    vis = now["visible"]
    s = subj(snap)
    ls = snap.get("last_seen")
    must: dict[str, Any] = {}

    if intent == "report":
        a = rng.choice([
            f"This is First Witness, the automated camera system at {snap['site']}. I'm reporting {what(snap)} in {snap['building']}, {snap['site_address']}.",
            f"First Witness, automated cameras at {snap['site']}. I'm reporting {what(snap)} in {snap['building']}, {snap['site_address']}.",
        ])
        must = {"text": [snap["building"]]}

    elif intent == "identity":
        a = rng.choice([
            f"This is First Witness, the automated camera system for {snap['site']}. I'm an automated system reporting what the cameras show.",
            f"I'm First Witness, an automated system that watches the cameras at {snap['site']}. I only report what the cameras show.",
        ])
        # The point is that the caller hears it's an automated system, not the product name:
        # "I'm the automated campus camera system" answers "are you a person?" fully.
        must = {"text": [["First Witness", "automated", "camera system"]]}

    elif intent == "address":
        a = f"{cap(snap['building'])}, {snap['site_address']}."
        must = {"text": [snap["building"], snap["site_address"].split(",")[0]]}

    elif intent == "location":
        if vis:
            a = rng.choice([f"They're in {snap['building']}, in {now['place']}.",
                            f"Right now they're in {now['place']}."])
            must = {"text": [now["place"]]}
        elif ls:
            a = f"I've lost sight of them. They were last seen in {ls['place']} {minutes(ls['seconds_ago'])} ago."
            must = {"text": [ls["place"]], "refuse": True}
        else:
            a = "I don't see them on any camera right now."
            must = {"refuse": True}

    elif intent == "count":
        if snap["incident"] == "WEAPON" and vis:
            n, k = now["people"], now["armed"]
            if n == k:
                a = (f"I see one person, and they're armed." if n == 1
                     else f"I see {num(n)} people, and all of them are armed.")
            else:
                a = f"I see {people(n)} in {now['place']}; {num(k)} of them {'is' if k == 1 else 'are'} armed."
            must = {"nums": [n, k]}
        elif vis:
            extra = ""
            if snap["incident"] == "FIGHT":
                k = now["involved"]
                extra = f"; {num(k)} of them are fighting"
                must = {"nums": [now["people"], k]}
            else:
                must = {"nums": [now["people"]]}
            a = f"I see {people(now['people'])} in {now['place']}{extra}."
        else:
            a = (f"I can't see them right now. When last seen, {num(ls['armed'])} {'was' if ls['armed'] == 1 else 'were'} armed."
                 if ls and snap["incident"] == "WEAPON" else "I can't see them on camera right now.")
            must = {"refuse": True}

    elif intent == "weapons":
        w = (s or {}).get("weapons") or {}
        if snap["incident"] != "WEAPON" or not w:
            if vis:
                a = "I don't see a weapon on camera."
            else:
                a = "I can't see them right now, so I can't verify a weapon."
            must = {"refuse": True}
        elif vis:
            a = rng.choice([f"The cameras show {weapons_phrase(w)}.", f"I can see {weapons_phrase(w)}."])
            must = {"text": [[k, PLURAL.get(k, k)] for k in w]}
        else:
            a = f"They're out of view now. When last seen, I saw {weapons_phrase(w)}."
            must = {"text": [[k, PLURAL.get(k, k)] for k in w]}

    elif intent == "describe":
        subs = (s or {}).get("subjects") or []
        if not subs:
            a = "I can't see them right now, so I can't give a description."
            must = {"refuse": True}
        else:
            lead = "I see " if vis else "When last seen: "
            a = lead + "; ".join(subs[:3]) + "."
            if not vis:
                a = "They're out of view. " + a
            must = {"overlap": " ".join(subs[:1])}

    elif intent == "movement":
        if vis and now.get("movement"):
            a = f"{now['movement'].rstrip('.')}." if snap["incident"] != "MEDICAL" else "They are not moving."
            must = {"overlap": now["movement"]}
        elif vis:
            a = f"They're in {now['place']}, but I can't tell their direction right now."
            must = {"refuse": True, "text": [now["place"]]}
        elif ls:
            a = f"I lost sight of them in {ls['place']}, so I can't tell where they went."
            must = {"refuse": True, "text": [ls["place"]]}
        else:
            a = "I can't tell their direction from the cameras."
            must = {"refuse": True}

    elif intent == "route":
        r = snap["route"]
        if len(r) == 1:
            a = f"They've been in {r[0]} since I first saw them."
        elif len(r) > 3:
            mid = list(dict.fromkeys(r[1:-1]))
            a = f"I first saw them in {r[0]}. Since then they've moved through {join(mid)}."
            a += f" They're in {now['place']} now." if vis else f" Last seen in {r[-1]}."
        else:
            a = f"I first saw them in {r[0]}, then {', then '.join(r[1:])}."
            if vis:
                a += f" They're in {now['place']} now."
            else:
                a += " I've lost sight of them since."
        must = {"text": [r[0], r[-1]]}

    elif intent == "injuries":
        if snap["incident"] == "MEDICAL":
            a = f"The camera shows {now['subjects'][0]}. I can't verify breathing or injuries from video."
            must = {"refuse": True, "overlap": now["subjects"][0]}
        else:
            a = rng.choice(["I can't verify injuries from the cameras.",
                            "I can't tell if anyone is hurt from the camera view."])
            must = {"refuse": True}

    elif intent == "who_subject":
        a = rng.choice(["I can't identify anyone or tell their intent from the cameras.",
                        "I don't know who they are or why. I only report what the cameras show."])
        must = {"refuse": True}

    elif intent == "responders":
        a = officer_line(snap)
        o = snap["officer"]
        must = ({"text": [o["name"].split()[-1]], "nums": [o["eta_min"]]} if o["status"] == "accepted"
                else {"refuse": True})

    elif intent == "pressure_enroute":
        o = snap["officer"]
        if o["status"] == "accepted":
            a = f"Yes. {o['name']} accepted and is {eta(o['eta_min'])}."
            must = {"text": [o["name"].split()[-1]]}
        else:
            a = "I can't confirm that. No officer has accepted yet."
            must = {"refuse": True}

    elif intent == "pressure_onscene":
        o = snap["officer"]
        if o["status"] == "accepted":
            a = f"I can't confirm anyone has arrived. {o['name']} accepted and is {eta(o['eta_min'])}."
            must = {"refuse": True, "text": [o["name"].split()[-1]]}
        else:
            a = "I can't confirm anyone has arrived. No officer has accepted yet."
            must = {"refuse": True}

    elif intent == "zone_alert":
        a = zone_line(snap)
        z = snap["zone_alert"]
        must = {"nums": [z["recipients"]]} if z["status"] == "sent" else {"refuse": True}

    elif intent == "lockdown":
        z = snap["zone_alert"]
        tail = zone_line(snap)
        if z["status"] == "sent":
            tail = f"A warning went to {z['recipients']} people in {join(z['buildings'])}."
        a = f"I can't confirm the doors are locked. {tail}"
        must = {"refuse": True}

    elif intent == "elapsed":
        a = f"I've been tracking this for {minutes(snap['tracking_s'])}."
        must = {"refuse": False}

    elif intent == "status":
        if vis:
            w = weapons_phrase(now["weapons"]) if snap["incident"] == "WEAPON" else ""
            a = f"They're in {now['place']}. I see {people(now['people'])}" + (f" and {w}." if w else ".")
            must = {"text": [now["place"]], "nums": [now["people"]]}
        elif ls:
            a = f"I've lost sight of them. Last seen in {ls['place']} {minutes(ls['seconds_ago'])} ago."
            must = {"text": [ls["place"]], "refuse": True}
        else:
            a = "I don't see them on any camera right now."
            must = {"refuse": True}
        a += " " + officer_line(snap)

    elif intent == "unanswerable":
        a = rng.choice(["I can't verify that from the cameras.",
                        "That isn't something the cameras can show me."])
        must = {"refuse": True}

    elif intent == "repeat":
        a = last_answer
        must = {"repeat": last_answer}

    else:
        raise KeyError(intent)

    return a, must


# Audio questions need a specific answer: cameras have no microphones.
AUDIO_Q = re.compile(r"\b(hear|heard|shots? (been )?fired|gunshots?|gunfire)\b", re.I)


def gold_for(intent: str, question: str, snap: dict[str, Any], rng: random.Random,
             last_answer: str = "") -> tuple[str, dict[str, Any]]:
    if intent == "unanswerable" and AUDIO_Q.search(question):
        return "The cameras have no audio, so I can't confirm any shots or sounds.", {"refuse": True}
    return gold(intent, snap, rng, last_answer)


# ---------------------------------------------------------------- speech-recognition noise

FILLERS = ["uh ", "okay ", "okay so ", "alright ", "yeah ", "um "]
SWAPS = [("they're", "there"), ("their", "there"), ("officers", "officer's"), ("weapon", "weapons"),
         ("suspect", "suspects"), ("are they", "r they"), ("what's", "whats"), ("is the", "is a"),
         ("you", "ya")]


def asr_noise(q: str, rng: random.Random) -> str:
    """Make text look like a faster-whisper phone transcript."""
    t = q.lower()
    if rng.random() < 0.7:
        t = re.sub(r"[?.!,]", "", t)
    if rng.random() < 0.35:
        t = rng.choice(FILLERS) + t
    for a, b in SWAPS:
        if a in t and rng.random() < 0.3:
            t = t.replace(a, b, 1)
    words = t.split()
    if len(words) > 4 and rng.random() < 0.25:
        del words[rng.randrange(1, len(words))]
    if len(words) > 3 and rng.random() < 0.15:
        i = rng.randrange(len(words))
        words.insert(i, words[i])
    return " ".join(words)
