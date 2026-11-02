"""Fact snapshots: the only things the call agent is allowed to say.

A snapshot is what First Witness knows at the moment a dispatcher asks a
question. Two sources:

* real_snapshot(t)  - built from data/feeds/seville_scene_script.json
  (YOLO counts, dataset weapon labels, Qwen3-VL descriptions) and
  data/camera_map.json, exactly like services/voice/scene.py reads them.
  live_snapshot(t, officer, zone_alert) is the same scene with the call's
  real officer and warning status: what VoiceAgent sends the call model.
* synthetic_snapshot(rng) - randomized buildings, places, incident types,
  people and descriptions, so the model learns to READ the facts instead of
  memorizing MacQuarrie Hall.

Both get randomized officer and zone-alert status, which is what teaches the
"never promise help nobody has accepted" rule.

finetune/ imports this module, so training data and live answers build
FACTS the same way.
"""

from __future__ import annotations

import json
import random
from typing import Any

from services.brain.call_brief import site_config
from services.voice import scene  # the project's own scene reader
from services.voice.call_model import dumps  # noqa: F401  (the live agent's FACTS format)

# The demo site, from data/camera_map.json: the same names the live agent says.
_SITE = site_config()
SJSU = {
    "site": _SITE.get("name") or "San Jose State University",
    "site_address": _SITE.get("address") or "One Washington Square, San Jose, CA 95192",
}

# ---------------------------------------------------------------- vocab

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve"]


def num(n: int) -> str:
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


PLURAL = {"handgun": "handguns", "rifle": "rifles", "knife": "knives"}

SJSU_BUILDINGS = [
    "MacQuarrie Hall", "Clark Hall", "Duncan Hall", "the Engineering Building",
    "the Student Union", "the King Library", "Sweeney Hall", "Washington Square Hall",
    "the Boccardo Business Center", "the Spartan Recreation Center", "Washburn Hall",
    "Hugh Gillis Hall", "the Industrial Studies Building", "Dwight Bentel Hall",
]
# Clearly fictional campuses so the model never ties answers to one site.
OTHER_SITES = [
    ("Riverbend University", "400 College Avenue, Fresno, CA 93740",
     ["Hollis Hall", "the Science Center", "Marlow Library", "the Arts Building", "Kendra Hall"]),
    ("Northfield Community College", "2150 Oak Street, Sacramento, CA 95816",
     ["Building C", "the Learning Center", "the Gym", "the Student Services Building"]),
    ("Pacific Crest College", "88 Harbor Road, Santa Cruz, CA 95060",
     ["Ocean Hall", "the Library", "Cypress Hall", "the Dining Commons", "the Theater"]),
    ("Valley State University", "1 University Drive, Modesto, CA 95350",
     ["Founders Hall", "the Engineering Annex", "the Commons", "Redwood Hall"]),
]
# Only used in test_synth: unseen campus names.
HELDOUT_SITES = [
    ("Lakeside University", "700 Shoreline Boulevard, Stockton, CA 95211",
     ["Juniper Hall", "the Media Center", "the North Gym", "Alder Library"]),
    ("Mesa Verde College", "35 Canyon Way, Riverside, CA 92521",
     ["Building F", "the Welcome Center", "Sage Hall", "the Nursing Building"]),
]

FLOORS = ["ground", "first", "second", "third", "fourth"]
DIRS = ["east", "west", "north", "south"]


def place_pool(rng: random.Random) -> list[str]:
    f = rng.choice(FLOORS)
    d = rng.choice(DIRS)
    return [
        f"the {f}-floor lobby" if f != "ground" else "the ground-floor lobby",
        f"the {d} corridor on the {f} floor",
        f"the {d} stairwell",
        f"the {f}-floor study area",
        "the main entrance",
        f"the {d} entrance",
        "the cafeteria",
        f"the {f}-floor hallway near the elevators",
        f"level {rng.randint(2, 5)} of the parking garage",
        "the outdoor courtyard",
        f"the {d} side loading dock",
        f"the {f}-floor computer lab",
    ]


NOUNS = ["man", "man", "woman", "woman", "person"]
BUILDS = ["", "", "tall ", "heavyset ", "slim ", "short "]
TOPS = ["a black hoodie", "a red polo shirt", "a grey T-shirt", "a white button-up shirt",
        "a green jacket", "a blue sweater", "a black T-shirt", "a tan jacket",
        "an orange vest", "a navy windbreaker", "a yellow raincoat", "a maroon sweatshirt",
        "a striped shirt", "a denim jacket", "a white T-shirt"]
BOTTOMS = ["jeans", "dark pants", "khaki shorts", "black leggings", "grey sweatpants",
           "cargo pants", "blue shorts", "a black skirt", "camouflage pants"]
EXTRAS = ["", "", "", " and a backpack", " and a baseball cap", " and a face mask",
          " and white sneakers", " and sunglasses"]


def person(rng: random.Random) -> str:
    return (f"a {rng.choice(BUILDS)}{rng.choice(NOUNS)} in {rng.choice(TOPS)} and "
            f"{rng.choice(BOTTOMS)}{rng.choice(EXTRAS)}")


OFFICER_LAST = ["Lee", "Garcia", "Nguyen", "Patel", "Johnson", "Okafor", "Kim", "Martinez",
                "Chen", "Silva", "Brooks", "Haddad", "Ramirez", "Singh", "Walker", "Torres"]
OFFICER_FIRST_INITIAL = list("ABCDEJKLMNRST")


def officer_state(rng: random.Random, building: str, all_buildings: list[str]) -> dict[str, Any]:
    status = rng.choices(["none", "offered", "declined", "accepted"], [3, 3, 1, 5])[0]
    o: dict[str, Any] = {"status": status}
    if status in ("offered", "declined", "accepted"):
        o["name"] = f"Officer {rng.choice(OFFICER_FIRST_INITIAL)}. {rng.choice(OFFICER_LAST)}"
        o["unit"] = f"Unit {rng.randint(2, 40)}"
    if status == "accepted":
        o["eta_min"] = rng.choice([1, 2, 2, 3, 3, 4, 5, 6, 8])
    return o


def zone_state(rng: random.Random, building: str, all_buildings: list[str]) -> dict[str, Any]:
    status = rng.choices(["none", "drafted", "approved", "sent"], [3, 2, 1, 4])[0]
    z: dict[str, Any] = {"status": status}
    if status != "none":
        others = [b for b in all_buildings if b != building]
        z["buildings"] = [building] + rng.sample(others, k=min(len(others), rng.choice([0, 1, 1, 2])))
    if status == "sent":
        z["recipients"] = rng.choice([38, 64, 120, 212, 340, 415, 560, 780, 1240])
    return z


def finish(snap: dict[str, Any], rng: random.Random, all_buildings: list[str]) -> dict[str, Any]:
    return close(snap, officer_state(rng, snap["building"], all_buildings),
                 zone_state(rng, snap["building"], all_buildings))


def close(snap: dict[str, Any], officer: dict[str, Any], zone_alert: dict[str, Any]) -> dict[str, Any]:
    snap["officer"] = officer
    snap["zone_alert"] = zone_alert
    snap["console_alerted"] = True
    snap["cameras_have_audio"] = False
    return snap


# ---------------------------------------------------------------- real

SEVILLE_BUILDING = scene.BUILDING
if SEVILLE_BUILDING not in SJSU_BUILDINGS:
    SJSU_BUILDINGS.insert(0, SEVILLE_BUILDING)


def _row(t: int) -> dict[str, Any]:
    return scene.at(t)


def _descriptions(t: int) -> list[str]:
    for a, b, text in scene.DESCRIPTIONS:
        if a <= t < b:
            parts = [p.strip().rstrip(".") for p in text.replace(", and ", ". ").split(". ") if p.strip()]
            return [p[0].lower() + p[1:] for p in parts]
    d = scene.people_descriptions(_row(t))
    return [p.strip() for p in d.split(";") if p.strip()]


def _route(t: int) -> list[str]:
    hist = scene.moves(t)
    cams = ([hist[0][1]] + [b for _, _, b in hist]) if hist else [_row(t).get("camera")]
    places = [scene.PLACE.get(c, "inside the building") for c in cams]
    out: list[str] = []
    for p in places:
        if not out or out[-1] != p:
            out.append(p)
    return out


def real_snapshot(t: int, rng: random.Random) -> dict[str, Any]:
    return finish(scene_snapshot(t), rng, SJSU_BUILDINGS)


def live_snapshot(t: float, officer: dict[str, Any] | None = None,
                  zone_alert: dict[str, Any] | None = None) -> dict[str, Any]:
    """The clip second the wall shows, with the call's real officer/warning status."""
    return close(scene_snapshot(int(t)), dict(officer or {"status": "none"}),
                 dict(zone_alert or {"status": "none"}))


def scene_snapshot(t: int) -> dict[str, Any]:
    """Everything the cameras know at clip second t (no officer or warning yet)."""
    row = _row(t)
    people, weapons = scene.steady(t)
    armed = sum(weapons.values())
    place = scene.PLACE.get(row.get("camera"), "inside the building")
    snap: dict[str, Any] = {
        **SJSU,
        "building": SEVILLE_BUILDING,
        "incident": "WEAPON",
        "tracking_s": t + 5,
        "route": _route(t),
    }
    if armed > 0:
        snap["now"] = {
            "visible": True, "place": place, "people": people, "armed": armed,
            "weapons": weapons, "subjects": _descriptions(t),
            "observation": row.get("scene", ""), "movement": row.get("movement") or None,
        }
        snap["last_seen"] = None
    else:
        # Subject out of view: find the last second someone armed was visible.
        back = next((s for s in range(t - 1, -1, -1) if sum(scene.steady(s)[1].values()) > 0), None)
        snap["now"] = {"visible": False, "place": place, "people": people, "armed": 0,
                       "weapons": {}, "subjects": [], "observation": row.get("scene", ""),
                       "movement": None}
        if back is not None:
            bp, bw = scene.steady(back)
            snap["last_seen"] = {
                "place": scene.PLACE.get(_row(back).get("camera"), "inside the building"),
                "seconds_ago": t - back, "armed": sum(bw.values()), "weapons": bw,
                "subjects": _descriptions(back),
            }
        else:
            snap["last_seen"] = None
    return snap


# ---------------------------------------------------------------- synthetic

MOVES = [
    "walking toward {p}", "running toward {p}", "moving quickly toward {p}",
    "standing still near the doorway", "pacing back and forth", "walking away from the camera toward {p}",
    "heading toward {p}",
]


def _site(rng: random.Random, heldout: bool) -> tuple[str, str, list[str]]:
    if heldout:
        return rng.choice(HELDOUT_SITES)
    if rng.random() < 0.55:
        return SJSU["site"], SJSU["site_address"], SJSU_BUILDINGS
    return rng.choice(OTHER_SITES)


def synthetic_snapshot(rng: random.Random, heldout: bool = False) -> dict[str, Any]:
    site, addr, buildings = _site(rng, heldout)
    building = rng.choice(buildings)
    pool = place_pool(rng)
    route = rng.sample(pool, k=rng.choice([1, 1, 2, 2, 3]))
    incident = rng.choices(["WEAPON", "FIGHT", "MEDICAL", "THEFT"], [50, 18, 18, 14])[0]
    visible = rng.random() > 0.2
    tracking_s = rng.choice([8, 15, 25, 40, 55, 75, 95, 130, 180, 240, 320, 450, 600])

    subjects: list[str]
    weapons: dict[str, int] = {}
    involved = 0
    if incident == "WEAPON":
        armed = rng.choices([1, 2, 3, 4], [55, 25, 12, 8])[0]
        for _ in range(armed):
            w = rng.choices(["handgun", "rifle", "knife"], [60, 20, 20])[0]
            weapons[w] = weapons.get(w, 0) + 1
        kinds = [k for k, n in weapons.items() for _ in range(n)]
        rng.shuffle(kinds)
        subjects = [f"{person(rng)} with a {k}" for k in kinds][:3]
        people = armed + rng.choice([0, 0, 1, 2, 3, 5])
        obs = f"{num(people).capitalize()} {'person' if people == 1 else 'people'} visible; {num(armed)} holding {'a weapon' if armed == 1 else 'weapons'}."
    elif incident == "FIGHT":
        armed = 0
        involved = rng.choice([2, 2, 3, 4])
        subjects = [person(rng) for _ in range(min(involved, 3))]
        people = involved + rng.choice([0, 2, 4, 6])
        obs = f"{num(involved).capitalize()} people are fighting; others are standing nearby."
    elif incident == "MEDICAL":
        armed = 0
        subjects = [f"{person(rng)}, lying on the floor and not moving"]
        people = 1 + rng.choice([0, 0, 1, 2, 3])
        obs = "One person is lying on the floor and has not moved."
    else:
        armed = 0
        item = rng.choice(["a laptop", "a backpack", "a bike", "a purse", "a package"])
        subjects = [f"{person(rng)}, taking {item} and walking away"]
        people = 1 + rng.choice([0, 1, 2, 4])
        obs = f"A person picked up {item} that was left unattended and is leaving."

    nxt = rng.choice([p for p in pool if p not in route] or pool)
    movement = rng.choice(MOVES).format(p=nxt) if incident != "MEDICAL" else "not moving"
    movement = movement[0].upper() + movement[1:] + "."

    snap: dict[str, Any] = {
        "site": site, "site_address": addr, "building": building,
        "incident": incident, "tracking_s": tracking_s, "route": route,
    }
    if visible or incident == "MEDICAL":
        snap["now"] = {"visible": True, "place": route[-1], "people": people, "armed": armed,
                       "weapons": weapons, "subjects": subjects, "observation": obs,
                       "movement": movement if rng.random() > 0.15 else None}
        if involved:
            snap["now"]["involved"] = involved
        snap["last_seen"] = None
    else:
        cur = rng.choice([p for p in pool if p not in route] or pool)
        by = rng.choice([0, 0, 1, 2, 3])
        snap["now"] = {"visible": False, "place": cur, "people": by, "armed": 0, "weapons": {},
                       "subjects": [], "observation": f"{num(by).capitalize()} bystanders visible." if by else "No one visible.",
                       "movement": None}
        snap["last_seen"] = {"place": route[-1], "seconds_ago": rng.choice([5, 10, 20, 30, 45, 60, 90, 150]),
                             "armed": armed, "weapons": weapons, "subjects": subjects}
        if involved:
            snap["last_seen"]["involved"] = involved
    return finish(snap, rng, buildings)


if __name__ == "__main__":
    r = random.Random(0)
    print(json.dumps(real_snapshot(50, r), indent=1)[:1500])
    print(json.dumps(live_snapshot(255, {"status": "accepted", "name": "Officer A. Lee", "eta_min": 3}), indent=1)[:1500])
    print(json.dumps(synthetic_snapshot(r), indent=1))
