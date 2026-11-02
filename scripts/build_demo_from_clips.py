"""Build the demo's camera feeds and timing data from the three clips in data/clips.

One story, three cameras: the lobby clip, then the east corridor, then the
west corridor. Each output feed is the full demo length; a camera plays its
clip during its live window and holds its first/last frame otherwise (the
same "freeze-handoff" design as the original reconstructed feeds).

Writes:
  data/clips/CAM0{1,2,3}_*.mp4           synced H.264 feeds, 5 fps (gitignored)
  data/feeds/seville_option1_3cam_locked.json   cameras + live windows
  data/feeds/seville_weapon_timeline.json       weapons + holder boxes by clip time
  data/feeds/seville_scene_script.json          per second: camera, people, weapons,
                                                 armed people, scene, movement

People are counted with YOLO26s-pose (the live wall's detector). Weapons,
who holds them and the one-line scene come from DeepSeek V4.1 Flash on
Fireworks (serverless, vision), asked about every 2nd frame of each live
window; answers are cached in data/runtime/vlm_cache.json, so a rerun is free.

  python scripts/build_demo_from_clips.py            # needs FIREWORKS_API_KEY (.env)
  python scripts/build_demo_from_clips.py --feeds-only
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))
import envfile  # noqa: E402,F401  (FIREWORKS_API_KEY from .env)

CLIPS = ROOT / "data" / "clips"
FEEDS = ROOT / "data" / "feeds"
CACHE = ROOT / "data" / "runtime" / "vlm_cache.json"
FPS = 5
# Story order. `src` is a glob in data/clips; `out` is the synced feed the app plays.
# `view` tells the vision model what the camera looks at, so it can describe movement in
# building directions a dispatcher can use, not positions in the picture.
CAMERAS = [
    {"camera_id": "CAM-01", "src": "*1.53.43 PM.mp4", "out": "CAM01_lobby_entrance.mp4",
     "location": "Lobby entrance", "seville_label": "CAM5", "place": "the ground-floor lobby",
     "view": "the ground-floor lobby, looking at the glass entrance doors. Walking toward the doors means heading "
             "for the exit; walking away from the doors means heading into the building, toward the corridors"},
    {"camera_id": "CAM-02", "src": "*1.52.58 PM.mp4", "out": "CAM02_hallway_east.mp4",
     "location": "Yellow hallway east", "seville_label": "CAM1",
     "place": "the east corridor on the ground floor",
     "view": "a long corridor, looking down it from the lobby end. Walking away from the camera means going further "
             "down the corridor, deeper into the building; walking toward it means heading back toward the lobby"},
    {"camera_id": "CAM-03", "src": "*1.52.32 PM.mp4", "out": "CAM03_hallway_west.mp4",
     "location": "Yellow hallway west", "seville_label": "CAM1",
     "place": "the west corridor near the stairwell",
     "view": "a long corridor near the stairwell. Walking away from the camera means going further down the "
             "corridor toward the stairwell; walking toward it means heading back toward the east corridor"},
]
VLM = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
VLM_EVERY = 2  # frames: 2.5 samples per second
PROMPT = (
    "Security camera frame. Find every weapon visible: handgun, rifle (any long gun) or knife. Look closely at every "
    "hand; small dark handguns held low count too. Reply ONLY JSON: {\"weapons\":[{\"type\":\"handgun|rifle|knife\","
    "\"bbox_2d\":[x1,y1,x2,y2],\"holder_bbox_2d\":[x1,y1,x2,y2],\"holder\":\"male|female, build, top, bottom, hair\"}]} "
    "Coordinates on a 0-1000 scale. Empty weapons list if none."
)
# Once per second: what is happening, in words a dispatcher can repeat.
SCENE_PROMPT = (
    "Security camera in a university building. This camera shows {view}. Describe the frame for a 911 dispatcher "
    "who cannot see it. Never use positions in the picture: no left, right, foreground, background, top, bottom, "
    "corner, frame, or toward/away from the camera. Use the building instead: doors, entrance, exit, corridor, "
    "lobby, stairwell. Reply ONLY JSON: {{\"scene\":\"one plain sentence: how many people and what they are doing\","
    "\"movement\":\"one plain sentence: where they are heading in the building, or No one is moving.\"}}"
)
# Picture-relative wording a dispatcher can't use; such a sentence is dropped.
PICTURE_WORDS = re.compile(r"\b(left|right|foreground|background|top|bottom|upper|lower|corner|edge|frame|screen|"
                           r"camera|viewer)\b", re.I)


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _source(cam: dict[str, Any]) -> Path:
    hits = sorted(CLIPS.glob(cam["src"]))
    if not hits:
        sys.exit(f"{cam['camera_id']}: no clip matching {cam['src']} in {CLIPS}")
    return hits[0]


def _frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


# ---------------------------------------------------------------- feeds

def build_feeds() -> tuple[list[dict[str, Any]], int, int, int]:
    """Synced feeds; returns (cameras with start/n, total frames, width, height)."""
    start = 0
    for cam in CAMERAS:
        cam["path"] = _source(cam)
        cam["n"] = _frames(cam["path"])
        cam["start"] = start
        start += cam["n"]
    total = start
    cap = cv2.VideoCapture(str(CAMERAS[0]["path"]))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # What each feed was built from: an unchanged feed is not rewritten, so rebuilding
    # only the timing data never disturbs a running API.
    stamp_path = CLIPS / ".feeds.json"
    stamps = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.exists() else {}
    for cam in CAMERAS:
        out = CLIPS / cam["out"]
        stamp = {"src": cam["path"].name, "size": cam["path"].stat().st_size, "start": cam["start"],
                 "n": cam["n"], "total": total, "w": w, "h": h}
        if out.exists() and stamps.get(cam["out"]) == stamp:
            print(f"{cam['camera_id']}: {out.name} up to date")
            continue
        # Written next to the feed and swapped in whole: a reader never sees a half-written file.
        tmp = out.with_name(out.stem + ".part.mp4")
        cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
               "-crf", "23", "-g", str(FPS), "-movflags", "+faststart", str(tmp)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        cap = cv2.VideoCapture(str(cam["path"]))
        ok, first = cap.read()
        frame = cv2.resize(first, (w, h))
        for _ in range(cam["start"]):  # frozen until its live window
            proc.stdin.write(frame.tobytes())
        written = cam["start"]
        while ok:
            proc.stdin.write(cv2.resize(frame, (w, h)).tobytes())
            written += 1
            ok, nxt = cap.read()
            if ok:
                frame = nxt
        while written < total:  # frozen after it
            proc.stdin.write(cv2.resize(frame, (w, h)).tobytes())
            written += 1
        cap.release()
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit(f"ffmpeg failed for {out}")
        try:
            os.replace(tmp, out)
        except PermissionError:
            sys.exit(f"{out.name} is open in another program (the API?). Stop it, then run this again.")
        stamps[cam["out"]] = stamp
        stamp_path.write_text(json.dumps(stamps, indent=1), encoding="utf-8")
        print(f"{cam['camera_id']}: {cam['path'].name} -> {out.name}  live {cam['start'] / FPS:.1f}-"
              f"{(cam['start'] + cam['n']) / FPS:.1f} s of {total / FPS:.1f} s")
    return CAMERAS, total, w, h


def write_manifest(total: int) -> None:
    feed = {
        "id": "first_witness_3cam_clips",
        "version": "clips-v1",
        "purpose": "Three-camera demo feed built from data/clips (synced freeze-handoff)",
        "duration_s": round(total / FPS, 1),
        "fps": FPS,
        "path_story": "Lobby -> Hall east -> Hall west",
        "source": "US Mock Attack (University of Seville) CCTV clips in data/clips; built by "
                  "scripts/build_demo_from_clips.py",
        "cameras": [{
            "camera_id": c["camera_id"], "file": c["out"], "location": c["location"],
            "seville_label": c["seville_label"], "source_clip": c["path"].name,
            "live_windows_s": [[round(c["start"] / FPS, 1), round((c["start"] + c["n"]) / FPS, 1)]],
        } for c in CAMERAS],
    }
    (FEEDS / "seville_option1_3cam_locked.json").write_text(json.dumps(feed, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- vision model

def _ask(job: tuple[str, str, bytes]) -> tuple[str, dict[str, Any] | None]:
    key, prompt, jpg = job
    body = {"model": VLM, "temperature": 0, "max_tokens": 500, "reasoning_effort": "none", "messages": [{
        "role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
            {"type": "text", "text": prompt}]}]}
    req = urllib.request.Request("https://api.fireworks.ai/inference/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={
                                     "Authorization": "Bearer " + os.environ["FIREWORKS_API_KEY"],
                                     "Content-Type": "application/json"})
    for _ in range(3):
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=60).read())
            text = re.sub(r"<think>.*?</think>", "", out["choices"][0]["message"]["content"] or "", flags=re.S)
            m = re.search(r"\{.*\}", text, re.S)
            return key, json.loads(m.group(0)) if m else {"weapons": []}
        except Exception:  # noqa: BLE001
            continue
    return key, None


def vlm_pass(w: int, h: int) -> dict[str, Any]:
    """Weapons on every VLM_EVERY-th frame; scene + movement once per second."""
    cache: dict[str, Any] = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    jobs = []
    for cam in CAMERAS:
        cap = cv2.VideoCapture(str(cam["path"]))
        scene_prompt = SCENE_PROMPT.format(view=cam["view"])
        for i in range(cam["n"]):
            ok, bgr = cap.read()
            if not ok:
                break
            # Keyed by source clip, so swapping a camera's clip never reuses labels.
            want = []
            if i % VLM_EVERY == 0 and f"{cam['path'].name}:{i}" not in cache:
                want.append((f"{cam['path'].name}:{i}", PROMPT))
            if i % FPS == 0 and f"scene:{cam['camera_id']}:{cam['path'].name}:{i}" not in cache:
                want.append((f"scene:{cam['camera_id']}:{cam['path'].name}:{i}", scene_prompt))
            if not want:
                continue
            jpg = cv2.imencode(".jpg", cv2.resize(bgr, (w, h)), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            jobs += [(key, prompt, jpg) for key, prompt in want]
        cap.release()
    print(f"vision model: {len(jobs)} new frames ({len(cache)} cached)")
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for key, ans in ex.map(_ask, jobs):
            if ans is not None:
                cache[key] = ans
            done += 1
            if done % 100 == 0:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(cache), encoding="utf-8")
                print(f"  {done}/{len(jobs)}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def _px(box: Any, w: int, h: int) -> list[int] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    out = [round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)]
    return out if out[2] > out[0] and out[3] > out[1] else None


def _weapons(ans: dict[str, Any], w: int, h: int) -> list[dict[str, Any]]:
    out = []
    for wpn in ans.get("weapons") or []:
        kind = str(wpn.get("type", "")).lower()
        kind = "rifle" if kind in ("shotgun", "long gun") else kind
        box = _px(wpn.get("bbox_2d"), w, h)
        if kind not in ("handgun", "rifle", "knife") or box is None:
            continue
        out.append({"type": kind, "box": box, "holder": _px(wpn.get("holder_bbox_2d"), w, h) or box,
                    "person": str(wpn.get("holder") or "").strip()})
    return out


# ---------------------------------------------------------------- people (YOLO)

def people_per_second(total: int) -> dict[int, int]:
    from ultralytics import YOLO

    from services.vision.detector import DEFAULT_PT

    model = YOLO(str(DEFAULT_PT))
    out: dict[int, int] = {}
    for cam in CAMERAS:
        cap = cv2.VideoCapture(str(cam["path"]))
        for i in range(cam["n"]):
            ok, bgr = cap.read()
            if not ok:
                break
            if i % FPS:
                continue
            t = (cam["start"] + i) // FPS
            res = model.predict(bgr, conf=0.25, verbose=False)[0]
            out[t] = len(res.boxes)
        cap.release()
    return out


# ---------------------------------------------------------------- outputs

def write_timeline_and_script(cache: dict[str, Any], total: int, w: int, h: int, people: dict[int, int]) -> None:
    timeline: dict[str, list[dict[str, Any]]] = {}
    per_sec: dict[int, list[tuple[str, dict[str, Any], list[dict[str, Any]]]]] = {}
    for cam in CAMERAS:
        rows = []
        for i in range(0, cam["n"], VLM_EVERY):
            ans = cache.get(f"{cam['path'].name}:{i}")
            if ans is None:
                continue
            gi = cam["start"] + i
            wpns = _weapons(ans, w, h)
            rows.append({"frame": gi, "t": round(gi / FPS, 2),
                         "weapons": [{k: v for k, v in x.items() if k != "person"} for x in wpns]})
            per_sec.setdefault(gi // FPS, []).append((cam["camera_id"], ans, wpns))
        # Clear at the end of the live window, so the frozen frame shows no weapon.
        rows.append({"frame": cam["start"] + cam["n"], "t": round((cam["start"] + cam["n"]) / FPS, 2), "weapons": []})
        timeline[cam["camera_id"]] = rows
    (FEEDS / "seville_weapon_timeline.json").write_text(json.dumps({
        "source": f"{VLM} on every {VLM_EVERY}nd frame of each live window (scripts/build_demo_from_clips.py)",
        "width": w, "height": h, "cameras": timeline}, indent=1) + "\n", encoding="utf-8")

    place = {c["camera_id"]: c["place"] for c in CAMERAS}
    seconds = []
    last: dict[str, str] = {"scene": "", "movement": ""}
    dropped = 0
    for t in range(total // FPS + (1 if total % FPS else 0)):
        c = next(c for c in CAMERAS if c["start"] <= t * FPS < c["start"] + c["n"])
        cam = c["camera_id"]
        local = max(0, (t * FPS - c["start"]) // FPS * FPS)
        said = cache.get(f"scene:{cam}:{c['path'].name}:{local}") or {}
        for field in ("scene", "movement"):
            text = str(said.get(field) or "").strip()
            if text and not PICTURE_WORDS.search(text):
                last[field] = text
            elif text:
                dropped += 1  # picture wording: keep the last good sentence instead
        samples = [s for s in per_sec.get(t, []) if s[0] == cam]
        weapons: dict[str, int] = {}
        armed: list[dict[str, str]] = []
        for _, ans, wpns in samples:
            counts: dict[str, int] = {}
            for x in wpns:
                counts[x["type"]] = counts.get(x["type"], 0) + 1
            for k, n in counts.items():
                weapons[k] = max(weapons.get(k, 0), n)
            if len(wpns) >= len(armed):
                armed = [{"weapon": x["type"], "person": x["person"]} for x in wpns]
        n_people = max(people.get(t, 0), sum(weapons.values()))
        seconds.append({"t": t, "camera": cam, "place": place[cam], "people": n_people, "weapons": weapons,
                        "armed": armed, "scene": last["scene"], "movement": last["movement"]})
    (FEEDS / "seville_scene_script.json").write_text(json.dumps({
        "source": f"YOLO26s-pose people counts; weapons, holders, scene and movement from {VLM} "
                  "(scripts/build_demo_from_clips.py)", "seconds": seconds}, indent=1) + "\n", encoding="utf-8")
    armed_s = sum(1 for s in seconds if s["weapons"])
    print(f"scene script: {len(seconds)} s, weapons visible in {armed_s} s; "
          f"timeline: {sum(len(v) for v in timeline.values())} annotated frames; "
          f"{dropped} sentences dropped for picture wording")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feeds-only", action="store_true", help="write the synced feeds and manifest, skip the models")
    for cam, flag in zip(CAMERAS, ("--lobby", "--east", "--west")):
        ap.add_argument(flag, metavar="CLIP", help=f"{cam['camera_id']} source: a file name or glob in data/clips "
                                                    f"(default {cam['src']!r})")
    ap.add_argument("--west-label", help="Seville camera label of the --west clip, e.g. CAM7")
    args = ap.parse_args()
    for cam, flag in zip(CAMERAS, ("lobby", "east", "west")):
        if getattr(args, flag):
            cam["src"] = getattr(args, flag)
    if args.west_label:
        CAMERAS[2]["seville_label"] = args.west_label
    _, total, w, h = build_feeds()
    write_manifest(total)
    if args.feeds_only:
        return
    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit("Set FIREWORKS_API_KEY (or put it in .env) for the vision-model pass.")
    cache = vlm_pass(w, h)
    people = people_per_second(total)
    write_timeline_and_script(cache, total, w, h, people)


if __name__ == "__main__":
    main()
