# Windows demo setup

Sentinel is an independent staged demonstration. It is not an official product. A configured weapon incident can call watsonx for a Granite Guardian check; BeeAI then decides dispatch.

The runtime shows three synchronized playback cameras. Integrations are disabled unless you explicitly configure them. This document does not deploy anything.

## Prerequisites

- Windows with Python 3.12.
- The repository at `C:\Users\<you>\Documents\Sentinel AI IBM`.
- The three demo MP4s outside Git (below).
- OpenCV is already used by the optional vision path. ffmpeg is not required for MP4 playback.

Node and `make` are optional. If `node` is on PATH, run `node web/check.mjs`. If `make` is on PATH, `make check` runs the same Python self-checks listed below. This Windows image may not have either tool. Run the Python checks directly in that case.

## Media root

Put only these files in the folder:

`C:\Users\<you>\Documents\campus_sentinel_media\feeds\final_demo_3cam_reconstructed`

- `CAM01_lobby_entrance_IN_then_OUT_339s.mp4`
- `CAM02_hallway_east_IN_then_OUT_339s.mp4`
- `CAM03_hallway_west_IN_then_OUT_339s.mp4`

`REVIEW_three_camera_handoffs_339s.mp4` is review-only. It is not an application feed.

With no `CS_MEDIA_ROOT`, the API resolves the sibling folder next to the repo:

`<repo parent>\campus_sentinel_media\feeds\final_demo_3cam_reconstructed`

Set the variable when the clips live somewhere else:

```env
CS_MEDIA_ROOT=C:\Users\<you>\Documents\campus_sentinel_media\feeds\final_demo_3cam_reconstructed
```

Do not commit MP4s, weights, logs, or `.env`.

## Startup

From the repo, in two terminals. Bind localhost only.

```powershell
cd "C:\Users\<you>\Documents\Sentinel AI IBM"
$env:PYTHONPATH = "."
python -m services.api --host 127.0.0.1 --port 8080
```

```powershell
cd "C:\Users\<you>\Documents\Sentinel AI IBM"
python -m http.server 8090 --bind 127.0.0.1 --directory web
```

Open:

`http://127.0.0.1:8090/?ws=ws://127.0.0.1:8080/ws`

That URL is the three-camera demo. The page without `?ws=` is an older offline mock and is not this route.

Stop both processes when you are done. Do not leave them running.

## Three cameras and handoffs

The wall plays CAM-01, CAM-02, and CAM-03 from `/media/cam-01`, `/media/cam-02`, and `/media/cam-03`. Cameras 4 through 6 are not feeds. Each MP4 already holds a person-free still from that same camera outside its live window, so the person is on one feed at a time.

| Time | Camera | What you see |
| --- | --- | --- |
| 00:00.0 to 01:00.0 | CAM-01 | Lobby and entrance |
| 01:00.0 to 01:57.0 | CAM-02 | East hallway |
| 01:57.0 to 03:36.6 | CAM-03 | West hallway |
| 03:36.6 to 03:50.2 | CAM-03 | West hallway return |
| 03:50.2 to 04:27.8 | CAM-02 | East hallway return |
| 04:27.8 to 05:39.8 | CAM-01 | Lobby exit |

Handoffs are at 60.0, 117.0, 216.6, 230.2, and 267.8 seconds. The clip ends at 339.8 seconds.

Weapon boxes on the recording come from `data/feeds/seville_weapon_timeline.json`. Scene-script answers come from `data/feeds/seville_scene_script.json`. Both are precomputed. YOLO tracks people on the playback when the vision process is running. Qwen is a local VLM and runs only if that runtime is configured and up. A weapon incident calls Granite Guardian on watsonx when those settings are present, then BeeAI decides whether dispatch may continue.

## Validation

```powershell
cd "C:\Users\<you>\Documents\Sentinel AI IBM"
$env:PYTHONPATH = "."
python services/brain/check.py
python services/api/check.py
python services/voice/check.py
python services/vision/osnet.py
node web/check.mjs
```

Skip a command the machine cannot run, and say so. Do not treat a skipped command as a pass.

`python services/api/check.py` checks that the three media paths respond and that cam-04 is not a feed. Confirm the review MP4 is not in `services/media_root.py` clip map.

## Optional environment

Copy `.env.example` to `.env` on the machine. Never commit `.env`.

Calls, SMS, and ElevenLabs stay off until you set the flags. Watson variables in the example are reserved names. This build does not read them.

| Variable | Role |
| --- | --- |
| `CS_SIGNALWIRE_ENABLED` | Must be `1` before any real call or SMS. Default off. |
| `CS_KILL_SWITCH` | `1` blocks the provider even if the enable flag is on. |
| `CS_SIGNALWIRE_SMS_FALLBACK_CALL` | Must be `1` before a failed SMS may place a call. Default off. |
| `SIGNALWIRE_SPACE`, `SIGNALWIRE_PROJECT_ID`, `SIGNALWIRE_API_TOKEN` | Provider credentials. |
| `SIGNALWIRE_FROM_NUMBER` | Caller ID. Alias of `SIGNALWIRE_FROM`. |
| `SIGNALWIRE_TO_NUMBER` | The only allowed destination. Alias of `CS_DEMO_TO_NUMBER`. |
| `CS_PUBLIC_BASE` | Public HTTPS origin for provider webhooks, if you enable calls later. |
| `ELEVENLABS_API_KEY` | Optional phone speech. Also set `CS_TTS_PROVIDER=elevenlabs`. |
| `ELEVENLABS_VOICE_ID` | Alias of `CS_ELEVENLABS_VOICE_ID`. |
| `CS_MEDIA_ROOT` | Optional clip folder override. |
| `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` | Unused. Do not expect a client. |

Outbound calls and SMS go only to that one verified demo number. Emergency numbers, short codes, and every other destination are rejected. The demo never calls emergency services. Logs and the UI say "verified demo handset" instead of printing a number.

## Public HTTPS later

Do not deploy from this setup. Do not open a tunnel and do not publish a port.

`100.125.217.67` is a private Tailscale-style address. It is not a public internet address.

A later public host still needs, and this repo does not provide them:

- A hostname and TLS certificate you control, with only port 443 published.
- `CS_PUBLIC_BASE` set to that `https://` origin before any real webhook.
- Secrets kept in an untracked `.env`.
- The same enable flags and the single verified demo handset. No emergency numbers.
- A supervisor so the API and the static console are not left on an open interface by accident.

Until those exist, use `http://127.0.0.1:8090/?ws=ws://127.0.0.1:8080/ws` on the machine itself.
