# Windows demo setup

First Witness is a staged demonstration on recorded CCTV, not an emergency service. It never calls 911: live calls go only to one verified demo handset, and only when you turn calls on.

## Prerequisites

- Windows with Python 3.11 or 3.12 (the voice code imports `audioop`, removed in 3.13).
- An NVIDIA GPU is optional. YOLO runs on CUDA when PyTorch's CUDA build and the matching CUDA `torchvision` are installed; otherwise set `CS_VISION_DEVICE=cpu`.
- Node is optional (`node web/check.mjs`). `make` is optional; every target's command is in the `Makefile`.
- Keep a few GB free on the system drive: model caches and temporary files land there.

```powershell
pip install -r requirements.txt
```

`requirements.txt` pins `numpy==2.3.5`. Install `fireworks-ai[training-sdk]`, not the full `[training]` extra, which upgrades NumPy and pulls torch/transformers/wandb. Install `torchvision` from the PyTorch index that matches your torch build (for example `--index-url https://download.pytorch.org/whl/cu128 --no-deps`), or YOLO's box filtering fails on CUDA.

## Video and data

Put the three source clips in `data\clips\` (gitignored). Each camera's file pattern is set in `CAMERAS` at the top of `scripts\build_demo_from_clips.py`. Then build the feeds and timing data once:

```powershell
python scripts\build_demo_from_clips.py
```

It writes `CAM01_lobby_entrance.mp4`, `CAM02_hallway_east.mp4` and `CAM03_hallway_west.mp4` (three synced 290 s feeds, H.264, 5 fps) and the camera windows, weapon timeline and per-second scene script in `data\feeds\`. The weapon labels come from DeepSeek V4.1 Flash on Fireworks, so it needs `FIREWORKS_API_KEY` in `.env`; answers are cached in `data\runtime\`, so rebuilding is free. `--feeds-only` skips the model.

| Time | Camera | What you see |
| --- | --- | --- |
| 0:00.0 to 2:10.8 | CAM-01 | Lobby and entrance: the group gathers, weapons out, a man crosses with a rifle |
| 2:10.8 to 3:49.0 | CAM-02 | East hallway |
| 3:49.0 to 4:49.8 | CAM-03 | West hallway (the same Seville camera as CAM-02) |

Outside its window each camera holds its first or last frame, so the action is on one feed at a time.

YOLO26s-pose weights go in `services\vision\weights\yolo26s-pose.pt` (gitignored); ultralytics downloads them on first use, or run `python services\vision\pull_weights.py`.

## Startup

From the repo, in two terminals. Bind localhost only.

```powershell
$env:PYTHONPATH = "."
python -m services.api --host 127.0.0.1 --port 8080
```

```powershell
python -m http.server 8090 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8090/?ws=ws://127.0.0.1:8080/ws` (add `&nosplash=1` to skip the intro). The page without `?ws=` is an offline mock, not this demo.

The API reads `.env`. Look for these startup lines:

- `[vision-bridge] started cams=[...]`: live vision is on. It turns on by itself when the three feeds and the YOLO weights are present; `CS_VISION_SEVILLE=0` forces it off.
- `[api] call model: training-pool ...`: the fine-tuned call agent is configured. It answers the dispatcher's questions on a live phone call. The call's own report and updates (`services/voice/briefing.py`) need no model.

Stop both processes when you are done.

## Validation

```powershell
$env:PYTHONPATH = "."
python services/brain/check.py
python services/api/check.py
python services/voice/check.py
python -m services.voice.call_model
python -m services.voice.briefing
python services/vision/check.py
node web/check.mjs
```

Skip a command the machine cannot run, and say so. A skipped command is not a pass.

## Environment

Copy `.env.example` to `.env` on the machine. Never commit `.env`.

| Variable | Role |
| --- | --- |
| `FIREWORKS_API_KEY` | Fireworks key: the call model, the weapon-labelling pass, the benchmarks. |
| `CS_CALL_MODEL_BACKEND`, `CS_CALL_MODEL`, `CS_CALL_MODEL_SNAPSHOT` | The fine-tuned call agent on Fireworks' serverless training pool (values in `.env.example`). |
| `CS_VISION_SEVILLE` | Live vision: unset = automatic, `0` off, `1` on. |
| `CS_VISION_DEVICE` | `cuda:0` (default) or `cpu`. |
| `CS_MEDIA_ROOT` | Feed folder override. Default `data\clips`. |
| `CS_SIGNALWIRE_ENABLED` | Must be `1` before any real call or SMS. Default off. |
| `CS_KILL_SWITCH` | `1` blocks the provider even if the enable flag is on. |
| `SIGNALWIRE_SPACE`, `SIGNALWIRE_PROJECT_ID`, `SIGNALWIRE_API_TOKEN` | Provider credentials. |
| `SIGNALWIRE_FROM_NUMBER`, `SIGNALWIRE_TO_NUMBER` | Caller ID and the only allowed destination. |
| `CS_PUBLIC_BASE` | Public HTTPS origin for provider webhooks, needed only for live calls. |
| `ELEVENLABS_API_KEY`, `CS_TTS_PROVIDER=elevenlabs` | Phone speech. |

Outbound calls and SMS go only to the one verified demo number. Emergency numbers, short codes and every other destination are rejected.

## Public HTTPS

Live phone calls need SignalWire to reach `/signalwire/*`, which needs a public HTTPS origin. This guide doesn't set one up. Until you do, keep calls off and use the console on the machine itself.
