# IBM Sentinel AI narration

Independent staged demonstration. Not an official IBM product. No IBM logo, no IBM Cloud, no Granite, and no watsonx. Do not generate a video from this script. Read it while you screen-record the localhost demo.

Product on screen: IBM Sentinel AI. Palette: IBM blue `#0f62fe` on the existing carbon and white console.

## Before you record

- Start the API on `127.0.0.1:8080` and the console on `127.0.0.1:8090`. See `docs/WINDOWS_DEMO_SETUP.md`.
- Open `http://127.0.0.1:8090/?ws=ws://127.0.0.1:8080/ws`.
- Confirm three tiles only: CAM-01 lobby, CAM-02 east hallway, CAM-03 west hallway. Each tile says PLAYBACK.
- Confirm the header says the demo is staged and not an official IBM product.
- Leave SignalWire and ElevenLabs off unless you have already decided to use the verified demo handset. Say "simulated" if they are off.
- Do not dial emergency services. Do not show a phone number. Say "verified demo handset" if you mention the line.
- Record the whole 339.8 seconds, or cut on the beats below without moving the handoff times.

## Narration

**00:00.0.** Three cameras. One building. One person. IBM Sentinel AI is a staged security console: playback video, a local detector, and a script you can audit.

**00:08.** Camera 1 is the lobby. The other two cameras hold a still from their own rooms. The person is not copied onto a second feed.

**00:20.** Detection on this recording is YOLO person tracking when that process is running. The weapon marks are a precomputed timeline, not a live weapon model. If a local Qwen VLM is up, it can classify a clip. If it is not up, the console must not pretend otherwise.

**00:45.** The guard panel is this dashboard. It is IBM-inspired in color and name. It is not an IBM service.

**01:00.0. Handoff.** The person leaves the lobby and enters the east hallway. Camera 2 plays. Camera 1 returns to its own empty still.

**01:57.0. Handoff.** East hallway to the west hallway. Camera 3 plays. Say the place, not a camera id, if you are voicing the dispatch line.

**02:20.** Open the simulated dispatch surface. The label says simulated. It does not say a responder is connected. The transcript is staged. Scene-script lines are precomputed answers for this recording.

**03:00.** The broadcast control records a staged message. It does not notify an audience. There is no campus-wide alert in this build. SMS, if you enable it later, goes only to the verified demo handset, and the UI still must not claim the building was warned.

**03:36.6.** Camera 3 continues through the return toward the east side. Same camera, second window. Do not cut in another angle.

**03:50.2. Handoff.** Back to the east hallway on Camera 2.

**04:27.8. Handoff.** Back to the lobby on Camera 1. The person exits the way they came.

**05:20.** If a voice call is on screen, it is either fully simulated or a call you explicitly enabled to the verified demo handset. Never emergency services. Never a live claim when the provider is off.

**05:39.8.** End. One person, three cameras, no duplicate body on a mirror tile. Independent staged demonstration. Not an official IBM product.

## Where each surface appears

| Beat | Surface | What to say it is |
| --- | --- | --- |
| 00:00 to 01:00 | CAM-01 playback, CAM-02 and CAM-03 stills | Synchronized files, not live CCTV |
| About 00:20 | Boxes on the active camera | YOLO on playback, plus a precomputed weapon timeline |
| Throughout | Header and system page | IBM Sentinel AI guard panel, staged |
| After the first alert | Dispatch and transcript | Simulated, or an explicitly enabled demo handset |
| When you open broadcast | Broadcast dialog | Staged record only. Nobody was notified |
| Only if you enabled SMS | Handset path | Verified demo handset. No number on screen |
| Only if you enabled voice | Call panel | SIM, not a connected emergency line |

## Screen-recording checklist

- Three tiles, labeled CAM-01, CAM-02, and CAM-03. No fourth live picture of the same person.
- PLAYBACK on the tiles. SIM on a dispatch timer. No LIVE badge on playback or on a mock call.
- Header disclosure visible at least once: staged, not an official IBM product.
- System page visible once: precomputed weapon timeline, local Qwen only if configured, dispatch simulated unless enabled.
- Handoffs land on the times above. Do not slip the cut so two cameras show the person.
- Broadcast result text does not say an audience was notified.
- No phone number, API key, or `.env` value in the frame.
- End card or closing line repeats the disclosure.

## Honest inventory

- Real: the three MP4 files, the locked handoff times, and this console.
- Precomputed: weapon timeline and scene-script answers.
- Simulated: dispatch transcript, broadcast, and the call script when no provider is enabled.
- Mocked: the offline page without `?ws=`, which is not this three-camera route.
- Unused: watsonx variables. They are reserved names only.
