"""Self-check for voice forced/demo path — no ASR/TTS models."""

from __future__ import annotations

import asyncio
import base64
import threading
import time
import sys
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts import IncidentClass, IncidentState, Severity, utcnow  # noqa: E402
from contracts.incident import IncidentRecord, TimelineEvent  # noqa: E402
from services.brain.call_brief import assemble_call_brief  # noqa: E402
from services.voice.agent import VoiceAgent  # noqa: E402
from services.voice import agent as agent_mod  # noqa: E402


def _check_signalwire_request() -> None:
    from services.voice import signalwire_bridge
    from services.voice.signalwire_bridge import SignalWireConfig, build_call_request

    cfg = SignalWireConfig(
        space="example.signalwire.com",
        project_id="project-id",
        api_token="secret-token",
        from_number="+12025550123",
        to_number="+14085550123",
        public_base="https://demo.example.com",
        enabled=True,
    )
    req = build_call_request(cfg, "inc 1")
    assert req.full_url == (
        "https://example.signalwire.com/api/laml/2010-04-01/Accounts/"
        "project-id/Calls.json"
    )
    assert parse_qs(req.data.decode()) == {
        "To": ["+14085550123"],
        "From": ["+12025550123"],
        "Url": ["https://demo.example.com/signalwire/voice?incident_id=inc+1"],
        "Method": ["POST"],
    }
    auth = req.get_header("Authorization")
    assert auth == "Basic " + base64.b64encode(b"project-id:secret-token").decode()

    class _Health:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"ok": true}'

    original = signalwire_bridge.urlopen
    signalwire_bridge.urlopen = lambda *_args, **_kwargs: _Health()
    try:
        assert signalwire_bridge.service_ready("http://127.0.0.1:8093/transcribe") is True
        assert signalwire_bridge.service_ready("") is False
    finally:
        signalwire_bridge.urlopen = original



def _check_signalwire_gate() -> None:
    """Enable flag, destination allowlist, and disabled SMS fallback. No network."""
    import json
    import os

    from services.voice import signalwire_bridge as sw

    keys = [
        "CS_SIGNALWIRE_ENABLED",
        "CS_KILL_SWITCH",
        "CS_SIGNALWIRE_SMS_FALLBACK_CALL",
        "SIGNALWIRE_SPACE",
        "SIGNALWIRE_PROJECT_ID",
        "SIGNALWIRE_API_TOKEN",
        "SIGNALWIRE_FROM",
        "SIGNALWIRE_FROM_NUMBER",
        "CS_DEMO_TO_NUMBER",
        "SIGNALWIRE_TO_NUMBER",
        "CS_PUBLIC_BASE",
        "ELEVENLABS_VOICE_ID",
        "CS_ELEVENLABS_VOICE_ID",
    ]
    saved = {key: os.environ.get(key) for key in keys}

    def clear() -> None:
        for key in keys:
            os.environ.pop(key, None)

    try:
        clear()
        os.environ["SIGNALWIRE_SPACE"] = "example.signalwire.com"
        os.environ["SIGNALWIRE_PROJECT_ID"] = "project-id"
        os.environ["SIGNALWIRE_API_TOKEN"] = "secret-token"
        os.environ["SIGNALWIRE_FROM_NUMBER"] = "+12025550123"
        os.environ["SIGNALWIRE_TO_NUMBER"] = "+14085550123"
        os.environ["CS_PUBLIC_BASE"] = "https://demo.example.com"
        assert sw.load_config() is None
        assert sw.load_sms_config() is None
        os.environ["CS_SIGNALWIRE_ENABLED"] = "1"
        sms_cfg = sw.load_sms_config()
        assert sms_cfg is not None and sms_cfg.to_number == "+14085550123"
        assert sms_cfg.from_number == "+12025550123"
        assert sw.load_config() is not None
        os.environ.pop("SIGNALWIRE_TO_NUMBER", None)
        os.environ["CS_DEMO_TO_NUMBER"] = "911"
        try:
            sw.load_sms_config()
            raised = False
        except ValueError:
            raised = True
        assert raised
        os.environ["CS_DEMO_TO_NUMBER"] = "12345"
        try:
            sw.load_sms_config()
            raised_short = False
        except ValueError:
            raised_short = True
        assert raised_short
        os.environ.pop("CS_DEMO_TO_NUMBER", None)
        os.environ["SIGNALWIRE_TO_NUMBER"] = "+14085550123"
        good = sw.SignalWireConfig(
            "example.signalwire.com",
            "project-id",
            "secret-token",
            "+12025550123",
            "+14085550123",
            "https://demo.example.com",
            True,
        )
        other = sw.SignalWireConfig(
            "example.signalwire.com",
            "project-id",
            "secret-token",
            "+12025550123",
            "+19995550100",
            "https://demo.example.com",
            True,
        )

        def boom(*_args, **_kwargs):
            raise AssertionError("network")

        original = sw.urlopen
        sw.urlopen = boom
        try:
            blocked = sw.send_sms("staged", config=other)
            assert blocked["ok"] is False and blocked["reason"] == "not_allowlisted"
            fallback = sw.sos_call("staged", config=good)
            assert fallback["reason"] == "sms_fallback_call_disabled"
        finally:
            sw.urlopen = original
        os.environ["CS_SIGNALWIRE_ENABLED"] = "0"
        assert sw.load_sms_config() is None
        blob = json.dumps(sw.status())
        assert "secret-token" not in blob
        assert "+1408" not in blob and "+1202" not in blob
        from services.voice.elevenlabs import _voice_id

        os.environ["CS_ELEVENLABS_VOICE_ID"] = "legacy-voice"
        assert _voice_id() == "legacy-voice"
        os.environ["ELEVENLABS_VOICE_ID"] = "primary-voice"
        assert _voice_id() == "primary-voice"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _main() -> None:
    _check_signalwire_request()
    _check_signalwire_gate()
    events: list[object] = []

    async def pub(ev: object) -> None:
        events.append(ev)

    now = utcnow()
    rec = IncidentRecord(
        incident_id="inc-voice-1",
        track_id="t-1",
        camera_id="cam-01",
        peak_ts=now,
        class_token=IncidentClass.WEAPON,
        class_logprob_calibrated=0.0,
        router_score=0.9,
        fused_prob=0.92,
        severity=Severity.SEVERE,
        description="weapon on cam-01",
        location_text="Lobby entrance",
        person_description="Adult, dark jacket, long firearm visible.",
        state=IncidentState.ALERTED,
        clip_uri="",
        created_at=now,
        updated_at=now,
        timeline=[TimelineEvent(ts=now, state=IncidentState.NEW, note="created")],
    )
    brief = assemble_call_brief(rec)
    agent = VoiceAgent(pub)
    script = [
        agent_mod.CallScriptStep(0.05, "sentinel", "reporting"),
        agent_mod.CallScriptStep(0.05, "sentinel", tool="lookup_location"),
        agent_mod.CallScriptStep(0.05, "sentinel", tool="get_person_description"),
        agent_mod.CallScriptStep(0.2, "dispatcher", "Copy. Stay on the line."),
    ]
    await agent.start_call(rec, brief, script=script)
    assert agent._task is not None
    await asyncio.sleep(0.08)
    await agent.notify_whereabouts(
        "cam-02", "Academic Walk east corridor (floor 1; near 4th St)"
    )
    await agent._task
    types = [getattr(e, "type", None) for e in events]
    assert "call.transcript_delta" in types
    assert "tool.call_live" in types
    assert any("reporting" in getattr(e, "text", "") for e in events)
    spoken_blob = " ".join(getattr(e, "text", "") for e in events).lower()
    assert "simulat" not in spoken_blob and "never emergency" not in spoken_blob
    assert any("the east corridor" in getattr(e, "text", "") for e in events)
    assert any(
        getattr(e, "scenario_id", "").startswith("security_alert:") for e in events
    )

    class _Qwen:
        calls = 0

        def answer_dispatcher(self, facts, question):
            self.calls += 1
            self.last_facts = facts
            assert facts["camera_id"] == "cam-02"
            if question == "What color is the door?":
                return "The door appears gray. I cannot verify whether it is locked."
            if question == "What is happening near the door?":
                return "The person is moving east, but I cannot confirm their destination."
            if question == "Is there smoke?":
                return "I cannot confirm that from the camera."
            return "The cameras do not confirm that detail."

    live_events: list[object] = []

    async def live_pub(ev: object) -> None:
        live_events.append(ev)

    qwen = _Qwen()
    live = VoiceAgent(live_pub, zrt=qwen)  # type: ignore[arg-type]
    live.use_model = False  # keyword rules here; the call model path is checked below
    await live.start_live_call(rec, assemble_call_brief(rec))
    opener = getattr(live_events[-2], "text", "")
    assert "MacQuarrie Hall" in opener and "IBM" not in opener
    assert "armed people" in opener.lower()
    low_open = opener.lower()
    assert "simulat" not in low_open and "staged" not in low_open
    assert "never emergency" not in low_open and "demo-only" not in low_open and "psap" not in low_open
    # Fixed questions answer instantly from the scene script at the clip second.
    live.clock = lambda: 20.0
    ask = lambda q: live.answer_dispatcher(rec.incident_id, q)  # noqa: E731
    assert "armed people" in (await ask("911, what is your emergency?")).lower()
    assert "First Witness" in await ask("Where are you calling from?")
    assert "MacQuarrie Hall, One Washington Square" in await ask("What is the exact address?")
    assert "IBM" not in await ask("What is the exact address?")
    # Numbers and clothing come from the footage at that second (data/feeds/seville_scene_script.json).
    assert await ask("How many people do you see?") == "Eight people in the ground-floor lobby; two of them armed."
    assert "handgun" in await ask("What weapons do they have?")
    assert "light blue t-shirt" in await ask("Describe the persons")
    assert "can't verify injuries" in await ask("Is anyone harmed?")
    live.clock = lambda: 250.0
    assert "west corridor" in await ask("Where are they now?")
    assert "lobby to the east corridor to the west corridor" in await ask("Where did they move?")
    # Camera handoff on a live call: place names only, from -> to.
    event_count = len(live_events)
    await live.notify_whereabouts("cam-01", assemble_call_brief(rec).address)
    assert len(live_events) == event_count
    await live.notify_whereabouts("cam-02", assemble_call_brief(rec).address)
    spoken = [getattr(e, "text", "") for e in live_events[event_count:]]
    assert (
        "Update: they have moved from the ground-floor lobby into the east corridor on the ground floor."
        in spoken
    ), spoken
    assert not any("cam-" in line or "curb" in line or "(" in line for line in spoken)
    # An unmatched question is answered from this second of the scene script.
    assert await ask("Is the door locked?") == "I don't see that on the current camera."
    assert qwen.calls == 1 and qwen.last_facts["place"] == "MacQuarrie Hall"
    assert qwen.last_facts["clip_second"] == 250 and qwen.last_facts["camera_id"] == "cam-02"
    assert qwen.last_facts["scene"].get("camera")
    event_count = len(live_events)
    assert await ask("Okay.") == ""
    assert await ask("really go and all the pressure clear.") == ""
    assert len(live_events) == event_count and qwen.calls == 1

    # Call model path: FACTS for this second + officer status, through the guard.
    seen: list[tuple[dict, list, str]] = []
    replies = iter([
        "No officer has accepted yet.",
        "An officer is on the way.",  # promise without an accept: the guard blocks it
        None,  # timeout/error: keyword rules answer
        "Officer A. Lee accepted and is about three minutes away.",
    ])

    def fake_answer(facts, history, question, timeout_s=None):
        seen.append((facts, list(history), question))
        return next(replies)

    saved_cm = (agent_mod.call_model.configured, agent_mod.call_model.answer)
    agent_mod.call_model.configured = lambda: True  # type: ignore[assignment]
    agent_mod.call_model.answer = fake_answer  # type: ignore[assignment]
    try:
        live.use_model = True
        assert await ask("Is anyone coming?") == "No officer has accepted yet."
        facts, history, _ = seen[-1]
        assert facts["officer"] == {"status": "none"} and facts["building"] == "MacQuarrie Hall"
        assert history and history[-1][0] == "First Witness"
        assert await ask("Are officers on the way?") == "I can't confirm that."
        blocked = live.audit.entries(rec.incident_id)
        assert blocked and blocked[-1].action == "guard_blocked" and "without accept" in blocked[-1].reason
        assert "west corridor" in await ask("Where are they now?")
        live.set_officer("offered", name="A. Lee")
        assert live.officer == {"status": "offered", "name": "Officer A. Lee"}
        live.set_officer("accepted", name="A. Lee", eta_min=3)
        assert "Officer A. Lee" in await ask("Is anyone coming?")
        assert seen[-1][0]["officer"] == {"status": "accepted", "name": "Officer A. Lee", "eta_min": 3}
        assert len(seen[-1][1]) == 4  # at most two earlier exchanges, as in training
        try:
            live.set_officer("arrived")
            raise AssertionError("unknown officer status accepted")
        except ValueError:
            pass
    finally:
        agent_mod.call_model.configured, agent_mod.call_model.answer = saved_cm
        live.use_model = False

    # On-screen call: the model answers the dispatcher (asked as the question
    # appears, with the longer on-screen timeout); the scripted line is the fallback.
    asked: list[tuple[str, float | None]] = []

    def screen_answer(facts, history, question, timeout_s=None):
        asked.append((question, timeout_s))
        return "They are in the ground-floor lobby." if question.startswith("How many") else None

    screen_events: list[object] = []

    async def screen_pub(ev: object) -> None:
        screen_events.append(ev)

    saved_cm = (agent_mod.call_model.configured, agent_mod.call_model.answer, agent_mod.plan)
    agent_mod.call_model.configured = lambda: True  # type: ignore[assignment]
    agent_mod.call_model.answer = screen_answer  # type: ignore[assignment]
    agent_mod.plan = lambda _t: [  # type: ignore[assignment]
        (0.0, "dispatcher", "How many people do you see?", None),
        (0.2, "sentinel", "scripted count", "How many people do you see?"),
        (0.3, "dispatcher", "Is anyone hurt?", None),
        (0.5, "sentinel", "I can't verify injuries from the cameras.", "Is anyone hurt?"),
    ]
    try:
        screen = VoiceAgent(screen_pub)
        await screen.start_scripted_call(rec, assemble_call_brief(rec))
        await asyncio.sleep(1.0)
        said = [getattr(e, "text", "") for e in screen_events if getattr(e, "speaker", "") == "sentinel"]
        assert said == ["They are in the ground-floor lobby.",
                        "I can't verify injuries from the cameras."], said
        assert [q for q, _ in asked] == ["How many people do you see?", "Is anyone hurt?"]
        assert all(t == 6.0 for _, t in asked), asked
        await screen.cancel()
    finally:
        agent_mod.call_model.configured, agent_mod.call_model.answer, agent_mod.plan = saved_cm

    # Demo recording: lines run from call start and finish inside DURATION.
    from services.voice.demo_call import DURATION

    scripted_events: list[object] = []

    async def scripted_pub(ev: object) -> None:
        scripted_events.append(ev)

    demo = VoiceAgent(scripted_pub, zrt=qwen)  # type: ignore[arg-type]
    demo.clock = lambda: 0.0
    await demo.start_scripted_call(rec, assemble_call_brief(rec))
    await asyncio.sleep(1.0)
    assert len(scripted_events) == 1, scripted_events
    await asyncio.sleep(3.2)
    await demo.cancel()
    said = [(getattr(e, "speaker", ""), getattr(e, "text", "")) for e in scripted_events]
    assert said[0] == ("dispatcher", "Campus police dispatch, what's your emergency?"), said
    assert len(said) == 2, said
    assert said[1] == ("sentinel", "Hi, this is First Witness at San Jose State."), said
    script_blob = " ".join(text for _, text in said).lower()
    assert "simulat" not in script_blob and "psap" not in script_blob
    assert "deterministic" not in script_blob and "never emergency" not in script_blob
    from services.voice.demo_call import DEMO_CALL as _LINES, OPENING

    # The recorded call's opening, at its own seconds after connect.
    assert _LINES[1] == OPENING[1]
    assert [at_s for at_s, _, _ in _LINES[:3]] == [0.0, 2.5, 5.62], _LINES
    assert not any("ibm" in text.lower() for _, _, text in _LINES), _LINES

    # A real camera handoff only updates internal call context. It must not
    # insert an out-of-order transcript message into the fixed demo timeline.
    scripted_events.clear()
    demo = VoiceAgent(scripted_pub, zrt=qwen)  # type: ignore[arg-type]
    await demo.start_scripted_call(rec, assemble_call_brief(rec))
    event_count = len(scripted_events)
    await demo.notify_whereabouts("cam-02", assemble_call_brief(rec).address)
    await asyncio.sleep(1.2)
    handoff_events = scripted_events[event_count:]
    handoff_lines = [getattr(event, "text", "") for event in handoff_events]
    assert not any(line.startswith("Update:") for line in handoff_lines), handoff_lines
    await asyncio.sleep(2.6)
    handoff_events = scripted_events[event_count:]
    handoff_lines = [getattr(event, "text", "") for event in handoff_events]
    assert not any("monitoring" in line.lower() for line in handoff_lines), handoff_lines
    assert not any("keep tracking" in line.lower() for line in handoff_lines), handoff_lines
    await demo.cancel()
    assert DURATION > _LINES[-1][0]

    from services.voice.media_bridge import MediaStreamBridge

    asked: list[tuple[str, str]] = []

    async def answer(incident_id: str, question: str) -> str:
        asked.append((incident_id, question))
        return "answered"

    bridge = MediaStreamBridge(
        incident_id=rec.incident_id,
        send=lambda _text: asyncio.sleep(0),
        recv=lambda: asyncio.sleep(0, result=None),
        subscribe=lambda _id: asyncio.Queue(),
        unsubscribe=lambda _id, _q: None,
        publish=live_pub,
        answer=answer,
    )
    bridge._asr = type("_Asr", (), {"transcribe": lambda _self, _pcm: "What is the location?"})()
    await bridge._transcribe_and_publish(b"pcm")
    assert asked == [(rec.incident_id, "What is the location?")]

    timing_events: list[object] = []

    async def timing_pub(ev: object) -> None:
        timing_events.append(ev)

    class _BlockingAsr:
        started = threading.Event()
        release = threading.Event()

        def transcribe(self, _pcm):
            self.started.set()
            self.release.wait(1)
            return "What is happening near the door?"

    timing_bridge = MediaStreamBridge(
        incident_id=rec.incident_id,
        send=lambda _text: asyncio.sleep(0),
        recv=lambda: asyncio.sleep(0, result=None),
        subscribe=lambda _id: asyncio.Queue(),
        unsubscribe=lambda _id, _q: None,
        publish=timing_pub,
        answer=answer,
    )
    blocking_asr = _BlockingAsr()
    timing_bridge._asr = blocking_asr
    pending = asyncio.create_task(timing_bridge._transcribe_and_publish(b"x" * 48_000))
    try:
        await asyncio.to_thread(blocking_asr.started.wait, 1)
        assert [getattr(event, "text", "") for event in timing_events] == ["One moment."]
    finally:
        blocking_asr.release.set()
    await pending
    assert [getattr(event, "speaker", "") for event in timing_events] == [
        "sentinel",
        "dispatcher",
    ]

    timing_events.clear()
    timing_bridge._asr = type(
        "_SecondAsr", (), {"transcribe": lambda _self, _pcm: "Where is the person?"}
    )()
    await timing_bridge._transcribe_and_publish(b"x" * 48_000)
    assert [getattr(event, "text", "") for event in timing_events] == [
        "Let me check that.",
        "Where is the person?",
    ]

    timing_events.clear()
    timing_bridge._asr = type(
        "_AckAsr", (), {"transcribe": lambda _self, _pcm: "Okay."}
    )()
    await timing_bridge._transcribe_and_publish(b"x" * 40_000)
    assert [getattr(event, "speaker", "") for event in timing_events] == ["dispatcher"]

    timing_events.clear()
    timing_bridge._asr = type(
        "_EmptyAsr", (), {"transcribe": lambda _self, _pcm: ""}
    )()
    await timing_bridge._transcribe_and_publish(b"x" * 48_000)
    assert [getattr(event, "text", "") for event in timing_events] == [
        "Give me a moment.",
        "I didn't catch that. Please repeat.",
    ]

    timing_bridge._tts = type(
        "_Tts", (), {"synthesize": lambda _self, _text: b""}
    )()
    timing_bridge._inbound_buf.extend(b"race-window audio")
    timing_bridge._heard_speech = True
    timing_bridge._turn_q.put_nowait(b"queued race-window turn")
    await timing_bridge._speak("One moment.")
    assert (
        not timing_bridge._inbound_buf
        and timing_bridge._heard_speech is False
        and timing_bridge._turn_q.empty()
    )

    class _SlowAsr:
        active = 0
        max_active = 0

        def transcribe(self, _pcm):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            self.active -= 1
            return "Repeat that."

    ordered: list[str] = []

    async def ordered_answer(_incident_id: str, question: str) -> str:
        if question == "first":
            await asyncio.sleep(0.05)
        ordered.append(question)
        return "answered"

    bridge.answer = ordered_answer
    bridge._asr = type(
        "_OrderAsr", (), {"transcribe": lambda _self, pcm: pcm.decode()}
    )()
    for utterance in (b"first", b"second", b"third", b"fourth", b"fifth"):
        bridge._inbound_buf.extend(utterance)
        bridge._flush_inbound()
    assert bridge._asr_task is not None
    await bridge._asr_task
    assert ordered == ["first", "second", "third", "fourth", "fifth"]

    slow = _SlowAsr()
    bridge.answer = answer
    bridge._asr = slow
    for utterance in (b"first utterance", b"second utterance"):
        bridge._inbound_buf.extend(utterance)
        bridge._flush_inbound()
    assert bridge._asr_task is not None
    await bridge._asr_task
    assert slow.max_active == 1

    bridge._asr = type("_Asr", (), {"transcribe": lambda _self, _pcm: "Is anyone hurt?"})()
    speech = (1000).to_bytes(2, "little", signed=True) * 320
    silence = b"\0" * 640
    bridge._speaking = True
    bridge._buffer_inbound(speech)
    bridge._speaking = False
    bridge._ignore_inbound_until = time.monotonic() + 1
    bridge._buffer_inbound(speech)
    assert not bridge._inbound_buf
    bridge._ignore_inbound_until = 0
    for _ in range(300):
        bridge._buffer_inbound(silence)
    assert not bridge._inbound_buf
    for _ in range(751):
        bridge._buffer_inbound(speech)
    for _ in range(25):
        bridge._buffer_inbound(silence)
    assert not bridge._inbound_buf and bridge._turn_q.empty()
    for _ in range(10):
        bridge._buffer_inbound(speech)
    for _ in range(25):
        bridge._buffer_inbound(silence)
    assert bridge._asr_task is not None
    await bridge._asr_task
    assert asked[-1] == (rec.incident_id, "Is anyone hurt?")
    await _check_phone_tts()
    print("voice self-check OK")


async def _check_phone_tts() -> None:
    """ElevenLabs is mocked: no network, no characters billed."""
    import os
    import urllib.error

    from services.voice import elevenlabs

    saved = {k: os.environ.get(k) for k in ("CS_TTS_PROVIDER", "ELEVENLABS_API_KEY")}

    class _Kokoro:
        calls = 0

        def synthesize(self, _text):
            self.calls += 1
            return b"\x00\x10" * 1600  # 100 ms PCM16 16 kHz

    class _Remote:
        calls = 0
        fail: Exception | None = None

        def synthesize_mulaw(self, _text):
            self.calls += 1
            if self.fail is not None:
                raise self.fail
            return b"\xff" * 800

    try:
        kokoro, remote = _Kokoro(), _Remote()
        tts = elevenlabs.PhoneTts(kokoro=kokoro, remote=remote)
        os.environ.pop("ELEVENLABS_API_KEY", None)
        os.environ["CS_TTS_PROVIDER"] = "elevenlabs"
        # No key: Kokoro only, resampled to 8 kHz mulaw (1 byte/sample).
        assert len(tts.synthesize_mulaw("Hello.")) == 800 and remote.calls == 0
        assert elevenlabs.status()["provider"] == "kokoro"

        os.environ["ELEVENLABS_API_KEY"] = "test-key-not-real"
        assert elevenlabs.status()["provider"] == "elevenlabs"
        assert "test-key-not-real" not in str(elevenlabs.status())
        assert tts.synthesize_mulaw("One moment.") == b"\xff" * 800
        assert tts.synthesize_mulaw("One moment.") == b"\xff" * 800
        assert remote.calls == 1 and tts.last_provider == "elevenlabs"  # cached filler
        tts.prewarm(("One moment.", "Let me check that."))
        assert remote.calls == 2
        tts.synthesize_mulaw("Let me check that.")
        assert remote.calls == 2

        remote.fail = TimeoutError()
        kokoro.calls = 0
        assert len(tts.synthesize_mulaw("A longer sentence that is not cached at all.")) == 800
        assert kokoro.calls == 1 and tts.last_provider == "kokoro"
        assert elevenlabs._state["parked_until"] == 0.0  # timeouts don't park

        remote.fail = urllib.error.HTTPError("u", 401, "quota_exceeded", {}, None)
        tts.synthesize_mulaw("Another uncached sentence for the quota path.")
        assert elevenlabs.status()["parked_s"] > 0
        before = remote.calls
        tts.synthesize_mulaw("Parked, so this must not reach the remote.")
        assert remote.calls == before and tts.last_provider == "kokoro"
        elevenlabs._state["parked_until"] = 0.0

        await _check_streaming(elevenlabs, tts, kokoro)
        _check_keepalive(elevenlabs)
    finally:
        elevenlabs._state["parked_until"] = 0.0
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def _check_streaming(elevenlabs, tts, kokoro) -> None:
    """Streamed lines hit the wire chunk-by-chunk; a slow first chunk falls
    back to Kokoro for that line only. Mocked: no network."""
    import os
    import urllib.error

    from services.voice.media_bridge import MediaStreamBridge

    class _StreamRemote:
        def __init__(self, chunks, first_delay=0.0):
            self.chunks, self.first_delay = chunks, first_delay
            self.calls = self.aborts = 0
            self.fail: Exception | None = None
            self.gate: threading.Event | None = None

        def stream(self, _text, on_chunk):
            self.calls += 1
            if self.fail is not None:
                raise self.fail
            time.sleep(self.first_delay)
            for i, chunk in enumerate(self.chunks):
                if not on_chunk(chunk):
                    return False
                if i == 0 and self.gate is not None:
                    # The rest only "arrives" once chunk 1 is already on the wire.
                    assert self.gate.wait(1), "first chunk was not played before the rest"
            return True

        def abort(self):
            self.aborts += 1

    sent: list[bytes] = []
    speaking_while_sent: list[bool] = []

    async def send(raw: str) -> None:
        msg = __import__("json").loads(raw)
        sent.append(base64.b64decode(msg["media"]["payload"]))
        speaking_while_sent.append(bridge._speaking)
        if remote.gate is not None:
            remote.gate.set()

    def fresh_bridge() -> MediaStreamBridge:
        b = MediaStreamBridge(
            incident_id="inc-stream",
            send=send,
            recv=lambda: asyncio.sleep(0, result=None),
            subscribe=lambda _id: asyncio.Queue(),
            unsubscribe=lambda _id, _q: None,
        )
        b._stream_sid = "MZ-test"
        b._tts = tts
        return b

    saved_first = os.environ.get("CS_ELEVENLABS_FIRST_CHUNK_S")
    os.environ["CS_ELEVENLABS_FIRST_CHUNK_S"] = "0.2"
    try:
        # 1) Uneven network chunks are re-framed to 160 bytes and the first
        #    frame is sent before the remote delivers anything else.
        remote = _StreamRemote([b"\x01" * 200, b"\x02" * 300, b"\x03" * 50])
        remote.gate = threading.Event()
        tts._remote_factory = lambda: remote
        assert tts.new_remote() is remote
        bridge = fresh_bridge()
        bridge._remote = remote
        line = "A streamed sentence that is far too long to be cached."
        await bridge._speak(line)
        assert b"".join(sent) == b"\x01" * 200 + b"\x02" * 300 + b"\x03" * 50
        assert [len(f) for f in sent] == [160, 160, 160, 70]
        assert all(speaking_while_sent) and bridge._speaking is False
        assert bridge._ignore_inbound_until > time.monotonic()  # echo guard after line
        assert tts.last_provider == "elevenlabs" and tts.last_error == "" and remote.aborts == 0

        # 2) Short lines are cached after a complete stream: replay is free.
        sent.clear()
        remote.gate = None
        await bridge._speak("Units en route.")
        await bridge._speak("Units en route.")
        assert remote.calls == 2
        assert len(b"".join(sent)) == 2 * 550

        # 3) First chunk too slow: Kokoro speaks the line, late chunks dropped.
        kokoro_line = tts.kokoro_mulaw("x")
        sent.clear()
        kokoro.calls = 0
        remote.first_delay = 0.5
        await bridge._speak("Slow network sentence that will not be cached ever.")
        assert kokoro.calls == 1 and remote.aborts == 1
        assert b"".join(sent) == kokoro_line  # only Kokoro, never both
        assert tts.last_provider == "kokoro"
        await asyncio.sleep(0.4)  # abandoned worker sees the drop
        assert b"".join(sent) == kokoro_line

        # 4) Error before audio (quota): Kokoro for this line, ElevenLabs parked.
        sent.clear()
        kokoro.calls = 0
        remote.first_delay = 0.0
        remote.fail = urllib.error.HTTPError("u", 402, "quota_exceeded", {}, None)
        await bridge._speak("Quota path sentence that is long enough to skip cache.")
        assert kokoro.calls == 1 and tts.last_error == "http_402"
        assert elevenlabs.status()["parked_s"] > 0
        before = remote.calls
        await bridge._speak("Parked sentence that must not reach the remote at all.")
        assert remote.calls == before and kokoro.calls == 2
    finally:
        elevenlabs._state["parked_until"] = 0.0
        tts._remote_factory = elevenlabs.ElevenLabsTts
        if saved_first is None:
            os.environ.pop("CS_ELEVENLABS_FIRST_CHUNK_S", None)
        else:
            os.environ["CS_ELEVENLABS_FIRST_CHUNK_S"] = saved_first


def _check_keepalive(elevenlabs) -> None:
    """ElevenLabsTts over a fake HTTPSConnection: one connection reused, a
    stale keep-alive is reconnected once, errors surface as HTTPError."""
    import http.client
    import urllib.error

    class _Resp:
        def __init__(self, status, body):
            self.status, self.reason, self.headers = status, "x", {}
            self._body = [body[i : i + 3] for i in range(0, len(body), 3)]

        def read1(self, _n):
            return self._body.pop(0) if self._body else b""

        def read(self):
            self._body = []
            return b""

    class _Conn:
        made = 0
        plan: list = []
        paths: list[str] = []

        def __init__(self, host, timeout):
            assert host == "api.elevenlabs.io"
            type(self).made += 1
            self.sock = None

        def connect(self):
            self.sock = object()

        def request(self, _method, path, body, headers):
            assert headers["xi-api-key"] and b"eleven_flash_v2_5" in body
            self.paths.append(path)
            self.sock = self.sock or object()
            step = self.plan.pop(0)
            if isinstance(step, Exception):
                raise step
            self._resp = step

        def getresponse(self):
            return self._resp

        def close(self):
            self.sock = None

    saved = elevenlabs.http.client.HTTPSConnection
    elevenlabs.http.client.HTTPSConnection = _Conn
    try:
        client = elevenlabs.ElevenLabsTts()
        client.warm()
        assert _Conn.made == 1 and client._conn.sock is not None
        _Conn.plan = [_Resp(200, b"abcdefg"), _Resp(200, b"hij")]
        got: list[bytes] = []
        assert client.stream("hi", lambda c: got.append(c) is None)
        assert got == [b"abc", b"def", b"g"]
        assert client.synthesize_mulaw("again") == b"hij" and _Conn.made == 1
        assert "optimize_streaming_latency=3" in _Conn.paths[0]
        assert "output_format=ulaw_8000" in _Conn.paths[0] and "/stream?" in _Conn.paths[0]

        _Conn.plan = [http.client.RemoteDisconnected("stale"), _Resp(200, b"ok")]
        assert client.synthesize_mulaw("after idle") == b"ok" and _Conn.made == 2

        _Conn.plan = [_Resp(401, b"")]
        try:
            client.synthesize_mulaw("bad key")
            raise AssertionError("expected HTTPError")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        assert _Conn.made == 2 and client._conn is not None  # drained error keeps it

        _Conn.plan = [_Resp(200, b"abcdef")]
        assert client.stream("stop early", lambda _c: False) is False
        assert client._conn is None  # undrained body is never reused
        client.close()
    finally:
        elevenlabs.http.client.HTTPSConnection = saved


if __name__ == "__main__":
    asyncio.run(_main())
