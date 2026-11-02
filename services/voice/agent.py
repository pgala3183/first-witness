"""Voice service — on-screen 911 conversation and live call answers.

Demo recording plays services.voice.demo_call on the transcript only.
A live call speaks through the existing media bridge after the operator
places it. Whereabouts stay current either way.

Live answers: the fine-tuned call model (call_model) answers from this
moment's FACTS (facts.live_snapshot), every reply goes through the honesty
guard, and a timeout or error falls back to the keyword rules below.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from contracts import (
    CallBrief,
    CallTranscriptDelta,
    DemoControl,
    IncidentRecord,
    ToolCallLive,
    call_brief_to_dict,
    utcnow,
)
from services.brain.audit import AuditLog
from services.brain.call_brief import scene_facts
from services.brain.zrt_client import ZRTClient

from . import call_model, scene
from .demo_call import plan
from .facts import live_snapshot
from .guard import guarded
from .tools import TOOL_HANDLERS, ToolName

PublishFn = Callable[[Any], Awaitable[None]]


@dataclass
class CallScriptStep:
    after_s: float
    speaker: str  # dispatcher | sentinel
    text: str = ""
    tool: ToolName | None = None


# Officer Q&A. First Witness only speaks map/camera facts + tools.
WEAPON_SCRIPT: list[CallScriptStep] = [
    CallScriptStep(0.8, "dispatcher", "911, what's your emergency?"),
    CallScriptStep(
        0.6,
        "sentinel",
        "Armed individual with a visible firearm on camera {camera}. Location: {address}.",
    ),
    CallScriptStep(0.9, "dispatcher", "Is the person conscious? Any injuries? Names?"),
    CallScriptStep(
        0.5,
        "sentinel",
        "I cannot confirm consciousness, injury, pulse, or identity. I only report what the cameras show.",
    ),
    CallScriptStep(0.8, "dispatcher", "Give me a description of the subject."),
    CallScriptStep(0.4, "sentinel", tool="get_person_description"),
    CallScriptStep(0.7, "dispatcher", "Exact location and how long have you been tracking?"),
    CallScriptStep(0.4, "sentinel", tool="lookup_location"),
    CallScriptStep(0.4, "sentinel", tool="get_elapsed_time"),
    CallScriptStep(0.8, "dispatcher", "Direction of travel? Which camera next?"),
    CallScriptStep(
        0.5,
        "sentinel",
        "Subject last seen on {camera}. Adjacent cameras are monitored; I will update if they reappear.",
    ),
    CallScriptStep(0.8, "dispatcher", "Copy. Stay on the line for updates."),
    CallScriptStep(
        0.5,
        "sentinel",
        "Acknowledged. I will keep reporting camera handoffs.",
    ),
]


def heard(text: str) -> str:
    """Keep transcript and phone audio free of demo-only annotations."""
    cleaned = re.sub(r"\s*\(demo-only\)", "", text or "", flags=re.I)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def handoff_line(camera_id: str, building: str = "") -> str:
    """Spoken handoff: a place name only, never camera ids or map notes."""
    place = str(scene_facts(camera_id).get("spoken_location") or "").strip()
    if not place:
        place = f"another part of {building}" if building else "another camera view"
    # No "security has been re-alerted": nothing confirms anyone saw it.
    return f"Update: the person has moved to {place}."


def count_answer(people: int, armed: int) -> str:
    """Live person count from the tracker overlay, not a static fact."""
    if people <= 0:
        return "I don't see anyone on the current camera right now."
    who = "one person" if people == 1 else f"{people} people"
    if armed <= 0:
        return f"The current camera shows {who}."
    if people == 1:
        return "The current camera shows one person, and they are armed."
    which = "one of them is" if armed == 1 else f"{armed} of them are"
    return f"The current camera shows {who}; {which} armed."


class VoiceAgent:
    """Plays officer script; accepts live whereabouts updates when subject hits cam-02/03."""

    def __init__(self, publish: PublishFn, *, zrt: ZRTClient | None = None) -> None:
        self.publish = publish
        # Clip clock of the camera wall (seconds); answers describe that moment.
        self.clock: Callable[[], float] = lambda: 0.0
        self._zrt = zrt or ZRTClient()
        self._task: asyncio.Task[None] | None = None
        self._live = False
        self._scripted = False
        self._model_source = "model"
        self._incident_id: str | None = None
        self._camera = ""
        self._address = ""
        self._building = ""
        self._site_address = ""
        self._person = ""
        self._visible = True
        self._last_answer = ""
        self._unknown_answer_index = 0
        self._live_brief: CallBrief | None = None
        # camera_id -> (people, armed) from the latest overlay.boxes.
        self._counts: dict[str, tuple[int, int]] = {}
        self._update_q: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        # (speaker, text) turns the call model sees, as in training.
        self._history: list[tuple[str, str]] = []
        # Officer status as the operator reports it; the model only promises
        # help once this says "accepted". No zone warnings exist yet.
        self.officer: dict[str, Any] = {"status": "none"}
        self.zone_alert: dict[str, Any] = {"status": "none"}
        # Off for the keyword-router baseline in finetune/eval.py.
        self.use_model = True
        self.audit = AuditLog()

    def busy(self) -> bool:
        return self._live or (self._task is not None and not self._task.done())

    def active_incident_id(self) -> str | None:
        return self._incident_id if self.busy() else None

    def set_officer(
        self, status: str, name: str = "", eta_min: int | None = None, unit: str = ""
    ) -> dict[str, Any]:
        """none -> offered -> accepted (or declined), in the training format."""
        if status not in ("none", "offered", "declined", "accepted"):
            raise ValueError(f"unknown officer status {status!r}")
        o: dict[str, Any] = {"status": status}
        name = name.strip()
        if status != "none" and name:
            o["name"] = name if name.lower().startswith("officer ") else f"Officer {name}"
            if unit.strip():
                o["unit"] = unit.strip()
        if status == "accepted" and eta_min is not None:
            o["eta_min"] = max(0, int(eta_min))
        self.officer = o
        return dict(o)

    def _new_call(self) -> None:
        self._history = []
        self.officer = {"status": "none"}
        self.zone_alert = {"status": "none"}

    async def start_call(
        self,
        rec: IncidentRecord,
        brief: CallBrief,
        *,
        script: list[CallScriptStep] | None = None,
    ) -> None:
        if self.busy():
            return
        self._update_q = asyncio.Queue()
        self._new_call()
        self._incident_id = rec.incident_id
        self._camera = rec.camera_id
        self._address = brief.address
        self._building = scene.BUILDING
        self._site_address = scene.FULL_ADDRESS
        self._person = brief.person_description
        self._live_brief = brief
        self._visible = True
        self._unknown_answer_index = 0
        self._task = asyncio.create_task(
            self._run(rec, brief, script or WEAPON_SCRIPT),
            name=f"call-{rec.incident_id}",
        )

    async def start_scripted_call(
        self, rec: IncidentRecord, brief: CallBrief | None = None
    ) -> None:
        """Demo recording: the on-screen 911 conversation. Does not dial."""
        if self.busy():
            return
        self._update_q = asyncio.Queue()
        self._new_call()
        if self.use_model:
            asyncio.get_running_loop().run_in_executor(None, call_model.warmup)
        self._scripted = True
        self._incident_id = rec.incident_id
        self._camera = rec.camera_id
        self._building = scene.BUILDING
        self._address = scene.FULL_ADDRESS
        self._site_address = scene.FULL_ADDRESS
        self._task = asyncio.create_task(
            self._run_scripted(rec.incident_id), name=f"demo-call-{rec.incident_id}"
        )

    async def _run_scripted(self, incident_id: str) -> None:
        started = asyncio.get_running_loop().time()
        # On screen a short pause is fine; a phone call keeps CS_CALL_MODEL_TIMEOUT_S.
        try:
            timeout_s = float(os.environ.get("CS_CALL_MODEL_SCRIPTED_TIMEOUT_S", "") or 6.0)
        except ValueError:
            timeout_s = 6.0
        pending: asyncio.Task[str | None] | None = None
        try:
            # Keyed to the clip second the wall shows as the call connects.
            lines = plan(self.clock())
            for i, (at_s, speaker, text, question) in enumerate(lines):
                # One line per wait. The timestamp already includes speaking time.
                wait = at_s - (asyncio.get_running_loop().time() - started)
                if wait > 0:
                    await asyncio.sleep(wait)
                if question:
                    # The fine-tuned model answers from this second's facts (through the
                    # guard); the scripted scene answer stands if it's off or slow.
                    reply = await pending if pending is not None else await self._model_answer(question, timeout_s)
                    pending = None
                    text = reply or text
                    self._history += [("Dispatcher", question), (call_model.SPEAKER, heard(text))]
                elif i + 1 < len(lines) and lines[i + 1][3]:
                    # Ask as the dispatcher's question appears, so the model's latency
                    # overlaps the pause before the answer is due.
                    pending = asyncio.create_task(self._model_answer(lines[i + 1][3], timeout_s))
                await self.publish(
                    CallTranscriptDelta(
                        incident_id=incident_id,
                        speaker=speaker,  # type: ignore[arg-type]
                        text=heard(text),
                        ts=utcnow(),
                        source="script" if speaker == "sentinel" else "",
                    )
                )
            # The opening establishes the emergency.  Further transcript
            # turns are event-driven by real camera handoffs.
            await asyncio.Event().wait()
        finally:
            if pending is not None:
                pending.cancel()
            self._scripted = False
            self._incident_id = None

    async def start_live_call(self, rec: IncidentRecord, brief: CallBrief) -> None:
        if self.busy():
            return
        self._live = True
        self._new_call()
        if self.use_model:
            # Off the event loop: the first answer would otherwise pay the model's cold start.
            asyncio.get_running_loop().run_in_executor(None, call_model.warmup)
        self._incident_id = rec.incident_id
        self._camera = rec.camera_id
        self._address = scene.FULL_ADDRESS
        self._building = scene.BUILDING
        self._person = brief.person_description
        self._live_brief = brief
        self._visible = True
        self._unknown_answer_index = 0
        self._site_address = scene.FULL_ADDRESS
        text = heard(
            f"Hi, this is {scene.PRODUCT}. "
            f"I'm reporting armed people inside {scene.BUILDING}. "
            f"The address is {scene.ADDRESS or scene.BUILDING}."
        )
        self._last_answer = text
        self._history.append((call_model.SPEAKER, text))
        await self.publish(
            CallTranscriptDelta(
                incident_id=rec.incident_id,
                speaker="sentinel",
                text=text,
                ts=utcnow(),
            )
        )
        await self.publish(
            DemoControl(
                action="scenario",
                scenario_id=f"security_alert:{rec.camera_id}",
                ts=utcnow(),
            )
        )

    async def answer_dispatcher(self, incident_id: str, question: str) -> str:
        if not self._live or incident_id != self._incident_id:
            return f"There is no active {scene.PRODUCT} call for that incident."
        facts = scene_facts(self._camera)
        facts.update(
            camera_id=self._camera,
            address=self._address,
            person_description=self._person,
        )
        if self._camera in self._counts:
            # Live tracker count replaces the static map fact.
            facts["subject_count"], facts["armed_count"] = self._counts[self._camera]
        q = question.lower()
        if q.strip(" .,!?") in {
            "ok", "okay", "copy", "got it", "understood", "thanks", "thank you"
        }:
            return ""
        reply = await self._model_answer(question)
        if reply is not None:
            return await self._say(incident_id, question, reply, source=self._model_source)
        # Fallback: fixed questions answer instantly from the scene script at this clip second.
        fixed = scene.answer(question, self.clock())
        if fixed:
            return await self._say(incident_id, question, fixed)
        if "repeat" in q or "say that again" in q:
            if "address" in q:
                answer = f"{self._site_address}."
            else:
                answer = self._last_answer
        elif "emergency" in q or "what are you reporting" in q:
            answer = f"I'm reporting an armed person at {self._building}."
        elif "address" in q or q.strip(" ?.!") == "where are you":
            answer = f"{self._site_address}."
        elif "where" in q or "location" in q:
            answer = (
                f"The person is in {facts['spoken_location']}."
                if self._visible
                and facts.get("spoken_location")
                else "I don't see the person now; this is their last confirmed camera."
            )
        elif not self._visible and any(
            word in q
            for word in (
                "weapon", "gun", "firearm", "direction", "travel", "going",
                "headed", "describe", "description", "wearing", "clothing",
                "look like",
            )
        ):
            answer = "They aren't visible, so I can't verify that now."
        elif any(word in q for word in ("weapon", "gun", "firearm")):
            answer = f"The cameras show a visible {facts.get('visible_weapon', 'weapon')}."
        elif any(word in q for word in ("hurt", "injur", "medical", "conscious", "breath", "pulse")):
            answer = "I can't verify their medical condition from this camera."
        elif any(word in q for word in ("direction", "travel", "going", "headed")):
            answer = f"The person is {facts.get('direction', 'moving in an unconfirmed direction')}."
        elif any(word in q for word in ("describe", "description", "wearing", "clothing", "look like")):
            answer = f"The person appears to be {self._person.rstrip('.').lower()}."
        elif any(word in q for word in ("how many", "count", "number of")):
            answer = count_answer(*self._counts.get(self._camera, (0, 0)))
        elif any(word in q for word in ("name", "identity", "who is", "intent", "why")):
            answer = "I can't identify the person or their intent from this footage."
        else:
            if not q.rstrip().endswith("?") and not q.lstrip().startswith(
                (
                    "what", "where", "when", "who", "how", "is ", "are ",
                    "can ", "could ", "do ", "does ", "did ", "has ", "have ",
                    "please ", "tell me",
                )
            ):
                return ""
            try:
                answer = await asyncio.to_thread(
                    self._zrt.answer_dispatcher, self._frame_facts(), question
                )
            except Exception:
                answer = ""
            refusal = heard(answer).strip().lower()
            if not answer or refusal.startswith(
                ("the cameras do not confirm", "i cannot confirm", "i can't confirm")
            ):
                answers = (
                    "I don't see that on the current camera.",
                    "That isn't visible in the current camera view.",
                )
                answer = answers[self._unknown_answer_index % len(answers)]
                self._unknown_answer_index += 1
            else:
                answer = re.split(r"(?<=[.!?])\s+", answer, maxsplit=1)[0]
        return await self._say(incident_id, question, answer)

    async def _model_answer(self, question: str, timeout_s: float | None = None) -> str | None:
        """The fine-tuned call model on this moment's FACTS, through the guard.
        None (not configured, timeout, error) means the keyword rules answer."""
        if not self.use_model or not call_model.configured():
            return None
        snap = live_snapshot(self.clock(), self.officer, self.zone_alert)
        # Training showed at most two earlier exchanges.
        reply = await asyncio.to_thread(call_model.answer, snap, self._history[-4:], question, timeout_s)
        if reply is None:
            return None
        text, violations = guarded(reply, snap)
        self._model_source = "guard" if violations else "model"
        if violations:
            self.audit.record(
                incident_id=self._incident_id or "",
                action="guard_blocked",
                reason="; ".join(detail for _, detail in violations),
                confidence=1.0,
                detail={"question": question, "reply": reply},
            )
        return text

    async def _say(self, incident_id: str, question: str, text: str, *, source: str = "rules") -> str:
        answer = heard(text)
        self._last_answer = answer
        self._history += [("Dispatcher", question), (call_model.SPEAKER, answer)]
        await self.publish(
            CallTranscriptDelta(
                incident_id=incident_id,
                speaker="sentinel",
                text=answer,
                ts=utcnow(),
                source=source,
                model=call_model.model_label() if source in ("model", "guard") else "",
            )
        )
        return answer

    def _frame_facts(self) -> dict[str, Any]:
        """The current second of the locked scene script, for the answering model."""
        t = float(self.clock())
        snap = scene.at(t)
        people, weapons = scene.steady(t)
        return {
            "place": scene.BUILDING,
            "clip_second": int(t),
            "camera_id": self._camera,
            "location": scene.location(snap) if snap else "",
            "people": people,
            "weapons": weapons,
            "scene": snap,
        }

    def update_visual(
        self, camera_id: str, visible: bool, people: int | None = None, armed: int = 0
    ) -> None:
        """Keep live answers aligned with the latest overlay without model work."""
        self._counts[camera_id] = (int(visible) if people is None else people, armed)
        if self._live and camera_id == self._camera:
            self._visible = visible

    async def notify_whereabouts(self, camera_id: str, address: str) -> None:
        """Vision escalated the chase on another camera — update the live call."""
        if not self.busy() or not self._incident_id:
            return
        if camera_id == self._camera:
            return
        if self._scripted:
            # Transcript timing belongs exclusively to DEMO_CALL. Vision
            # handoffs still keep scenario state current, but never insert an
            # early or duplicate spoken update.
            self._camera = camera_id
            self._address = address
            self._visible = True
            if self._live_brief is not None:
                self._live_brief.camera_id = camera_id
                self._live_brief.address = address
            await self.publish(
                DemoControl(
                    action="scenario",
                    scenario_id=f"security_alert:{camera_id}",
                    ts=utcnow(),
                )
            )
        elif self._live:
            await self._publish_update(camera_id, address)
        else:
            await self._update_q.put((camera_id, address))

    async def cancel(self) -> None:
        self._live = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._incident_id = None

    async def _publish_update(self, camera_id: str, address: str) -> None:
        previous = self._camera
        self._camera = camera_id
        self._address = address
        self._visible = True
        if self._live_brief is not None:
            self._live_brief.camera_id = camera_id
            self._live_brief.address = address
        report = scene.update_line(previous.upper(), camera_id.upper())
        await self.publish(
            CallTranscriptDelta(
                incident_id=self._incident_id or "",
                speaker="sentinel",
                text=report,
                ts=utcnow(),
            )
        )
        await self.publish(
            DemoControl(
                action="scenario",
                scenario_id=f"security_alert:{camera_id}",
                ts=utcnow(),
            )
        )

    async def _drain_updates(self, rec: IncidentRecord) -> None:
        while True:
            try:
                cam, address = self._update_q.get_nowait()
            except asyncio.QueueEmpty:
                return
            await self._publish_update(cam, address)
            await self.publish(
                ToolCallLive(
                    incident_id=rec.incident_id,
                    tool="lookup_location",
                    args={"camera_id": cam},
                    result={
                        "address": address,
                        "spoken": f"Location on file: {address}.",
                    },
                    ts=utcnow(),
                )
            )

    async def _run(
        self,
        rec: IncidentRecord,
        brief: CallBrief,
        script: list[CallScriptStep],
    ) -> None:
        ctx: dict[str, Any] = {
            "camera": rec.camera_id,
            "address": brief.address,
            "person": brief.person_description,
            "brief": brief,
            "record": rec,
            "brief_dict": call_brief_to_dict(brief),
        }
        try:
            await self.publish(
                DemoControl(
                    action="scenario",
                    scenario_id=f"security_alert:{rec.camera_id}",
                    ts=utcnow(),
                )
            )
            for step in script:
                await self._drain_updates(rec)
                await asyncio.sleep(step.after_s)
                await self._drain_updates(rec)
                ctx["camera"] = self._camera or rec.camera_id
                ctx["address"] = self._address or brief.address
                ctx["person"] = self._person or brief.person_description
                now = utcnow()
                if step.tool:
                    handler = TOOL_HANDLERS[step.tool]
                    args, result = handler(ctx)
                    await self.publish(
                        ToolCallLive(
                            incident_id=rec.incident_id,
                            tool=step.tool,
                            args=args,
                            result=result,
                            ts=now,
                        )
                    )
                    spoken = str(result.get("spoken") or result.get("text") or "")
                    if spoken:
                        await self.publish(
                            CallTranscriptDelta(
                                incident_id=rec.incident_id,
                                speaker="sentinel",
                                text=spoken,
                                ts=utcnow(),
                            )
                        )
                    continue
                text = step.text.format(
                    camera=ctx["camera"],
                    address=ctx["address"],
                    person=ctx["person"],
                )
                await self.publish(
                    CallTranscriptDelta(
                        incident_id=rec.incident_id,
                        speaker=step.speaker,  # type: ignore[arg-type]
                        text=text,
                        ts=now,
                    )
                )
            for _ in range(20):
                await self._drain_updates(rec)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        finally:
            self._incident_id = None
