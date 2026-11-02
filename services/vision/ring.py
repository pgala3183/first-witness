"""Per-camera ring buffer — last 8 seconds of frames (playbook C / D).

Qwen3-VL only ever sees 384 px frames, so the live pipeline stores them at
that size and at most a few per second (max_side / max_fps): full 960x540
frames at 15 fps cost ~187 MB per camera. Both default to off.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from contracts.incident import require_utc

from .decode import Frame


@dataclass(frozen=True)
class RingItem:
    ts: datetime
    image: object
    person_xywh: tuple[float, float, float, float]


def shrink(image: object, max_side: int) -> tuple[object, float]:
    """(image with its longest side <= max_side, scale applied)."""
    shape = getattr(image, "shape", None)
    if not max_side or shape is None or len(shape) < 2:
        return image, 1.0
    h, w = int(shape[0]), int(shape[1])
    s = max_side / max(h, w)
    if s >= 1.0:
        return image, 1.0
    import cv2

    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA), s


class RingBuffer:
    """Keep the last `window_s` seconds per camera. Evict by timestamp."""

    def __init__(self, window_s: float = 8.0, *, max_side: int = 0, max_fps: float = 0.0) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self.window_s = window_s
        self.max_side = max_side
        self.min_gap_s = 1.0 / max_fps if max_fps > 0 else 0.0
        self._buf: dict[str, deque[RingItem]] = {}

    def push(self, frame: Frame) -> None:
        require_utc("frame.ts", frame.ts)
        q = self._buf.setdefault(frame.camera_id, deque())
        if q and self.min_gap_s and (frame.ts - q[-1].ts).total_seconds() < self.min_gap_s:
            return
        image, s = shrink(frame.image, self.max_side)
        x, y, w, h = frame.person_xywh
        q.append(
            RingItem(
                ts=frame.ts,
                image=image,
                person_xywh=(x * s, y * s, w * s, h * s),
            )
        )
        cutoff = frame.ts - timedelta(seconds=self.window_s)
        while q and q[0].ts < cutoff:
            q.popleft()

    def window(self, camera_id: str) -> list[RingItem]:
        return list(self._buf.get(camera_id, ()))

    def timestamps(self, camera_id: str) -> list[float]:
        return [item.ts.timestamp() for item in self.window(camera_id)]
