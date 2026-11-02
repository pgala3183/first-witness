"""Shared media-root helpers for the lean three-camera demo runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

CORE_CAM_FILES = (
    "CAM01_lobby_entrance_IN_then_OUT_339s.mp4",
    "CAM02_hallway_east_IN_then_OUT_339s.mp4",
    "CAM03_hallway_west_IN_then_OUT_339s.mp4",
)

BANNED_FEED_FILES = {"REVIEW_three_camera_handoffs_339s.mp4"}

_DEMO_DIR = "final_demo_3cam_reconstructed"


def _clean(raw: str) -> Path:
    return Path(str(raw).strip().strip('"').strip("'")).expanduser()


def default_media_root(repo_root: Path) -> Path:
    """Repo-adjacent media root used by default on Linux/macOS/Windows."""
    return repo_root.parent / "campus_sentinel_media" / "feeds" / _DEMO_DIR


def ensure_allowed_feed_files(files: Iterable[str]) -> None:
    bad = sorted(BANNED_FEED_FILES.intersection({Path(name).name for name in files}))
    if bad:
        raise FileNotFoundError(
            "Refusing blocked review feed file(s): " + ", ".join(bad)
        )


def resolve_media_root(
    repo_root: Path,
    feed: dict | None = None,
    *,
    required_files: Iterable[str] = CORE_CAM_FILES,
    strict: bool = False,
) -> Path:
    """Resolve media root, preferring explicit env override then local default."""
    ensure_allowed_feed_files(required_files)
    roots: list[Path] = []

    override = os.environ.get("CS_MEDIA_ROOT", "").strip()
    if override:
        roots.append(_clean(override))

    roots.append(default_media_root(repo_root))

    media_root = (feed or {}).get("media_root") if isinstance(feed, dict) else {}
    if isinstance(media_root, dict):
        for key in ("windows", "zgx", "mac_mount"):
            raw = media_root.get(key)
            if raw:
                roots.append(_clean(str(raw)))

    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        if all((root / name).is_file() for name in required_files):
            return root

    if not strict and roots:
        return roots[0]

    expected = ", ".join(required_files)
    hints = " | ".join(str(r) for r in roots)
    raise FileNotFoundError(
        f"Demo media root not found with required files ({expected}). Tried: {hints}"
    )


def clip_map(media_root: Path) -> dict[str, tuple[Path, str]]:
    """The only runtime feeds. Cameras 4-6 must not mirror these files."""
    return {
        "cam-01": (media_root, CORE_CAM_FILES[0]),
        "cam-02": (media_root, CORE_CAM_FILES[1]),
        "cam-03": (media_root, CORE_CAM_FILES[2]),
    }
