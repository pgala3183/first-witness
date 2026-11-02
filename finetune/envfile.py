"""Read KEY=value lines from the repo's .env into os.environ (never overrides).

.env is gitignored; it's where FIREWORKS_API_KEY lives so every terminal,
not only the one that ran `$env:FIREWORKS_API_KEY = ...`, can use it.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.split(" #", 1)[0].strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


load()
