"""Fact snapshots for the fine-tune kit: re-exported from services/voice/facts.py.

The live call agent and the training data build FACTS with the same code.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.voice.facts import *  # noqa: E402,F401,F403
from services.voice.facts import real_snapshot, synthetic_snapshot  # noqa: E402

if __name__ == "__main__":
    r = random.Random(0)
    print(json.dumps(real_snapshot(50, r), indent=1)[:1500])
    print(json.dumps(real_snapshot(255, r), indent=1)[:1500])
    print(json.dumps(synthetic_snapshot(r), indent=1))
