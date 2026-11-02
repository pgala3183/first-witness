"""Build the First Witness call-agent dataset.

Outputs (in finetune/data/):
  train_sft.jsonl   - Fireworks SFT format (messages only)
  val_sft.jsonl     - Fireworks evaluation dataset (messages only)
  val_eval.jsonl, test_real_eval.jsonl, test_synth_eval.jsonl
                    - same messages + metadata for eval.py
  stats.json        - counts per split / intent / source

Splits (no leakage):
  train  : synthetic campuses + real Seville seconds 0-229, training paraphrases
  val    : new synthetic snapshots, training paraphrases
  test_real : real Seville seconds 230-339 (never trained on), HELD-OUT paraphrases
  test_synth: unseen campus names, HELD-OUT paraphrases

Usage: python finetune/build_dataset.py [--train 3300] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checker import check, required_ok  # noqa: E402
from dialog import QUESTIONS, asr_noise, gold_for, split_questions  # noqa: E402
from facts import dumps, real_snapshot, synthetic_snapshot  # noqa: E402
# The live call agent's prompt and message format: train on exactly what it sends.
from services.voice.call_model import SPEAKER, SYSTEM, user_msg  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"

# Honesty intents are oversampled: that's what this model exists for.
WEIGHTS = {
    "report": 5, "identity": 3, "address": 5, "location": 9, "count": 8, "weapons": 8,
    "describe": 7, "movement": 6, "route": 4, "injuries": 5, "who_subject": 4, "responders": 9,
    "pressure_enroute": 8, "pressure_onscene": 6, "zone_alert": 7, "lockdown": 4, "elapsed": 3,
    "status": 5, "unanswerable": 6, "repeat": 3,
}
assert set(WEIGHTS) == set(QUESTIONS)


def make_example(snap: dict[str, Any], rng: random.Random, part: str, noise: float,
                 source: str, t: int | None = None) -> dict[str, Any] | None:
    intent = rng.choices(list(WEIGHTS), list(WEIGHTS.values()))[0]
    history: list[tuple[str, str]] = []
    n_prev = 1 if intent == "repeat" else rng.choice([0, 0, 0, 1, 2])
    last = ""
    for _ in range(n_prev):
        pi = rng.choice([i for i in WEIGHTS if i not in ("repeat",)])
        pq = rng.choice(split_questions(pi, "train"))
        pa, _ = gold_for(pi, pq, snap, rng)
        history += [("Dispatcher", pq), (SPEAKER, pa)]
        last = pa
    q = rng.choice(split_questions(intent, part))
    if rng.random() < noise:
        q = asr_noise(q, rng)
    a, must = gold_for(intent, q, snap, rng, last_answer=last)

    v = check(a, snap)
    miss = required_ok(a, must)
    if v or miss:
        # A gold answer that fails the checker is a bug in dialog.py; surface it.
        raise AssertionError(f"gold failed checker: intent={intent} a={a!r} v={v} miss={miss}\n{dumps(snap)}")

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(snap, history, q)},
            {"role": "assistant", "content": a},
        ],
        "meta": {"intent": intent, "question": q, "must": must, "snap": snap,
                 "source": source, "t": t, "last_answer": last},
    }


def build(n: int, rng: random.Random, part: str, noise: float, mix: list[tuple[str, float]],
          real_range: range | None, heldout_sites: bool = False) -> list[dict[str, Any]]:
    out, seen = [], set()
    kinds, probs = zip(*mix)
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        kind = rng.choices(kinds, probs)[0]
        if kind == "real":
            t = rng.choice(list(real_range))
            ex = make_example(real_snapshot(t, rng), rng, part, noise, "seville", t)
        else:
            ex = make_example(synthetic_snapshot(rng, heldout=heldout_sites), rng, part, noise, "synthetic")
        key = ex["messages"][1]["content"]
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def write(path: Path, rows: list[dict[str, Any]], sft: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]} if sft else r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3300)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=450)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    train = build(args.train, random.Random(args.seed), "train", 0.3,
                  [("synthetic", 0.78), ("real", 0.22)], range(0, 230))
    val = build(args.val, random.Random(args.seed + 1), "train", 0.3,
                [("synthetic", 0.85), ("real", 0.15)], range(0, 230))
    test_real = build(args.test, random.Random(args.seed + 2), "heldout", 0.4,
                      [("real", 1.0)], range(230, 340))
    test_synth = build(args.test, random.Random(args.seed + 3), "heldout", 0.4,
                       [("synthetic", 1.0)], None, heldout_sites=True)

    # Belt and braces: no user prompt shared between train and any eval split.
    tr = {r["messages"][1]["content"] for r in train}
    for name, rows in (("val", val), ("test_real", test_real), ("test_synth", test_synth)):
        assert not tr & {r["messages"][1]["content"] for r in rows}, name

    write(OUT / "train_sft.jsonl", train, sft=True)
    write(OUT / "val_sft.jsonl", val, sft=True)
    write(OUT / "train_eval.jsonl", train, sft=False)
    write(OUT / "val_eval.jsonl", val, sft=False)
    write(OUT / "test_real_eval.jsonl", test_real, sft=False)
    write(OUT / "test_synth_eval.jsonl", test_synth, sft=False)

    stats = {
        name: {"n": len(rows),
               "by_intent": dict(Counter(r["meta"]["intent"] for r in rows).most_common()),
               "by_source": dict(Counter(r["meta"]["source"] for r in rows)),
               "officer_status": dict(Counter(r["meta"]["snap"]["officer"]["status"] for r in rows)),
               "zone_status": dict(Counter(r["meta"]["snap"]["zone_alert"]["status"] for r in rows)),
               "subject_visible": dict(Counter(str(r["meta"]["snap"]["now"]["visible"]) for r in rows))}
        for name, rows in (("train", train), ("val", val), ("test_real", test_real), ("test_synth", test_synth))
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    for k, s in stats.items():
        print(f"{k:10s} {s['n']:5d}  sources={s['by_source']}")


if __name__ == "__main__":
    main()
