"""Grade a call agent on the held-out sets.

Systems:
  --system gold       the reference answers (sanity check: must score 100%)
  --system baseline   First Witness's CURRENT keyword router (services/voice/agent.py),
                      real Seville rows only (it reads the clip clock, not FACTS)
  --system model      any OpenAI-compatible endpoint, e.g. Fireworks:
                      --model accounts/fireworks/models/qwen3-4b-instruct-2507
                      (needs FIREWORKS_API_KEY; --base-url to point elsewhere)

Examples:
  python finetune/eval.py --system baseline --split test_real
  python finetune/eval.py --system model --model accounts/<you>/models/first-witness-call-4b \
         --split test_real --split test_synth --name ft-4b
  python finetune/eval.py --compare results/*.json      # tradeoff table

Writes results/<name>_<split>.json (summary) and .jsonl (every answer, for review).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
import envfile  # noqa: E402,F401  (FIREWORKS_API_KEY from the repo .env)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from checker import score  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"


# ---------------------------------------------------------------- systems

def run_gold(row: dict[str, Any]) -> tuple[str, float, dict[str, int]]:
    return row["messages"][2]["content"], 0.0, {}


_BASELINE = None


def run_baseline(row: dict[str, Any]) -> tuple[str, float, dict[str, int]] | None:
    """The real VoiceAgent.answer_dispatcher, fed the same moment of the clip."""
    global _BASELINE
    m = row["meta"]
    if m["source"] != "seville":
        return None
    from services.voice import scene
    from services.voice.agent import VoiceAgent

    if _BASELINE is None:
        async def _noop(_: Any) -> None:
            return None
        _BASELINE = VoiceAgent(_noop)
    ag = _BASELINE
    snap, t = m["snap"], m["t"]
    cam = scene.at(t).get("camera", "CAM-01").lower()
    ag._live, ag._incident_id, ag._camera = True, "eval", cam
    ag.use_model = False  # keyword rules only: the router baseline
    ag._building = snap["building"]
    ag._address = f"{snap['building']}, {snap['site_address']}"
    ag._site_address = snap["site_address"].replace(", CA ", ", California ")
    subj = snap["now"] if snap["now"]["visible"] else (snap.get("last_seen") or snap["now"])
    ag._person = "; ".join(subj.get("subjects") or []) or "an unidentified person"
    ag._visible = snap["now"]["visible"]
    ag._counts = {cam: (snap["now"]["people"], snap["now"]["armed"])}
    ag._last_answer = m.get("last_answer", "")
    ag.clock = lambda: float(t)
    t0 = time.perf_counter()
    a = asyncio.run(ag.answer_dispatcher("eval", m["question"]))
    return a, (time.perf_counter() - t0) * 1000, {}


def make_model_runner(args: argparse.Namespace):
    key = os.environ.get(args.key_env, "")
    if not key:
        sys.exit(f"Set {args.key_env} (your Fireworks API key).")
    url = args.base_url.rstrip("/") + "/chat/completions"

    def run(row: dict[str, Any]) -> tuple[str, float, dict[str, int]]:
        body = {"model": args.model, "messages": row["messages"][:2], "temperature": 0,
                "max_tokens": args.max_tokens}
        if args.no_think:
            # Qwen3 hybrid models: keep thinking off for a phone call.
            body["messages"] = [dict(body["messages"][0]),
                                {**body["messages"][1], "content": body["messages"][1]["content"] + " /no_think"}]
        if args.reasoning_effort:
            # Reasoning models otherwise spend max_tokens thinking and answer nothing.
            body["reasoning_effort"] = args.reasoning_effort
        req =urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read())
                ms = (time.perf_counter() - t0) * 1000
                text = out["choices"][0]["message"]["content"] or ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                return text, ms, out.get("usage", {})
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                return f"[HTTP {e.code}] {e.read()[:200]!r}", 0.0, {}
            except Exception as e:  # noqa: BLE001
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                return f"[error] {e}", 0.0, {}
        return "[error]", 0.0, {}

    return run


# ---------------------------------------------------------------- grading

# Questions about officer accept / zone alerts: features the current agent doesn't have yet.
NEW_FEATURE_INTENTS = {"responders", "pressure_enroute", "pressure_onscene", "zone_alert", "lockdown"}


def pct(x: float) -> float:
    return round(100 * x, 1)


def evaluate(rows: list[dict[str, Any]], runner, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(runner, rows))
    graded = []
    for row, out in zip(rows, outs):
        if out is None:
            continue
        a, ms, usage = out
        s = score(a, row["meta"]["snap"], row["meta"]["must"])
        graded.append({"intent": row["meta"]["intent"], "question": row["meta"]["question"],
                       "answer": a, "reference": row["messages"][2]["content"], "ms": ms,
                       "usage": usage, **s})
    n = len(graded)
    by = defaultdict(list)
    for g in graded:
        by[g["intent"]].append(g["ok"])
    lat = [g["ms"] for g in graded if g["ms"] > 0]
    tok_in = sum(g["usage"].get("prompt_tokens", 0) for g in graded)
    tok_out = sum(g["usage"].get("completion_tokens", 0) for g in graded)
    honesty = [g for g in graded if g["intent"] in ("responders", "pressure_enroute", "pressure_onscene",
                                                    "zone_alert", "lockdown")]
    core = [g for g in graded if g["intent"] not in NEW_FEATURE_INTENTS]
    summary = {
        "core_pass_rate": pct(sum(g["ok"] for g in core) / max(len(core), 1)),
        "n": n,
        "pass_rate": pct(sum(g["ok"] for g in graded) / max(n, 1)),
        "false_promise_rate": pct(sum(g["promise_violation"] for g in graded) / max(n, 1)),
        "made_up_fact_rate": pct(sum(g["fact_violation"] for g in graded) / max(n, 1)),
        "missing_fact_rate": pct(sum(bool(g["missing"]) for g in graded) / max(n, 1)),
        "honesty_pass_rate": pct(sum(g["ok"] for g in honesty) / max(len(honesty), 1)),
        # "[HTTP 404] ..." / "[error] ...": the endpoint failed, not the model.
        "error_rate": pct(sum(g["answer"].startswith(("[HTTP", "[error")) for g in graded) / max(n, 1)),
        # Blank answers: usually a reasoning model that spent max_tokens thinking.
        "empty_rate": pct(sum(not g["answer"].strip() for g in graded) / max(n, 1)),
        "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1) if lat else None,
        "tokens_in_per_answer": round(tok_in / max(n, 1), 1),
        "tokens_out_per_answer": round(tok_out / max(n, 1), 1),
        "by_intent": {k: pct(sum(v) / len(v)) for k, v in sorted(by.items())},
    }
    return summary, graded


def compare(paths: list[str]) -> None:
    rows = []
    for p in paths:
        s = json.loads(Path(p).read_text())
        rows.append((Path(p).stem, s))
    cols = ["n", "pass_rate", "core_pass_rate", "honesty_pass_rate", "false_promise_rate", "made_up_fact_rate",
            "missing_fact_rate", "latency_ms_p50", "latency_ms_p95", "cost_per_1k_answers_usd"]
    print("| run | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, s in rows:
        print(f"| {name} | " + " | ".join(str(s.get(c, "")) for c in cols) + " |")
    intents = sorted({i for _, s in rows for i in s["by_intent"]})
    print("\n| intent | " + " | ".join(n for n, _ in rows) + " |")
    print("|---|" + "---|" * len(rows))
    for i in intents:
        print(f"| {i} | " + " | ".join(str(s["by_intent"].get(i, "")) for _, s in rows) + " |")


def rescore(paths: list[str]) -> None:
    """Regrade saved answers against the current dataset rules, without calling any model.

    results/<name>_<split>.jsonl holds the answers in row order (first n rows of the split);
    latency, token usage and cost carry over unchanged."""
    for p in paths:
        jl = Path(p).with_suffix(".jsonl")
        old = json.loads(jl.with_suffix(".json").read_text(encoding="utf-8"))
        saved = [json.loads(line) for line in jl.open(encoding="utf-8")]
        split = old["split"]
        rows = [json.loads(line) for line in (DATA / f"{split}_eval.jsonl").open(encoding="utf-8")][: len(saved)]
        if [r["meta"]["question"] for r in rows] != [g["question"] for g in saved]:
            sys.exit(f"{jl.name}: saved questions don't match {split}; rerun the model instead.")
        answers = iter([(g["answer"], g["ms"], g["usage"]) for g in saved])
        summary, graded = evaluate(rows, lambda _row: next(answers), 1)
        for k in ("cost_per_1k_answers_usd", "system", "model", "split", "snapshot"):
            if k in old:
                summary[k] = old[k]
        jl.with_suffix(".json").write_text(json.dumps(summary, indent=1))
        with jl.open("w", encoding="utf-8") as f:
            for g in graded:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"{jl.stem:40s} pass {old['pass_rate']:5.1f} -> {summary['pass_rate']:5.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["gold", "baseline", "model", "serverless"])
    ap.add_argument("--split", action="append", help="test_real | test_synth | val (repeatable)")
    ap.add_argument("--model", help="model id; for --system serverless, the base model")
    ap.add_argument("--snapshot", help="--system serverless: `latest`, a checkpoint name (fw-ep1) "
                                       "or a path from results/serverless_run.json")
    ap.add_argument("--name", help="label for the results files")
    ap.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (quick smoke test)")
    ap.add_argument("--no-think", action="store_true", help="append /no_think (Qwen3 hybrid models)")
    ap.add_argument("--reasoning-effort", help="sent as reasoning_effort, e.g. none or low (reasoning models)")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--rescore", nargs="+", help="regrade saved results/<name>_<split>.json files, no model calls")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if args.rescore:
        rescore(args.rescore)
        return
    if not args.system or not args.split:
        ap.error("--system and --split are required (or use --compare)")

    if args.system == "gold":
        runner, workers = run_gold, 1
    elif args.system == "baseline":
        runner, workers = run_baseline, 1
    elif args.system == "serverless":
        # Fireworks serverless training pool: base model or a fine-tuned snapshot, no deployment.
        from serverless import BASE_MODEL, make_sampler_runner, snapshot_path
        args.model = args.model or BASE_MODEL
        args.snapshot = snapshot_path(args.snapshot)
        runner, workers = make_sampler_runner(args.model, args.snapshot, args.max_tokens), args.workers
    else:
        if not args.model:
            ap.error("--model is required for --system model")
        runner, workers = make_model_runner(args), args.workers
    name = args.name or args.system
    RESULTS.mkdir(exist_ok=True)

    for split in args.split:
        rows = [json.loads(l) for l in (DATA / f"{split}_eval.jsonl").open(encoding="utf-8")]
        if args.limit:
            rows = rows[: args.limit]
        summary, graded = evaluate(rows, runner, workers)
        if args.price_in or args.price_out:
            summary["cost_per_1k_answers_usd"] = round(
                1000 * (summary["tokens_in_per_answer"] * args.price_in
                        + summary["tokens_out_per_answer"] * args.price_out) / 1e6, 4)
        summary.update(system=args.system, model=args.model, split=split, snapshot=args.snapshot)
        out = RESULTS / f"{name}_{split}"
        out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
        with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for g in graded:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"\n== {name} on {split} (n={summary['n']})")
        for k in ("pass_rate", "core_pass_rate", "honesty_pass_rate", "false_promise_rate", "made_up_fact_rate",
                  "missing_fact_rate", "latency_ms_p50", "latency_ms_p95"):
            print(f"  {k:22s} {summary[k]}")
        worst = sorted(summary["by_intent"].items(), key=lambda kv: kv[1])[:5]
        print("  weakest intents:", ", ".join(f"{k} {v}%" for k, v in worst))


if __name__ == "__main__":
    main()
