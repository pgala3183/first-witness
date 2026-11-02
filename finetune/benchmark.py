"""Benchmark open models on the call-agent task, then pick the one to fine-tune.

Every model answers the same held-out questions with the same system prompt
(prompted, no fine-tune) and is graded by eval.py. References are frontier
models we compare against; candidates are small enough to fine-tune and to
answer inside the live call's 1.5 s budget. The pick is the best candidate
that meets that budget, not simply the top score: Kimi K3 may win outright,
but it can't answer a phone call in time or be hosted cheaply.

Usage (needs FIREWORKS_API_KEY):
  python finetune/benchmark.py list                  # models, and which need a deployment
  python finetune/benchmark.py resolve               # check every id exists on Fireworks
  python finetune/benchmark.py run --limit 150       # quick pass: 150 rows per split
  python finetune/benchmark.py run                   # full: 450 + 450 rows per model
  python finetune/benchmark.py run --only kimi-k3 glm-5p3-flash
  python finetune/benchmark.py report                # table + the model to fine-tune

Writes results/bench-<slug>_<split>.json (+ .jsonl) and results/benchmark.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONFIG = HERE / "benchmark_models.json"
SPLITS = ("test_real", "test_synth")
# The live call gives up after this long (CS_CALL_MODEL_TIMEOUT_S).
BUDGET_MS = 1500.0


def models(only: list[str] | None = None) -> list[dict[str, Any]]:
    ms = json.loads(CONFIG.read_text())["models"]
    if only:
        ms = [m for m in ms if m["slug"] in only]
        missing = set(only) - {m["slug"] for m in ms}
        if missing:
            sys.exit(f"unknown slug(s): {', '.join(sorted(missing))}")
    return ms


def target(m: dict[str, Any]) -> str | None:
    """The model string to send, or None if it needs a deployment it doesn't have."""
    if m.get("serverless"):
        return m["id"]
    dep = (m.get("deployment") or "").strip()
    return f"{m['id']}#{dep}" if dep else None


# ---------------------------------------------------------------- list / resolve

def cmd_list(_: argparse.Namespace) -> None:
    for m in models():
        how = "serverless" if m.get("serverless") else (
            "deployment set" if target(m) else "NEEDS DEPLOYMENT")
        print(f"{m['slug']:22s} {m['role']:9s} {how:17s} {m['id']}")
    print("\nCreate a deployment:  firectl deployment create <model-id> --min-replica-count 1")
    print("Then put accounts/<you>/deployments/<id> in its \"deployment\" field. Delete it after the run.")


def _list_fireworks(key: str) -> list[dict[str, Any]]:
    out, token = [], ""
    for _ in range(20):
        url = "https://api.fireworks.ai/v1/accounts/fireworks/models?pageSize=200" + (
            f"&pageToken={token}" if token else "")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            page = json.loads(r.read())
        out += page.get("models", [])
        token = page.get("nextPageToken") or ""
        if not token:
            break
    return out


def cmd_resolve(_: argparse.Namespace) -> None:
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        sys.exit("Set FIREWORKS_API_KEY.")
    try:
        catalog = _list_fireworks(key)
    except (urllib.error.URLError, ValueError) as e:
        sys.exit(f"Could not list Fireworks models ({e}). Use: firectl model list -a fireworks")
    ids = {c["name"]: c for c in catalog}
    bad = 0
    for m in models():
        if m["id"] in ids:
            c = ids[m["id"]]
            tunable = c.get("tunable", "?")
            print(f"ok    {m['slug']:22s} {m['id']}  (tunable: {tunable})")
            continue
        bad += 1
        words = [w for w in m["name"].lower().replace(".", " ").split() if len(w) > 1]
        near = [c["name"] for c in catalog
                if sum(w in (c.get("displayName", "") + " " + c["name"]).lower() for w in words) >= max(2, len(words) - 1)]
        print(f"MISS  {m['slug']:22s} {m['id']}")
        for n in near[:5]:
            print(f"        try {n}")
    print(f"\n{bad} id(s) to fix in {CONFIG.name}." if bad else "\nAll ids found.")


# ---------------------------------------------------------------- run

def _eval(m: dict[str, Any], model: str, name: str, splits: tuple[str, ...], limit: int,
          workers: int) -> int:
    cmd = [sys.executable, str(HERE / "eval.py"), "--system", "model", "--model", model, "--name", name,
           "--workers", str(workers), "--max-tokens", "120"]
    for s in splits:
        cmd += ["--split", s]
    if limit:
        cmd += ["--limit", str(limit)]
    if m.get("no_think"):
        cmd.append("--no-think")
    if m.get("reasoning_effort"):
        cmd += ["--reasoning-effort", m["reasoning_effort"]]
    if m.get("price_in") or m.get("price_out"):
        cmd += ["--price-in", str(m.get("price_in", 0)), "--price-out", str(m.get("price_out", 0))]
    return subprocess.call(cmd)


def cmd_run(args: argparse.Namespace) -> None:
    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit("Set FIREWORKS_API_KEY.")
    for m in models(args.only):
        model = target(m)
        if model is None:
            print(f"\n-- skip {m['slug']}: needs a deployment (see `benchmark.py list`)")
            continue
        print(f"\n-- {m['name']}  ({model})")
        # 5-row smoke test: a wrong id or a cold deployment fails here, not 900 rows later.
        _eval(m, model, f"smoke-{m['slug']}", ("test_real",), 5, 5)
        smoke = json.loads((RESULTS / f"smoke-{m['slug']}_test_real.json").read_text(encoding="utf-8"))
        if smoke.get("error_rate", 0) >= 40:
            first = (RESULTS / f"smoke-{m['slug']}_test_real.jsonl").read_text(encoding="utf-8").splitlines()[0]
            print(f"   endpoint failing ({smoke['error_rate']}% errors), skipped: {json.loads(first)['answer'][:160]}")
            continue
        if smoke.get("empty_rate", 0) >= 40:
            print(f"   {smoke['empty_rate']}% blank answers: it's thinking through max_tokens. Set "
                  f"\"reasoning_effort\" for {m['slug']} in {CONFIG.name}; skipped.")
            continue
        _eval(m, model, f"bench-{m['slug']}", SPLITS, args.limit, args.workers)
    print("\nNext: python finetune/benchmark.py report")


# ---------------------------------------------------------------- report

def _load(name: str, split: str) -> dict[str, Any] | None:
    p = RESULTS / f"{name}_{split}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _both(name: str) -> dict[str, Any] | None:
    """test_real + test_synth, weighted by rows."""
    parts = [s for s in (_load(name, sp) for sp in SPLITS) if s]
    if not parts:
        return None
    n = sum(p["n"] for p in parts)

    def w(k: str, digits: int = 1) -> float | None:
        vals = [(p[k], p["n"]) for p in parts if p.get(k) is not None]
        return round(sum(v * c for v, c in vals) / sum(c for _, c in vals), digits) if vals else None

    out = {k: w(k) for k in ("pass_rate", "honesty_pass_rate", "false_promise_rate", "made_up_fact_rate",
                             "missing_fact_rate", "error_rate", "empty_rate")}
    out["cost_per_1k_answers_usd"] = w("cost_per_1k_answers_usd", 4)
    out["n"] = n
    out["latency_ms_p50"] = max((p.get("latency_ms_p50") or 0) for p in parts) or None
    out["latency_ms_p95"] = max((p.get("latency_ms_p95") or 0) for p in parts) or None
    out["real"] = (_load(name, "test_real") or {}).get("pass_rate")
    out["synth"] = (_load(name, "test_synth") or {}).get("pass_rate")
    return out


def _cell(v: Any, unit: str = "") -> str:
    return "-" if v is None else f"{v}{unit}"


def cmd_report(args: argparse.Namespace) -> None:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for m in models():
        s = _both(f"bench-{m['slug']}")
        if s:
            rows.append((m, s))
    if not rows:
        sys.exit("No benchmark results yet: run `python finetune/benchmark.py run` first.")
    rows.sort(key=lambda r: -(r[1]["pass_rate"] or 0))

    lines = ["# Call-agent benchmark (prompted, no fine-tune)", "",
             f"Same system prompt and held-out questions for every model. Pass = no false promise, "
             f"no made-up fact, and the required facts present. Latency budget: {BUDGET_MS:.0f} ms p95.", "",
             "| Model | Role | Pass (both) | test_real | test_synth | Honesty | False promises | Made-up facts "
             "| p50 ms | p95 ms | $/1k answers | Errors | Blank |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = _load("baseline", "test_real")
    if base:
        lines.append(f"| Keyword router (today) | baseline | - | {base['pass_rate']}% | n/a | "
                     f"{base['honesty_pass_rate']}% | {base['false_promise_rate']}% | {base['made_up_fact_rate']}% "
                     f"| {_cell(base.get('latency_ms_p50'))} | {_cell(base.get('latency_ms_p95'))} | $0 | - | - |")
    for m, s in rows:
        cost = _cell(s["cost_per_1k_answers_usd"]) if m.get("serverless") else "dedicated GPU"
        lines.append(
            f"| {m['name']} | {m['role']} | **{_cell(s['pass_rate'], '%')}** | {_cell(s['real'], '%')} "
            f"| {_cell(s['synth'], '%')} | {_cell(s['honesty_pass_rate'], '%')} "
            f"| {_cell(s['false_promise_rate'], '%')} | {_cell(s['made_up_fact_rate'], '%')} "
            f"| {_cell(s['latency_ms_p50'])} | {_cell(s['latency_ms_p95'])} | {cost} "
            f"| {_cell(s['error_rate'], '%')} | {_cell(s['empty_rate'], '%')} |")

    ok = [(m, s) for m, s in rows
          if m["role"] == "candidate" and (s["error_rate"] or 0) < 5 and (s["empty_rate"] or 0) < 5]
    fast = [(m, s) for m, s in ok if (s["latency_ms_p95"] or 1e9) <= args.budget_ms]
    lines += ["", "## Pick"]
    if not ok:
        lines.append("No candidate has clean results yet: deploy the candidates and run them.")
    else:
        pool = fast or ok
        best_m, best_s = pool[0]
        top_ref = next(((m, s) for m, s in rows if m["role"] == "reference"), None)
        why = (f"highest pass rate ({best_s['pass_rate']}%) among candidates inside the "
               f"{args.budget_ms:.0f} ms p95 budget" if fast else
               f"highest pass rate ({best_s['pass_rate']}%) among candidates; none met the "
               f"{args.budget_ms:.0f} ms budget, so check latency on a dedicated deployment")
        lines.append(f"**Fine-tune {best_m['name']}** (`{best_m['id']}`): {why}.")
        if top_ref:
            gap = round((top_ref[1]["pass_rate"] or 0) - (best_s["pass_rate"] or 0), 1)
            lines.append(f"Gap to the best reference ({top_ref[0]['name']}, {top_ref[1]['pass_rate']}%): "
                         f"{gap} points. This gap is what the fine-tune has to close.")
        slow = [m["name"] for m, s in ok if (s["latency_ms_p95"] or 0) > args.budget_ms]
        if slow:
            lines.append(f"Over the latency budget: {', '.join(slow)}.")
    lines += ["", "Latency is measured with parallel requests, so it runs high; recheck the pick on "
              "its own deployment with `--workers 1`."]
    text = "\n".join(lines) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("resolve").set_defaults(fn=cmd_resolve)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="+", help="slugs to run (default: all)")
    r.add_argument("--limit", type=int, default=0, help="rows per split (0 = all 450)")
    r.add_argument("--workers", type=int, default=8)
    r.set_defaults(fn=cmd_run)
    rp = sub.add_parser("report")
    rp.add_argument("--budget-ms", type=float, default=BUDGET_MS)
    rp.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
