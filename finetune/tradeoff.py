"""Tradeoff charts: pass rate vs p95 latency (with the 1.5 s call budget) and vs cost.

  python3 finetune/tradeoff.py                       # reads finetune/benchmarks/results
  python3 finetune/tradeoff.py --dir finetune/results --out finetune/results/tradeoff

Pass rate is the mean of test_real and test_synth when a model has both, else test_real.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUDGET_MS = 1500


def load(d: Path) -> list[dict]:
    by = defaultdict(dict)
    for f in sorted(d.glob("*.json")):
        name, _, split = f.stem.rpartition("_test_")
        if not name or name.startswith("gold"):
            continue
        by[name.removeprefix("bench-")]["test_" + split] = json.loads(f.read_text())
    out = []
    for name, s in by.items():
        runs = [r for r in (s.get("test_real"), s.get("test_synth")) if r and r.get("n")]
        if not runs:
            continue
        p95 = [r["latency_ms_p95"] for r in runs if r.get("latency_ms_p95") is not None]
        cost = [r["cost_per_1k_answers_usd"] for r in runs if r.get("cost_per_1k_answers_usd") is not None]
        out.append({
            "name": name,
            "pass": sum(r["pass_rate"] for r in runs) / len(runs),
            "splits": "+".join(k.removeprefix("test_") for k in ("test_real", "test_synth") if s.get(k, {}).get("n")),
            "p95": max(p95) if p95 else None,
            "cost": sum(cost) / len(cost) if cost else 0.0,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "benchmarks" / "results"))
    ap.add_argument("--out", default=str(HERE / "benchmarks" / "tradeoff"))
    args = ap.parse_args()
    rows = load(Path(args.dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, label in ((axes[0], "p95", "p95 latency per answer (ms)"),
                           (axes[1], "cost", "Cost per 1,000 answers (USD)")):
        for r in rows:
            if r[key] is None:
                continue
            ok = r["p95"] is not None and r["p95"] <= BUDGET_MS
            ax.scatter(r[key], r["pass"], s=70, color="#2a7de1" if ok else "#b0b6bf", zorder=3)
            ax.annotate(f"{r['name']} ({r['splits']})", (r[key], r["pass"]), textcoords="offset points",
                        xytext=(7, -3), fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("Pass rate (%)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axvline(BUDGET_MS, color="#d9534f", ls="--", lw=1)
    axes[0].text(BUDGET_MS, axes[0].get_ylim()[0], " 1.5 s call budget", color="#d9534f", fontsize=8, va="bottom")
    fig.suptitle("Call agent: quality vs speed and cost (blue = fits the live-call budget)")
    fig.tight_layout()
    out = Path(args.out).with_suffix(".png")
    fig.savefig(out, dpi=160)
    print(f"saved {out}")
    for r in sorted(rows, key=lambda r: -r["pass"]):
        print(f"{r['name']:<28} {r['splits']:<10} pass {r['pass']:5.1f}%  p95 {r['p95']}  $/1k {r['cost']:.3f}")


if __name__ == "__main__":
    main()
