"""Is the gain real? McNemar + cluster bootstrap on paired eval results.

Compares a BEFORE run against one or more AFTER runs on the same split, using the
per-answer files eval.py writes (results/<name>_<split>.jsonl). Answers pair up by
row id, so every run must cover the same rows (same split, same --limit).

  python3 finetune/stats.py results/baseline_test_real.jsonl \\
      results/base-4b_test_real.jsonl results/ft-4b_test_real.jsonl

Writes results/stats_<split>.md (table) and, if matplotlib is installed,
results/stats_<split>_mcnemar.png and results/stats_<split>_bootstrap.png.

McNemar: of the questions exactly one system got right, does the AFTER system win
more than a coin flip would? Gives a p-value (two-sided exact binomial).

Cluster bootstrap: resample whole clusters (one Seville second, or one synthetic
snapshot: questions about the same facts aren't independent), 5,000 times, and
take the middle 95% of the pass-rate gaps. If it excludes zero, the gain holds on
a different draw of test cases.

Call a gain real only when both agree.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load(path: str) -> dict[int, dict]:
    """Rows by id. Older result files (no id/cluster) are matched to the eval set
    by position, and only if every question matches its dataset row exactly."""
    lines = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    if all("id" in r for r in lines):
        return {r["id"]: r for r in lines}
    from eval import cluster_key  # same cluster definition eval.py writes

    data_file = HERE / "data" / f"{split_of(path)}_eval.jsonl"
    if not data_file.is_file():
        sys.exit(f"{path}: rows have no id and {data_file} is missing. Run finetune/build_dataset.py.")
    metas = [json.loads(l)["meta"] for l in data_file.open(encoding="utf-8")]
    rows = {}
    for i, r in enumerate(lines):
        if i >= len(metas) or r.get("question") != metas[i]["question"]:
            sys.exit(f"{path}: row {i} doesn't match {data_file.name}; re-run eval.py to get ids.")
        rows[i] = {**r, "id": i, "cluster": cluster_key(metas[i])}
    return rows


def name_of(path: str) -> str:
    stem = Path(path).stem
    return stem.rsplit("_test_", 1)[0] if "_test_" in stem else stem


def split_of(path: str) -> str:
    stem = Path(path).stem
    return "test_" + stem.rsplit("_test_", 1)[1] if "_test_" in stem else "split"


def mcnemar(before: list[bool], after: list[bool]) -> tuple[int, int, float]:
    """(only_after_right, only_before_right, two-sided exact p)."""
    b_only = sum(1 for x, y in zip(before, after) if x and not y)
    a_only = sum(1 for x, y in zip(before, after) if y and not x)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return a_only, b_only, min(1.0, 2 * tail)


def cluster_bootstrap(clusters: list[str], before: list[bool], after: list[bool],
                      reps: int, seed: int) -> tuple[float, float, float, int]:
    """(observed gap pts, ci_low, ci_high, reps where AFTER > BEFORE)."""
    by = defaultdict(lambda: [0, 0, 0])  # n, before_right, after_right
    for c, x, y in zip(clusters, before, after):
        by[c][0] += 1
        by[c][1] += x
        by[c][2] += y
    units = list(by.values())
    n = sum(u[0] for u in units)
    observed = 100 * (sum(u[2] for u in units) - sum(u[1] for u in units)) / n
    rng = random.Random(seed)
    gaps = []
    for _ in range(reps):
        pick = [units[rng.randrange(len(units))] for _ in units]
        m = sum(u[0] for u in pick)
        gaps.append(100 * (sum(u[2] for u in pick) - sum(u[1] for u in pick)) / m)
    gaps.sort()
    lo = gaps[int(0.025 * reps)]
    hi = gaps[int(0.975 * reps) - 1]
    return observed, lo, hi, sum(g > 0 for g in gaps)


def verdict(p: float, lo: float, hi: float) -> str:
    sig_m = p < 0.05
    sig_b = lo > 0 or hi < 0
    if sig_m and sig_b:
        return "real gain" if lo > 0 else "real loss"
    if not sig_m and not sig_b:
        return "could be luck"
    return "tests disagree: trust the bootstrap"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", help="results/<baseline>_<split>.jsonl")
    ap.add_argument("after", nargs="+", help="one or more results/<system>_<split>.jsonl")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", help="output stem (default results/stats_<split>)")
    args = ap.parse_args()

    base = load(args.before)
    base_name = name_of(args.before)
    split = split_of(args.before)
    out_stem = Path(args.out) if args.out else RESULTS / f"stats_{split}"
    rows = []
    for path in args.after:
        aft = load(path)
        ids = sorted(set(base) & set(aft))
        if len(ids) < len(base) or len(ids) < len(aft):
            print(f"warning: {name_of(path)} shares {len(ids)} of {len(base)}/{len(aft)} rows; "
                  "comparing only those", file=sys.stderr)
        b = [bool(base[i]["ok"]) for i in ids]
        a = [bool(aft[i]["ok"]) for i in ids]
        cl = [base[i]["cluster"] for i in ids]
        a_only, b_only, p = mcnemar(b, a)
        gap, lo, hi, wins = cluster_bootstrap(cl, b, a, args.reps, args.seed)
        rows.append({
            "system": name_of(path), "n": len(ids), "clusters": len(set(cl)),
            "before": 100 * sum(b) / len(ids), "after": 100 * sum(a) / len(ids),
            "gap": gap, "ci_lo": lo, "ci_hi": hi, "boot_wins": wins,
            "only_after": a_only, "only_before": b_only, "p": p,
            "verdict": verdict(p, lo, hi),
        })

    lines = [
        f"# Is the gain real? ({split}, vs `{base_name}`)",
        "",
        f"McNemar = exact two-sided binomial on the disagreements. Bootstrap = {args.reps:,} "
        "resamples of whole clusters (one Seville second / one synthetic snapshot), 95% interval.",
        "",
        "| System | n | Clusters | Before | After | Gap (pts) | 95% CI | Wins / losses | McNemar p | Bootstrap ahead | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['system']} | {r['n']} | {r['clusters']} | {r['before']:.1f}% | {r['after']:.1f}% | "
            f"{r['gap']:+.1f} | {r['ci_lo']:+.1f} to {r['ci_hi']:+.1f} | {r['only_after']} / {r['only_before']} | "
            f"{r['p']:.2g} | {r['boot_wins']:,} / {args.reps:,} | {r['verdict']} |")
    few = [r["system"] for r in rows if r["clusters"] < 20]
    if few:
        lines += ["", f"Fewer than 20 clusters for {', '.join(few)}: the bootstrap interval is unreliable; "
                  "lean on McNemar there."]
    md = "\n".join(lines) + "\n"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".md").write_text(md)
    out_stem.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print(md)
    plot(rows, base_name, split, out_stem)


def plot(rows: list[dict], base_name: str, split: str, out_stem: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipped charts)")
        return
    names = [r["system"] for r in rows]
    ys = range(len(rows))

    fig, ax = plt.subplots(figsize=(7, 0.7 * len(rows) + 1.6))
    ax.barh([y + 0.18 for y in ys], [r["only_after"] for r in rows], height=0.34,
            color="#2a7de1", label="only this system right")
    ax.barh([y - 0.18 for y in ys], [r["only_before"] for r in rows], height=0.34,
            color="#b0b6bf", label=f"only {base_name} right")
    for y, r in zip(ys, rows):
        ax.text(max(r["only_after"], r["only_before"]) + 1, y, f"p = {r['p']:.2g}", va="center", fontsize=9)
    ax.set_yticks(list(ys), names)
    ax.set_xlabel("Questions where exactly one system was right")
    ax.set_title(f"McNemar: who wins the disagreements? ({split})")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_stem.parent / f"{out_stem.name}_mcnemar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.6))
    for y, r in zip(ys, rows):
        color = "#2a7de1" if r["ci_lo"] > 0 else ("#d9534f" if r["ci_hi"] < 0 else "#8a929c")
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color, lw=3)
        ax.plot(r["gap"], y, "o", color=color, ms=8)
        ax.text(r["ci_hi"] + 0.5, y, f"{r['gap']:+.1f} pts", va="center", fontsize=9)
    ax.axvline(0, color="#333", lw=1, ls="--")
    ax.set_yticks(list(ys), names)
    ax.set_xlabel(f"Pass-rate gap vs {base_name} (points), 95% cluster bootstrap")
    ax.set_title(f"Would the gain survive a different test set? ({split})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_stem.parent / f"{out_stem.name}_bootstrap.png", dpi=160)
    plt.close(fig)
    print(f"charts: {out_stem.name}_mcnemar.png, {out_stem.name}_bootstrap.png")


if __name__ == "__main__":
    main()
