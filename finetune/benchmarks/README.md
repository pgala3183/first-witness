# Benchmark results

Snapshot of `finetune/results/` (which is gitignored scratch output), committed
so the numbers behind the submission table can be checked.

- **[benchmark.md](benchmark.md)**: the comparison table, from `python finetune/benchmark.py report`.
- **`results/<run>_<split>.json`**: summary metrics per model and test set.
- **`results/<run>_<split>.jsonl`**: every question, answer, reference answer, and why it passed or failed.

## How these were run (Nov 2, 2026)

- Prompted, no fine-tune: the same system prompt and held-out questions for every model.
- 150 rows each from `test_real` (real clip seconds never trained on) and `test_synth`
  (campus names never trained on). The keyword router row uses all 450 `test_real` rows.
- Reasoning off (`reasoning_effort: none`) for Kimi K3 and DeepSeek V4.1 Flash. GLM 5.3 Flash
  can't turn thinking off, so it ran at `low` with a 1,024-token limit.
- Qwen3.6 35B A3B and Qwen3.8 27B ran on Fireworks' serverless training pool (`finetune/serverless.py`),
  with thinking turned off in the chat template. No deployment.
- Latency is measured with 8 parallel requests, so it reads higher than a single live call.
- The "identity" question now passes on "First Witness", "automated" or "camera system". It used to require
  the exact product name, which the system prompt never asks for, so every model scored about 0% on it.
  All saved answers were regraded with `eval.py --rescore` (no model calls); the reference answers still score 100%.

To refresh: run the benchmark, then copy `finetune/results/benchmark.md` and the
`bench-*` / `baseline_*` files here.

## Is the gap real? (`stats/`)

`finetune/stats.py` on the files above (McNemar + 5,000-rep cluster bootstrap; older
result files are paired with the eval set by position, checked question by question).
All comparisons use the same 150 rows; on those rows the keyword router scores **61.3%**
(55.3% is its score on all 450 `test_real` rows).

- [stats/vs_router_real.md](stats/vs_router_real.md): every prompted model beats the router (all p < 0.05).
- [stats/qwen36_vs_frontier_real.md](stats/qwen36_vs_frontier_real.md): Qwen3.6 vs the frontier models on real seconds.
  Kimi K3 (+11.3) and DeepSeek (+6.0) are really ahead; GLM's +4.7 could be luck.
- [stats/qwen36_vs_frontier_synth.md](stats/qwen36_vs_frontier_synth.md): on unseen campuses all three frontier
  models are really ahead (+11 to +14). This is the gap the fine-tune has to close.

[tradeoff.png](tradeoff.png) (`python3 finetune/tradeoff.py`): pass rate against p95 latency and cost.
Only GLM 5.3 Flash and the two Qwen candidates fit the 1.5 s call budget.
