# Benchmark results

Snapshot of `finetune/results/` (which is gitignored scratch output), committed
so the numbers behind the submission table can be checked.

- **[benchmark.md](benchmark.md)**: the comparison table, from `python finetune/benchmark.py report`.
- **`results/<run>_<split>.json`**: summary metrics per model and test set.
- **`results/<run>_<split>.jsonl`**: every question, answer, reference answer, and why it passed or failed.
- **[training_run.json](training_run.json)**: the fine-tune's settings, loss at every step and checkpoint ids.

## Headline (Nov 2, 2026)

Fine-tuning Qwen3.6 35B A3B with LoRA on 3,300 synthetic + real examples took it from
**77.7% to 95.7%**, ahead of every frontier model we tried, with **0% false promises**.

| Model | Pass | False promises | Made-up facts | Latency p50 / p95 (one call at a time) | $ / 1k answers |
|---|---|---|---|---|---|
| **First Witness (Qwen3.6 35B A3B + LoRA)** | **95.7%** | **0%** | 0.3% | **632 / 882 ms** | $0.23 |
| Kimi K3 | 90.3% | 0% | 3.3% | parallel only: 1,751 / 3,983 ms | $1.91 |
| DeepSeek V4.1 Flash | 87.7% | 0% | 0.7% | parallel only: 1,105 / 1,631 ms | $0.18 |
| GLM 5.3 Flash | 85.7% | 0.7% | 4.0% | parallel only: 773 / 1,468 ms | $0.09 |
| Qwen3.6 35B A3B, prompted | 77.7% | 0% | 1.4% | 584 / 890 ms | $0.23 |
| Keyword router (before) | 55.3% (test_real) | 0% | 0% | instant | $0 |

Pass rates use the same 150 test_real + 150 test_synth questions for every model.

**All 900 questions** (`sl-*` files), prompted vs fine-tuned Qwen3.6 35B A3B:

| Run | test_real (450) | test_synth (450) | Both | False promises |
|---|---|---|---|---|
| Prompted base | 83.6% | 71.3% | 77.4% | 0.1% |
| Fine-tuned, epoch 1 | 95.8% | 94.0% | 94.9% | 0% |
| **Fine-tuned, epoch 2 (used)** | **95.8%** | **94.4%** | **95.1%** | **0%** |

The biggest jump is on test_synth, the campuses the model never saw (71% to 94%): it learned
to read the facts it's given rather than memorize the demo building.

## How these were run

- Every model gets the same system prompt and held-out questions (question wordings never
  used in training, 40% with speech-recognition noise). test_real is clip seconds 230-339,
  never trained on; test_synth uses campus names never trained on.
- Reasoning off (`reasoning_effort: none`) for Kimi K3 and DeepSeek V4.1 Flash. GLM 5.3 Flash
  can't turn thinking off, so it ran at `low` with a 1,024-token limit.
- The Qwen models ran on Fireworks' serverless training pool (`finetune/serverless.py`), thinking
  off in the chat template, no deployment. Frontier models ran on Fireworks serverless inference.
- Table latency (`benchmark.md`) uses 8 parallel requests. The fine-tuned row there (p95 5.1 s) also
  includes the pool's queueing; the one-call-at-a-time numbers above (`lat-*` files, 40 questions)
  are what a live phone call sees.
- The "identity" question passes on "First Witness", "automated" or "camera system". It used to
  require the exact product name, which the system prompt never asks for, so every model scored
  about 0% on it. All saved answers were regraded with `eval.py --rescore` (no model calls); the
  reference answers still score 100%.

## Training

LoRA rank 16, learning rate 1e-4 (constant), batch 32, 2 epochs = 208 steps over 1.56M tokens per
epoch. Loss went from 1.21 to about 0.01 within the first epoch. Wall-clock 43 minutes (slower in
epoch 2 while evals shared the pool).

Fireworks billing record (`firectl billing get-usage --usage-type training`), both sessions on
NVIDIA GB300 in the serverless pool, billed as `SERVERLESS_TRAINING_TOKENS`:

| Session | Training tokens | Cost at $2.14 / 1M |
|---|---|---|
| `ts-4b9a196039fd4d81b654147ec1a8c01a` (64-example smoke test) | 60,668 | ~$0.13 |
| `ts-a90f06ad279043f9854e305ee95b1b16` (full run) | 3,111,942 | ~$6.66 |

Total Fireworks usage for the project, Nov 1-3 (training, all benchmarks, evals, live-call test): **$9.42**.

## Honest misses

- **Status questions (about 42%)**: for "Where are we at?" the model gives the location, not the full
  status update (people, weapons, officer) the reference expects. A fair reading, but less useful.
- **Route on unseen campuses (56%)**: it sometimes lists only the current place instead of the path.
- Made-up facts are rare (0.1% across 900 answers), and no answer promised help nobody accepted.

## Refresh

Run the benchmark (or `eval.py`), then copy `finetune/results/benchmark.md` and the matching
result files here.

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
