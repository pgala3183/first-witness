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
- Latency is measured with 8 parallel requests, so it reads higher than a single live call.

To refresh: run the benchmark, then copy `finetune/results/benchmark.md` and the
`bench-*` / `baseline_*` files here.
