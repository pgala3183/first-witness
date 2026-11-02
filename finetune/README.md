# First Witness call agent: fine-tuning data + eval

A small model (Qwen3-4B-Instruct-2507) learns to answer a dispatcher's
questions on the live call **using only the facts First Witness actually has**:
camera facts, officer accept status, and zone-alert status. It replaces the
keyword router in `services/voice/agent.py`, which breaks on paraphrases and
knows nothing about officer accepts.

## The honesty rules it is trained and graded on

1. No number, weapon, place, building or name that isn't in the facts.
2. Injuries, identity, intent, sounds: say it can't verify.
3. "An officer is coming" **only** after an officer has accepted, with name + ETA. Never "they've arrived".
4. "A warning went out" only if it was sent. Never "doors are locked".
5. Out of view: say so and use the last sighting.

## What each file does

| File | Role |
|---|---|
| `facts.py` | Builds fact snapshots: **real** ones from the Seville scene script (340 s) + camera map, and **synthetic** ones (other buildings, campuses, fights, medical, theft, people, clothing). Adds random officer and alert status. |
| `dialog.py` | 20 question types × 8 phrasings (2 per type held out for testing only), speech-recognition noise, and the rule-written correct answer for every snapshot. |
| `checker.py` | Code that grades any answer: false promises, made-up facts, missing facts, too long. Used to filter training data **and** to score models. |
| `build_dataset.py` | Writes `data/train_sft.jsonl` + `data/val_sft.jsonl` (Fireworks format) and the eval sets. |
| `eval.py` | Scores the current router (`baseline`), any Fireworks model (`model`), or the reference answers (`gold`, should be 100%). `--compare` prints the results table. |
| `teacher_paraphrase.py` | Optional: a large Fireworks model writes more question phrasings (questions only, never answers). |
| `benchmark.py` + `benchmark_models.json` | Runs 8 open models (prompted) through `eval.py` and picks the base model to fine-tune. |
| `run_fireworks.sh` | `check → data → train → status → deploy` with `firectl`. |

## Why 340 seconds isn't the training set

Consecutive seconds are near-duplicates (about 150 distinct count/weapon states,
one building, one set of clothes). Training only on them would teach the model
MacQuarrie Hall instead of reading the facts it's given. So:

| Split | Rows | What's in it |
|---|---|---|
| train | 3,300 | 78% synthetic + 22% real Seville seconds 0–229, training phrasings, 30% speech noise |
| val | 300 | new synthetic + real snapshots (Fireworks evaluation set) |
| **test_real** | 450 | real Seville seconds **230–339, never trained on**, held-out phrasings, 40% noise |
| **test_synth** | 450 | **unseen campus names**, held-out phrasings, 40% noise |

No prompt is shared between train and any test split (checked in code).

## Results so far (before any training)

| System | test_real pass | core questions | honesty questions | false promises | made-up facts |
|---|---|---|---|---|---|
| Reference answers (sanity) | 100% | 100% | 100% | 0% | 0% |
| **Current keyword router** | **55.3%** | **63.0%** | **38.1%** | 0% | 0% |

"Core" = the questions the current router was built for. "Honesty" = officer and
alert questions it can't answer yet. It scores 0% on identity, movement and
repeat questions when they're phrased differently. (Its made-up facts were 4.2%,
mostly "No one appears injured", until that line became "I can't verify
injuries from the cameras.")

## Picking the base model

`benchmark.py` runs 8 tunable open models from Fireworks, all prompted with the
same system prompt and none fine-tuned, on both test sets:

- **References** (serverless, billed per token): Kimi K3, DeepSeek V4.1 Flash, GLM 5.3 Flash.
  These are the frontier models we compare against.
- **Candidates** (need an on-demand deployment): Qwen3.8 27B, Qwen3.6 35B A3B,
  Gemma 4 26B A4B, Qwen3.5 9B, Ministral 3 3B. All are small enough to fine-tune and to host.

The pick is the candidate with the highest pass rate whose p95 latency fits the
call's 1.5 s budget. The top overall score doesn't decide it: a frontier model can't
answer a live call in time or be hosted cheaply. The pick's gap to the best
reference is what the fine-tune has to close.

```bash
python3 finetune/benchmark.py resolve            # check the model ids exist
python3 finetune/benchmark.py list               # which need a deployment
python3 finetune/benchmark.py run --limit 150    # quick pass, then without --limit
python3 finetune/benchmark.py report             # table + pick -> results/benchmark.md
```

Delete each candidate's deployment once its run finishes: they're billed while up.

## Fine-tuning on shared GPUs (no deployment)

Fireworks won't create dedicated deployments without a payment method on file,
so we train on **serverless training**: a shared LoRA trainer pool, billed per
token. `serverless.py` renders each example with the model's own chat template
(thinking off), trains only on the answer tokens, and saves a sampler checkpoint
per epoch. The same session samples those checkpoints, so `eval.py` grades the
base model and the fine-tune with no deployment.

Base model: **Qwen 3.6 35B A3B**, the cheapest on the pool (train $2.14/1M tokens).
Our data is 1.56M tokens per epoch, so 2 epochs cost about $6.70.

```bash
pip install "fireworks-ai[training]>=1.2.29"     # then re-pin numpy==2.3.5 (requirements.txt)
python finetune/serverless.py check              # 3 answers from the base model: access + credits
python finetune/serverless.py train --limit-train 64 --yes    # paid smoke run, about $0.13
python finetune/serverless.py train              # full run: asks before spending
python finetune/eval.py --system serverless --split test_real --split test_synth --name sl-base-q36
python finetune/eval.py --system serverless --snapshot latest \
    --split test_real --split test_synth --name sl-ft-q36
python finetune/serverless.py promote --as first-witness-call-q36   # register as a model (deploying it needs a card)
```

`benchmark.py run --only qwen3p6-35b-a3b qwen3p8-27b` benchmarks both pool models the same way.

## Build-day runbook

```bash
export FIREWORKS_API_KEY=...
python3 finetune/build_dataset.py                       # ~5 s
python3 finetune/eval.py --system baseline --split test_real
python3 finetune/benchmark.py run && python3 finetune/benchmark.py report   # pick the base

# Prompted 4B, no fine-tuning (needs a deployment of the base model, or use a serverless model id)
python3 finetune/eval.py --system model --model <base-4b-id> --split test_real --split test_synth --name base-4b

bash finetune/run_fireworks.sh check
bash finetune/run_fireworks.sh data
bash finetune/run_fireworks.sh train
bash finetune/run_fireworks.sh status
bash finetune/run_fireworks.sh deploy

python3 finetune/eval.py --system model --model <fine-tuned-model-id> --split test_real --split test_synth --name ft-4b
python3 finetune/eval.py --system model --model <large-teacher-id> --split test_real --split test_synth --name teacher
python3 finetune/eval.py --compare finetune/results/*_test_real.json
```

### Is the gain real?

```bash
# Every system against the router on test_real (the router can't answer test_synth: n=0)
python3 finetune/stats.py finetune/results/baseline_test_real.jsonl \
    finetune/results/base-4b_test_real.jsonl finetune/results/ft-4b_test_real.jsonl
# The fine-tune against its own prompted base: the before/after the rubric asks for
python3 finetune/stats.py finetune/results/base-4b_test_real.jsonl finetune/results/ft-4b_test_real.jsonl --out finetune/results/stats_ft_vs_base_real
python3 finetune/stats.py finetune/results/base-4b_test_synth.jsonl finetune/results/ft-4b_test_synth.jsonl --out finetune/results/stats_ft_vs_base_synth
```

McNemar (exact, on the disagreements) and a cluster bootstrap (resamples whole
Seville seconds / synthetic snapshots, 5,000 times). A gain is real only when both
agree. Writes a markdown table and two charts per run. Results files from before
`id`/`cluster` were added must be re-run through `eval.py`.

Add `--price-in/--price-out` (USD per 1M tokens) to get cost per 1,000 answers.
Read `results/<name>_<split>.jsonl` for every failed answer and why it failed.
Add `--no-think` if you evaluate a hybrid Qwen3 model that thinks by default.

## Ideas if there's time

- More phrasings: `teacher_paraphrase.py`, then rebuild and retrain.
- Reinforcement fine-tuning with `checker.score()` as the reward.
- Wire the model into `answer_dispatcher` (the client is `services/voice/call_model.py`; it owns the system prompt and message format this kit trains on).
