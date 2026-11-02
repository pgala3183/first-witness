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
| **Current keyword router** | **51.1%** | **56.9%** | **38.1%** | 0% | 4.2% |

"Core" = the questions the current router was built for. "Honesty" = officer and
alert questions it can't answer yet. The router's made-up facts are mostly
"No one appears injured," which the cameras can't know. It scores 0% on
identity, movement and repeat questions when they're phrased differently.

## Build-day runbook

```bash
export FIREWORKS_API_KEY=...
python3 finetune/build_dataset.py                       # ~5 s
python3 finetune/eval.py --system baseline --split test_real

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

Add `--price-in/--price-out` (USD per 1M tokens) to get cost per 1,000 answers.
Read `results/<name>_<split>.jsonl` for every failed answer and why it failed.
Add `--no-think` if you evaluate a hybrid Qwen3 model that thinks by default.

## Ideas if there's time

- More phrasings: `teacher_paraphrase.py`, then rebuild and retrain.
- Reinforcement fine-tuning with `checker.score()` as the reward.
- Wire the model into `answer_dispatcher` (the client is `services/voice/call_model.py`; it owns the system prompt and message format this kit trains on).
